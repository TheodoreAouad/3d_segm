from time import time
import os
from os.path import join

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import matplotlib.pyplot as plt


# from deep_morpho.datasets.generate_forms2 import get_random_diskorect
# from deep_morpho.datasets.generate_forms3 import get_random_rotated_diskorect
from deep_morpho.datasets.multi_rect_dataset import MultiRectDatasetGenerator, MultiRectDataset
from deep_morpho.models import LightningBiMoNN, BiSE, COBiSE, BiSEC, COBiSEC
import deep_morpho.observables as obs
from general.nn.observables import CalculateAndLogMetrics
from general.utils import format_time, log_console, create_logger, save_yaml
from deep_morpho.metrics import dice
from deep_morpho.args import all_args
from general.code_saver import CodeSaver


def main(args, logger):
    if (args['dataset_path'] is not None) and (args['dataset_path'] != 'generate'):
        dataloader = MultiRectDataset.get_loader(
            batch_size=args['batch_size'], dataset_path=args['dataset_path'], do_load_in_ram=args['in_ram'],
            morp_operation=args['morp_operation'], logger=console_logger, n_inputs=args['n_inputs'],
            num_workers=args['num_workers']
        )
    else:
        dataloader = MultiRectDatasetGenerator.get_loader(
            batch_size=args['batch_size'],
            n_inputs=args['n_inputs'],
            random_gen_fn=args['random_gen_fn'],
            random_gen_args=args['random_gen_args'],
            morp_operation=args['morp_operation'].morp_fn,
            device=device,
            num_workers=args['num_workers']
    )

    metrics = {'dice': lambda y_true, y_pred: dice(y_true, y_pred, threshold=.5).mean()}

    observables = [
        obs.SaveLoss(),
        CalculateAndLogMetrics(
            metrics=metrics,
            keep_preds_for_epoch=False,
        ),
        obs.InputAsPredMetric(metrics),
        obs.PlotParametersDilation(freq=1),
        obs.PlotWeightsDilation(freq=args['freq_imgs']),
        obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        obs.PlotPreds(freq=args['freq_imgs']),
        obs.CountInputs(),
        obs.CheckMorpOperation(selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50),
        obs.PlotGradientBise(freq=args['freq_imgs']),
    ]

    xs = torch.tensor(np.linspace(-6, 6, 100)).detach()

    model = LightningBiMoNN(
        model_args={
            "kernel_size": [args['kernel_size'] for _ in range(args['n_atoms'])],
            "atomic_element": args["atomic_element"] if args["atomic_element"] != "conv" else "bise",
            "threshold_mode": args['threshold_mode'],
            "activation_P": args['activation_P'],
            "constant_activation_P": args['constant_activation_P'],
            "constant_weight_P": args['constant_weight_P'],
            "init_weight_identity": args["init_weight_identity"],
            "alpha_init": args["alpha_init"],
        },
        learning_rate=args['learning_rate'],
        loss=args['loss'],
        optimizer=args['optimizer'],
        output_dir="deep_morpho/results",
        observables=observables,
        do_thresh_penalization=args['do_thresh_penalization'],
        args_thresh_penalization=args['args_thresh_penalization'],
        first_batch_pen=args['first_batch_pen'],
    )
    if isinstance(model.model.layers[0], (BiSE, COBiSE, BiSEC, COBiSEC)):
        ys = model.model.layers[0].activation_threshold_fn(xs).detach()
        fig, ax = plt.subplots(); ax.plot(xs, ys); ax.set_title(args['threshold_mode'])
        logger.experiment.add_figure("threshold_fn", fig)

    model.to(device)

    logger.experiment.add_graph(model, torch.ones(1, 1, 50, 50).to(device))
    hyperparams = dict(
        **{f'{k}_{layer_idx}': -1 for k in [
            f"weights/sum_norm_weights",
            f"params/weight_P",
            f"params/activation_P",
            f"weights/bias",
        ] for layer_idx in range(len(model.model.layers))},
        **{f"metrics_batch/dice_train": 0},
    )

    if args['atomic_element'] in ['bisec', 'cobisec']:
        hyperparams.update(dict(
            **{f'{k}_{layer_idx}': -1 for k in ["weights/norm_alpha"] for layer_idx in range(len(model.model.layers))}
        ))

    if args["atomic_element"] == 'cobise':
        hyperparams.update(dict(
            **{f'metrics/bias - lb(op)_{layer_idx}_0': 0 for layer_idx in range(len(model.model.layers))},
            **{f'metrics/ub(op) - bias_{layer_idx}_0': 0 for layer_idx in range(len(model.model.layers))},
            **{f'metrics/bias - lb(op)_{layer_idx}_1': 0 for layer_idx in range(len(model.model.layers))},
            **{f'metrics/ub(op) - bias_{layer_idx}_1': 0 for layer_idx in range(len(model.model.layers))},
        ))
    elif args["atomic_element"] in ["bise", "conv"]:
        hyperparams.update(dict(
            **{f'metrics/bias - lb(op)_{layer_idx}': 0 for layer_idx in range(len(model.model.layers))},
            **{f'metrics/ub(op) - bias_{layer_idx}': 0 for layer_idx in range(len(model.model.layers))},
        ))

    logger.log_hyperparams(args, hyperparams)

    for selem_idx, selem in enumerate(args['morp_operation'].selems):
        fig, ax = plt.subplots(); ax.imshow(selem); ax.set_title(args['morp_operation'].operations[selem_idx])
        logger.experiment.add_figure(f"target_SE/target_SE_{selem_idx}", fig)
        # logger.experiment.add_image(f"target_SE/target_SE_{selem_idx}", selem[np.newaxis, :].astype(float))


    trainer = Trainer(
        max_epochs=1,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
        progress_bar_refresh_rate=10,
        callbacks=observables.copy(),
        log_every_n_steps=10,
    )

    trainer.fit(model, dataloader)


if __name__ == '__main__':
    start_all = time()

    code_saver = CodeSaver(
        src_path=os.getcwd(),
        temporary_path="deep_morpho/results",
        ignore_patterns=("*__pycache__*", "*results*", "data", "*.ipynb", '.git', 'ssm'),
    )

    code_saver.save_in_temporary_file()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    bugged = []
    for args_idx, args in enumerate(all_args):

        name = join(args["experiment_name"], args['morp_operation'].name)

        name += f"_{args['atomic_element']}"

        logger = TensorBoardLogger("deep_morpho/results", name=name, default_hp_metric=False)
        code_saver.save_in_final_file(join(logger.log_dir, 'code'))
        save_yaml(args, join(logger.log_dir, 'args.yaml'))

        console_logger = create_logger(
            f'args_{args_idx}', all_logs_path=join(logger.log_dir, 'all_logs.log'), error_path=join(logger.log_dir, 'error_logs.log')
        )

        log_console('Device: {}'.format(device), logger=console_logger)
        log_console('==================', logger=console_logger)
        log_console('==================', logger=console_logger)
        log_console(f'Args number {args_idx + 1} / {len(all_args)}', logger=console_logger)
        log_console('Time since beginning: {} '.format(format_time(time() - start_all)), logger=console_logger)
        log_console(logger.log_dir, logger=console_logger)
        log_console(args['morp_operation'], logger.log_dir, logger=console_logger)
        main(args, logger)
        # try:
        #     main(args, logger)
        # except Exception:
        #     console_logger.exception(
        #         f'Args nb {args_idx + 1} / {len(all_args)} failed : ')
        #     bugged.append(args_idx+1)

    code_saver.delete_temporary_file()

    log_console(f'{len(bugged)} Args Bugged: ', bugged, logger=console_logger)
    log_console(f'{len(all_args)} args done in {format_time(time() - start_all)} ', logger=console_logger)
