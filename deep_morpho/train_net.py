from time import time
import os
from os.path import join
import pathlib

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# from deep_morpho.datasets.generate_forms2 import get_random_diskorect
# from deep_morpho.datasets.generate_forms3 import get_random_rotated_diskorect
from deep_morpho.datasets.multi_rect_dataset import InputOutputGeneratorDataset, MultiRectDataset
from deep_morpho.datasets.axspa_roi_dataset import AxspaROIDataset, AxspaROISimpleDataset
from deep_morpho.models import LightningBiMoNN, BiSE, COBiSE, BiSEC, COBiSEC
import deep_morpho.observables as obs
from general.nn.observables import CalculateAndLogMetrics
from general.utils import format_time, log_console, create_logger, save_yaml
from general.nn.utils import train_val_test_split
from deep_morpho.metrics import dice
from deep_morpho.args import all_args
from general.code_saver import CodeSaver


def get_dataloader(args):

    if args['dataset_type'] == 'diskorect':
        if (args['dataset_path'] is not None) and (args['dataset_path'] != 'generate'):
            dataloader = MultiRectDataset.get_loader(
                batch_size=args['batch_size'], dataset_path=args['dataset_path'], do_load_in_ram=args['in_ram'],
                morp_operation=args['morp_operation'], logger=console_logger, n_inputs=args['n_inputs'],
                num_workers=args['num_workers']
            )
        else:
            trainloader = InputOutputGeneratorDataset.get_loader(
                batch_size=args['batch_size'],
                n_inputs=args['n_inputs'],
                random_gen_fn=args['random_gen_fn'],
                random_gen_args=args['random_gen_args'],
                morp_operation=args['morp_operation'],
                device=device,
                num_workers=args['num_workers']
            )
            valloader = None
            testloader = None

    elif args['dataset_type'] == 'axspa_roi':
        data = pd.read_csv(args['dataset_path'])
        prop_train, prop_val, prop_test = args['train_test_split']
        trainloader, valloader, testloader = AxspaROISimpleDataset.get_train_val_test_loader(
            *train_val_test_split(
                data,
                train_size=int(prop_train * len(data)),
                val_size=int(prop_val * len(data)),
                test_size=int(prop_test * len(data))
            ),
            batch_size=args['batch_size'],
            preprocessing=args['preprocessing'],
            shuffle=True,
        )

    return trainloader, valloader, testloader


def main(args, logger):

    trainloader, valloader, testloader = get_dataloader(args)
    metrics = {'dice': lambda y_true, y_pred: dice(y_true, y_pred, threshold=.5).mean()}

    observables_dict = {
        "SaveLoss": obs.SaveLoss(),
        "CalculateAndLogMetrics": CalculateAndLogMetrics(
            metrics=metrics,
            keep_preds_for_epoch=False,
        ),
        "PlotPreds": obs.PlotPreds(freq={'train': args['freq_imgs'], 'val': 2}),
        "InputAsPredMetric": obs.InputAsPredMetric(metrics),
        "CountInputs": obs.CountInputs(),
        "PlotParametersBiSE": obs.PlotParametersBiSE(freq=1),
        "PlotWeightsBiSE": obs.PlotWeightsBiSE(freq=args['freq_imgs']),
        "PlotLUIParametersBiSEL": obs.PlotLUIParametersBiSEL(),
        "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        # "CheckMorpOperation": obs.CheckMorpOperation(
        #     selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50
        # ) if args['dataset_type'] == 'diskorect' else obs.Observable(),
        "PlotGradientBise": obs.PlotGradientBise(freq=args['freq_imgs']),
        "ConvergenceMetrics": obs.ConvergenceMetrics(metrics),
        # "ShowSelemAlmostBinary": obs.ShowSelemAlmostBinary(freq=args['freq_imgs']),
        "ShowSelemBinary": obs.ShowSelemBinary(freq=args['freq_imgs']),
        "ShowLUISetBinary": obs.ShowLUISetBinary(freq=args['freq_imgs']),
        # "ConvergenceAlmostBinary": obs.ConvergenceAlmostBinary(freq=100),
        "ConvergenceBinary": obs.ConvergenceBinary(freq=100),
    }

    observables = list(observables_dict.values())

    xs = torch.tensor(np.linspace(-6, 6, 100)).detach()

    model = LightningBiMoNN(
        model_args={
            "kernel_size": [args['kernel_size'] for _ in range(args['n_atoms'])],
            "channels": args['channels'],
            "atomic_element": args["atomic_element"] if args["atomic_element"] != "conv" else "bise",
            "threshold_mode": args['threshold_mode'],
            "activation_P": args['activation_P'],
            "constant_activation_P": args['constant_activation_P'],
            "constant_P_lui": args['constant_P_lui'],
            "constant_weight_P": args['constant_weight_P'],
            "init_weight_mode": args["init_weight_mode"],
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

    logger.experiment.add_graph(model, torch.ones(1, args['channels'][0], 50, 50).to(device))
    hyperparams = dict(
        # **{f'{k}_{layer_idx}': -1 for k in [
        #     f"weights/sum_norm_weights",
        #     f"params/weight_P",
        #     f"params/activation_P",
        #     f"weights/bias",
        # ] for layer_idx in range(len(model.model.layers))},
        **{
            f'{k}/layer_{layer_idx}_chout_{chan_output}_chin_{chan_input}': torch.tensor([np.nan]) for k in [
                "convergence/binary/bisel",
            ] for layer_idx in range(len(model.model.layers))
            for chan_input in range(model.model.layers[layer_idx].in_channels)
            for chan_output in range(model.model.layers[layer_idx].out_channels)
        },
        **{
            f'{k}/layer_{layer_idx}_chout_{chan_output}': torch.tensor([np.nan]) for k in [
                "convergence/binary/lui",
            ] for layer_idx in range(len(model.model.layers))
            for chan_output in range(model.model.layers[layer_idx].out_channels)
        },
        **{f"metrics_batch/dice_train": torch.tensor([np.nan])},
        **{f"convergence/metric_dice_train": torch.tensor([np.nan])},
    )

    if args['atomic_element'] in ['bisec', 'cobisec']:
        hyperparams.update(dict(
            **{f'{k}_{layer_idx}': -1 for k in ["weights/norm_alpha"] for layer_idx in range(len(model.model.layers))}
        ))

    if args["atomic_element"] == 'cobise':
        hyperparams.update(dict(
            **{f'metrics/bias - lb(op)_{layer_idx}_0': torch.tensor([np.nan]) for layer_idx in range(len(model.model.layers))},
            **{f'metrics/ub(op) - bias_{layer_idx}_0': torch.tensor([np.nan]) for layer_idx in range(len(model.model.layers))},
            **{f'metrics/bias - lb(op)_{layer_idx}_1': torch.tensor([np.nan]) for layer_idx in range(len(model.model.layers))},
            **{f'metrics/ub(op) - bias_{layer_idx}_1': torch.tensor([np.nan]) for layer_idx in range(len(model.model.layers))},
        ))
    elif args["atomic_element"] in ["bise", "conv"]:
        hyperparams.update(dict(
            **{f'metrics/bias - lb(op)_{layer_idx}': torch.tensor([np.nan]) for layer_idx in range(len(model.model.layers))},
            **{f'metrics/ub(op) - bias_{layer_idx}': torch.tensor([np.nan]) for layer_idx in range(len(model.model.layers))},
        ))

    logger.log_hyperparams(args, hyperparams)

    if args['dataset_type'] == "diskorect":
        pathlib.Path(join(logger.log_dir, "target_SE")).mkdir(exist_ok=True, parents=True)
        figs_selems = args['morp_operation'].plot_selem_arrays()
        for (layer_idx, chan_input, chan_output), fig in figs_selems.items():
            fig.savefig(join(logger.log_dir, "target_SE", f"target_SE_l_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
            logger.experiment.add_figure(f"target_SE/target_SE_l_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig)

        pathlib.Path(join(logger.log_dir, "target_UI")).mkdir(exist_ok=True, parents=True)
        figs_ui = args['morp_operation'].plot_ui_arrays()
        for (layer_idx, chan_output), fig in figs_ui.items():
            fig.savefig(join(logger.log_dir, "target_UI", f"target_UI_l_{layer_idx}_chin_chout_{chan_output}.png"))
            logger.experiment.add_figure(f"target_UI/target_UI_l_{layer_idx}_chin_chout_{chan_output}", fig)

        # for selem_idx, selem in enumerate(args['morp_operation'].selems):
        #     fig, ax = plt.subplots(); ax.imshow(selem); ax.set_title(args['morp_operation'].operations[selem_idx])
        #     fig.savefig(join(logger.log_dir, "target_SE", f"target_SE_{selem_idx}.png"))
        #     logger.experiment.add_figure(f"target_SE/target_SE_{selem_idx}", fig)
        #     logger.experiment.add_image(f"target_SE/target_SE_{selem_idx}", selem[np.newaxis, :].astype(float))


    trainer = Trainer(
        max_epochs=args['n_epochs'],
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
        # progress_bar_refresh_rate=10,
        callbacks=observables.copy(),
        log_every_n_steps=10,
    )

    trainer.fit(model, trainloader, valloader)

    for observable in observables:
        observable.save(join(trainer.log_dir, 'observables'))



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
    results = []

    for args_idx, args in enumerate(all_args):

        name = join(args["experiment_name"], args['experiment_subname'])

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
        results.append(main(args, logger))
        # try:
        #     main(args, logger)
        # except Exception:
        #     console_logger.exception(
        #         f'Args nb {args_idx + 1} / {len(all_args)} failed : ')
        #     bugged.append(args_idx+1)

    code_saver.delete_temporary_file()

    # results = pd.concat(results)

    log_console(f'{len(bugged)} Args Bugged: ', bugged, logger=console_logger)
    log_console(f'{len(all_args)} args done in {format_time(time() - start_all)} ', logger=console_logger)
