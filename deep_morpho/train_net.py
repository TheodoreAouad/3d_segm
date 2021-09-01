from time import time
import os
from os.path import join

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


# from deep_morpho.datasets.generate_forms2 import get_random_diskorect
# from deep_morpho.datasets.generate_forms3 import get_random_rotated_diskorect
from deep_morpho.datasets.multi_rect_dataset import get_loader
from deep_morpho.models import LightningBiSE, LightningLogicalNotBiSE, LightningOpeningNet, LightningBiMoNN
import deep_morpho.observables as obs
from general.nn.observables import CalculateAndLogMetrics
from general.utils import format_time, log_console, create_logger, save_yaml
from deep_morpho.metrics import dice
from deep_morpho.args import all_args
from general.code_saver import CodeSaver


def main(args, logger):
    dataloader = get_loader(
        batch_size=args['batch_size'],
        n_inputs=args['n_inputs'],
        random_gen_fn=args['random_gen_fn'],
        random_gen_args=args['random_gen_args'],
        device=device,
        selem=args['selem'],
        morp_operation=args['morp_operation'].morp_fn,
    )

    # loss = nn.BCEWithLogitsLoss()
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam
    metrics = {'dice': lambda y_true, y_pred: dice(y_true, y_pred, threshold=.5).mean()}

    observables = [
        obs.SaveLoss(),
        CalculateAndLogMetrics(
            metrics=metrics,
            keep_preds_for_epoch=False,
        ),
        obs.InputAsPredMetric(metrics),
        obs.PlotParametersDilation(freq=1),
        obs.PlotWeightsDilation(freq=100),
        obs.WeightsHistogramDilation(freq=100),
        obs.PlotPreds(freq=100),
        obs.CountInputs(),
    ]

    xs = torch.tensor(np.linspace(-6, 6, 100)).detach()

    # if args['morp_operation'].name.lower() in ['erosion', 'dilation']:
    #     model_args = {
    #         "kernel_size": (args['kernel_size'], args['kernel_size']), "activation_threshold_mode": args['threshold_mode'],
    #         "activation_P": args['activation_P'], "init_weight_identity": args["init_weight_identity"],
    #     }
    #     if args['logical_not']:
    #         lightning_model = LightningLogicalNotBiSE
    #         model_args['logical_not_threshold_mode'] = args['threshold_mode']
    #     else:
    #         lightning_model = LightningBiSE
    #
    #     model = lightning_model(
    #         model_args=model_args,
    #         learning_rate=args['learning_rate'],
    #         loss=loss,
    #         optimizer=optimizer,
    #         output_dir="deep_morpho/results",
    #         observables=observables,
    #         do_thresh_penalization=args['do_thresh_penalization'],
    #         args_thresh_penalization=args['args_thresh_penalization'],
    #         first_batch_pen=args['first_batch_pen'],
    #     )
    #     ys = model.model.activation_threshold_fn(xs).detach()
    # elif args['morp_operation'].name.lower() == 'opening':
    #     model = LightningOpeningNet(
    #         model_args={
    #             "share_weights": args['share_weights'], "kernel_size": (args['kernel_size'], args['kernel_size']), "activation_threshold_mode": args['threshold_mode'],
    #             "activation_P": args['activation_P'], "init_weight_identity": args["init_weight_identity"],
    #         },
    #         learning_rate=args['learning_rate'],
    #         loss=loss,
    #         optimizer=optimizer,
    #         output_dir="deep_morpho/results",
    #         observables=observables,
    #         do_thresh_penalization=args['do_thresh_penalization'],
    #         args_thresh_penalization=args['args_thresh_penalization'],
    #         first_batch_pen=args['first_batch_pen'],
    #     )
    #     ys = model.model.bises[0].activation_threshold_fn(xs).detach()
    #
    # else:
    model = LightningBiMoNN(
        model_args={
            "kernel_size": [args['kernel_size'] for _ in range(len(args['morp_operation']))],
            "threshold_mode": args['threshold_mode'],
            "activation_P": args['activation_P'],
            "init_weight_identity": args["init_weight_identity"],
            "logical_not": args["logical_not"],
            "alpha_init": args["alpha_init"],
        },
        learning_rate=args['learning_rate'],
        loss=loss,
        optimizer=optimizer,
        output_dir="deep_morpho/results",
        observables=observables,
        do_thresh_penalization=args['do_thresh_penalization'],
        args_thresh_penalization=args['args_thresh_penalization'],
        first_batch_pen=args['first_batch_pen'],
    )
    ys = model.model.bises[0].activation_threshold_fn(xs).detach()


    model.to(device)

    logger.experiment.add_graph(model, torch.ones(1, 1, 50, 50).to(device))

    for selem_idx, selem in enumerate(args['morp_operation'].selems):
        fig, ax = plt.subplots(); ax.imshow(selem); ax.set_title(args['morp_operation'].operations[selem_idx])
        logger.experiment.add_figure(f"target_SE/target_SE_{selem_idx}", fig)
        # logger.experiment.add_image(f"target_SE/target_SE_{selem_idx}", selem[np.newaxis, :].astype(float))
    fig, ax = plt.subplots(); ax.plot(xs, ys); ax.set_title(args['threshold_mode'])
    logger.experiment.add_figure("threshold_fn", fig)

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
        if args['logical_not']:
            name += "_logical_not"

        logger = TensorBoardLogger("deep_morpho/results", name=name)
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
