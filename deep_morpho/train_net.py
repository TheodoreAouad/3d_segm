import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


from deep_morpho.datasets.generate_forms2 import get_random_diskorect
from deep_morpho.datasets.multi_rect_dataset import get_loader
from deep_morpho.models import LightningDilationLayer, LightningOpeningNet
import deep_morpho.observables as obs
from general.nn.observables import CalculateAndLogMetrics
from deep_morpho.metrics import dice
from deep_morpho.args import all_args


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

for args_idx, args in enumerate(all_args):
    print('=============================')
    print(f'Args nb {args_idx + 1}/{len(all_args)}')
    print(args['morp_operation'], args['selem'].sum())

    dataloader = get_loader(
        batch_size=args['batch_size'],
        n_inputs=args['n_inputs'],
        random_gen_fn=get_random_diskorect,
        random_gen_args=args['random_gen_args'],
        device=device,
        selem=args['selem'],
        morp_operation=args['morp_operation'],
    )

    loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam
    metrics = {'dice': lambda y_true, y_pred: dice(y_true, y_pred, threshold=.5).mean()}

    observables = [
        obs.SaveLoss(),
        CalculateAndLogMetrics(
            metrics=metrics,
            keep_preds_for_epoch=False,
        ),
        obs.PlotParametersDilation(freq=1),
        obs.PlotWeightsDilation(freq=100),
        obs.WeightsHistogramDilation(freq=100),
        obs.PlotPreds(freq=100),
    ]

    xs = torch.tensor(np.linspace(-6, 6, 100)).detach()
    if args['morp_operation'].lower() != "opening":
        model = LightningDilationLayer(
            model_args={
                "in_channels": 1, "out_channels": 1, "kernel_size": (5, 5), "activation_threshold_mode": args['activation_mode'],
                "activation_P": args['activation_P']
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
        ys = model.model.activation_threshold_fn(xs).detach()
    else:
        model = LightningOpeningNet(
            model_args={
                "share_weights": args['share_weights'], "in_channels": 1, "out_channels": 1, "kernel_size": (5, 5), "activation_threshold_mode": args['activation_mode'],
                "activation_P": args['activation_P']
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
        ys = model.model.dilations[0].activation_threshold_fn(xs).detach()

    model.to(device)

    logger = TensorBoardLogger("deep_morpho/results", name=args['morp_operation'])
    print(logger.log_dir)

    logger.experiment.add_graph(model, torch.ones(1, 1, 50, 50).to(device))
    logger.experiment.add_image("target_SE", args['selem'][np.newaxis, :].astype(float))
    fig, ax = plt.subplots(); ax.plot(xs, ys); ax.set_title(args['activation_mode'])
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
