import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from skimage.morphology import disk
from torch import nn
import numpy as np


from deep_morpho.datasets.generate_forms2 import get_random_diskorect
from deep_morpho.datasets.multi_rect_dataset import get_loader
from deep_morpho.models import LightningDilationLayer, LightningOpeningNet
import deep_morpho.observables as obs
from general.nn.observables import CalculateAndLogMetrics
from deep_morpho.metrics import dice

## ARGS ############

# morp_operation = 'Dilation'
morp_operation = 'Erosion'
# morp_operation = 'Opening'

batch_size = 32
n_inputs = 500_000
random_gen_args={'size': (50, 50), 'n_shapes': 15, 'max_shape': (15, 15)}

share_weights = True
do_thresh_penalization = True
args_thresh_penalization = {
    'coef': .005,
    'degree': 4,
    'detach_weights': True,
}
first_batch_pen = 1000

selem = np.zeros((5, 5))
# selem[np.arange(5), 5 - np.arange(1, 6)] = 1
# selem[np.arange(5), np.arange(5)] = 1
# selem[:, 2] = 1
selem[2, :] = 1
# selem = disk(2)

####################



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

print(morp_operation, selem.sum())

dataloader = get_loader(
    batch_size=batch_size,
    n_inputs=n_inputs,
#     random_gen_fn=gfo.random_multi_rect,
#     random_gen_args={
#         'size': (50, 50),
#         'n_rectangles': 8,
#         'max_shape': (10, 10),
#     },
    random_gen_fn=get_random_diskorect,
    random_gen_args=random_gen_args,
    device=device,
    selem=selem,
    morp_operation=morp_operation,
)

loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam

if morp_operation.lower() != "opening":
    observables = [
        obs.SaveLoss(),
        obs.PlotParametersDilation(100),
        obs.PlotPreds(100),
        obs.WeightsHistogramDilation(1000),
        CalculateAndLogMetrics(
            metrics={'dice': lambda y_true, y_pred: dice(y_true, y_pred, threshold=.5).mean()}, keep_preds_for_epoch=False)
    ]

    model = LightningDilationLayer(
        model_args={
            "in_channels": 1, "out_channels": 1, "kernel_size": (5, 5)
        },
        learning_rate=1e-2,
        loss=loss,
        optimizer=optimizer,
        output_dir="deep_morpho/results",
        observables=observables,
        do_thresh_penalization=do_thresh_penalization,
        args_thresh_penalization=args_thresh_penalization,
        first_batch_pen=first_batch_pen,
    )
else:
    observables = [
        obs.SaveLoss(),
        obs.PlotParametersMultipleDilations(1),
        obs.PlotPreds(1),
        obs.WeightsHistogramMultipleDilations(1),
        CalculateAndLogMetrics(
            metrics={'dice': lambda y_true, y_pred: dice(y_true, y_pred, threshold=.5).mean()}, keep_preds_for_epoch=False)
    ]

    model = LightningOpeningNet(
        model_args={
            "share_weights": share_weights, "in_channels": 1, "out_channels": 1, "kernel_size": (5, 5)
        },
        learning_rate=5e-2,
        loss=loss,
        optimizer=optimizer,
        output_dir="deep_morpho/results",
        observables=observables,
        do_thresh_penalization=do_thresh_penalization,
        args_thresh_penalization=args_thresh_penalization,
        first_batch_pen=first_batch_pen,
    )

model.to(device)

logger = TensorBoardLogger("deep_morpho/results", name=morp_operation)
print(logger.log_dir)

logger.experiment.add_graph(model, torch.ones(1, 1, 50, 50).to(device))
logger.experiment.add_image("target_SE", selem[np.newaxis, :].astype(float))

trainer = Trainer(
    max_epochs=1,
    gpus=1 if torch.cuda.is_available() else 0,
    logger=logger,
    progress_bar_refresh_rate=10,
    callbacks=observables.copy(),
    log_every_n_steps=10,
)

trainer.fit(model, dataloader)
