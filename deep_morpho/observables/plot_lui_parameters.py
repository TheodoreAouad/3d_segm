from os.path import join
import pathlib
from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np

from general.nn.observables import Observable
from .observable_layers import ObservableLayersChans


# class PlotLUIParameters(Observable):

#     def on_train_batch_end_with_preds(
#         self,
#         trainer: "pl.Trainer",
#         pl_module: "pl.LightningModule",
#         outputs: "STEP_OUTPUT",
#         batch: "Any",
#         batch_idx: int,
#         preds: "Any",
#     ) -> None:
#         for output_chan in range(pl_module.model.chan_outputs):
#             metrics_chan = {}
#             for input_chan in range(pl_module.model.chan_inputs):
#                 metrics_chan[f"beta_in_{input_chan}"] = pl_module.model.weight[output_chan, input_chan]

#             trainer.logger.experiment.add_scalars(f'weights/train/beta_chan_{input_chan}', metrics_chan, trainer.global_step)
#             trainer.logger.log_metrics({f'weights/train/bias_chan_{input_chan}': pl_module.model.bias[output_chan]}, trainer.global_step)


class PlotLUIParametersBiSEL(ObservableLayersChans):

    def on_train_batch_end_layers_chan_output(
        self,
        trainer='pl.Trainer',
        pl_module='pl.LightningModule',
        outputs="STEP_OUTPUT",
        batch="Any",
        batch_idx=int,
        dataloader_idx=int,
        layer="nn.Module",
        layer_idx=int,
        chan_output=int,
    ):
        self.log_lui_chout(trainer, layer, layer_idx, chan_output)

    def on_train_start(self, trainer, pl_module):
        layers = self._get_layers(pl_module)
        for layer_idx, layer in enumerate(layers):
            self.on_train_batch_end_layers(
                trainer, pl_module, None, None, None, None, layer, layer_idx
            )


    def on_train_batch_end_layers_chans(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
        chan_input: int,
        chan_output: int
    ):
        self.log_lui_chin_chout(trainer, layer, layer_idx, chan_input, chan_output)

    def log_lui_chin_chout(
        self,
        trainer,
        layer,
        layer_idx,
        chan_input,
        chan_output,
    ):
        coef = layer.get_coef_lui(chout=chan_output)[chan_input]

        trainer.logger.experiment.add_scalars(
            f'params/lui_coefs/layer_{layer_idx}_chout_{chan_output}',
            {f"chin_{chan_input}": coef},
            trainer.global_step
        )

    def log_lui_chout(
        self,
        trainer,
        layer,
        layer_idx,
        chan_output,
    ):
        bias_lui = layer.get_bias_lui(chout=chan_output)
        activation_P_lui = layer.get_activation_P_lui(chout=chan_output)

        trainer.logger.experiment.add_scalars(
            f'params/lui_bias/layer_{layer_idx}',
            {f'chout_{chan_output}': bias_lui},
            trainer.global_step
        )

        trainer.logger.experiment.add_scalars(
            f'params/lui_P/layer_{layer_idx}',
            {f'chout_{chan_output}': activation_P_lui},
            trainer.global_step
        )
