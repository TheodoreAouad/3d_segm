from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch.nn as nn

from .observable_layers import ObservableLayersChans


class WeightsHistogramBiSE(ObservableLayersChans):

    def __init__(self, *args, freq: int = 100, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)
        self.freq_idx["on_train_batch_end"] = 1

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
        chan_output: int,
    ):
        weight = layer.get_weight_bise(chin=chan_input, chout=chan_output)

        trainer.logger.experiment.add_histogram(
            f"weights_hist/Normalized/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            weight,
            trainer.global_step
        )
        trainer.logger.experiment.add_histogram(
            f"weights_hist/Raw/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            weight,
            trainer.global_step
        )

    def on_train_start(self, trainer, pl_module):
        layers = self._get_layers(pl_module)
        for layer_idx, layer in enumerate(layers):
            self.on_train_batch_end_layers(
                trainer, pl_module, None, None, None, None, layer, layer_idx
            )

        for layer_idx, layer in enumerate(layers):
            self.on_train_batch_end_layers_always(
                trainer, pl_module, None, None, None, None, layer, layer_idx
            )

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""
        layers = self._get_layers(pl_module)
        if self.freq_idx["on_train_batch_end"] % self.freq == 0:
            for layer_idx, layer in enumerate(layers):
                self.on_train_batch_end_layers(
                    trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, layer, layer_idx
                )

        self.freq_idx["on_train_batch_end"] += 1
        for layer_idx, layer in enumerate(layers):
            self.on_train_batch_end_layers_always(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, layer, layer_idx
            )
