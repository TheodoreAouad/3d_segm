from typing import List

from general.nn.observables import Observable
from ..models.lightning_bise import LightningBiSE, LightningLogicalNotBiSE


class ObservableLayers(Observable):

    def __init__(self, layers: List["nn.Module"] = None, freq: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.freq = freq
        self.freq_idx = 0


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
        if self.freq_idx % self.freq == 0:
            layers = self._get_layers(pl_module)
            for layer_idx, layer in enumerate(layers):
                self.on_train_batch_end_layers(
                    trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, layer, layer_idx
                )
        self.freq_idx += 1

    def on_train_batch_end_layers(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
    ):
        raise NotImplementedError

    def _get_layers(self, pl_module):
        if self.layers is not None:
            return self.layers

        if hasattr(pl_module.model, 'bises'):
            return pl_module.model.bises

        if isinstance(pl_module, LightningBiSE):
            return [pl_module.model]

        if isinstance(pl_module, LightningLogicalNotBiSE):
            return [pl_module.model]

        raise NotImplementedError('Cannot automatically select layers for model. Give them manually.')
