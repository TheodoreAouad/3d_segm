from typing import List, Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch.nn as nn

from deep_morpho.models.bisel import BiSEL
from general.nn.observables import Observable
from ..models.lightning_bise import LightningBiSE, LightningBiSEC


class ObservableLayers(Observable):
    """ Class to do an observable on all the layers of the BiMoNN.
    """

    def __init__(self, layers: List["nn.Module"] = None, freq: int = 1, layer_name: str = 'layers', *args, **kwargs):
        """
        Initialize the filter with a list of layers and a frequency.

        Args:
            self: write your description
            layers: write your description
            List: write your description
            freq: write your description
            layer_name: write your description
        """
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.freq = freq
        self.freq_idx = 0
        self.layer_name = layer_name


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
        if self.freq_idx % self.freq == 0:
            for layer_idx, layer in enumerate(layers):
                self.on_train_batch_end_layers(
                    trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, layer, layer_idx
                )
        self.freq_idx += 1
        for layer_idx, layer in enumerate(layers):
            self.on_train_batch_end_layers_always(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, layer, layer_idx
            )



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
        """
        Callback when the training batch end the layers.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            outputs: write your description
            batch: write your description
            batch_idx: write your description
            dataloader_idx: write your description
            layer: write your description
            layer_idx: write your description
        """
        pass

    def on_train_batch_end_layers_always(
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
        """
        Callback when the training batch end all layers are always finished.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            outputs: write your description
            batch: write your description
            batch_idx: write your description
            dataloader_idx: write your description
            layer: write your description
            layer_idx: write your description
        """
        pass

    def _get_layers(self, pl_module):
        """
        Get the layers for the given pl_module.

        Args:
            self: write your description
            pl_module: write your description
        """

        if self.layers is not None:
            return self.layers

        if hasattr(pl_module.model, self.layer_name):
            return getattr(pl_module.model, self.layer_name)

        if isinstance(pl_module, LightningBiSE):
            return [pl_module.model]

        if isinstance(pl_module, LightningBiSEC):
            return [pl_module.model]

        raise NotImplementedError('Cannot automatically select layers for model. Give them manually.')



class ObservableLayersChans(ObservableLayers):

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
        """
        Callback when the training batch end all layers.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            outputs: write your description
            batch: write your description
            batch_idx: write your description
            dataloader_idx: write your description
            layer: write your description
            layer_idx: write your description
        """
        if isinstance(layer, BiSEL):
            for chan_output in range(layer.out_channels):
                self.on_train_batch_end_layers_chan_output(
                    trainer=trainer,
                    pl_module=pl_module,
                    outputs=outputs,
                    batch=batch,
                    batch_idx=batch_idx,
                    dataloader_idx=dataloader_idx,
                    layer=layer,
                    layer_idx=layer_idx,
                    chan_output=chan_output,
                )
                for chan_input in range(layer.in_channels):
                    self.on_train_batch_end_layers_chans(
                        trainer=trainer,
                        pl_module=pl_module,
                        outputs=outputs,
                        batch=batch,
                        batch_idx=batch_idx,
                        dataloader_idx=dataloader_idx,
                        layer=layer,
                        layer_idx=layer_idx,
                        chan_input=chan_input,
                        chan_output=chan_output,
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
        chan_output: int,
    ):
        """
        Callback when the training batch end of layers channels.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            outputs: write your description
            batch: write your description
            batch_idx: write your description
            dataloader_idx: write your description
            layer: write your description
            layer_idx: write your description
            chan_input: write your description
            chan_output: write your description
        """
        pass

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
        """
        Callback when the batch_end of a training batch includes the layer and the chan_output.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            outputs: write your description
            batch: write your description
            batch_idx: write your description
            int: write your description
            dataloader_idx: write your description
            int: write your description
            layer: write your description
            layer_idx: write your description
            int: write your description
            chan_output: write your description
            int: write your description
        """
        pass


    def on_train_batch_end_layers_always(
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
        """
        This method is called when the training batch is finished.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            outputs: write your description
            batch: write your description
            batch_idx: write your description
            dataloader_idx: write your description
            layer: write your description
            layer_idx: write your description
        """
        if isinstance(layer, BiSEL):
            for chan_output in range(layer.out_channels):
                self.on_train_batch_end_layers_chan_output_always(
                    trainer=trainer,
                    pl_module=pl_module,
                    outputs=outputs,
                    batch=batch,
                    batch_idx=batch_idx,
                    dataloader_idx=dataloader_idx,
                    layer=layer,
                    layer_idx=layer_idx,
                    chan_output=chan_output,
                )
                for chan_input in range(layer.in_channels):
                    self.on_train_batch_end_layers_chans_always(
                        trainer=trainer,
                        pl_module=pl_module,
                        outputs=outputs,
                        batch=batch,
                        batch_idx=batch_idx,
                        dataloader_idx=dataloader_idx,
                        layer=layer,
                        layer_idx=layer_idx,
                        chan_input=chan_input,
                        chan_output=chan_output,
                    )

    def on_train_batch_end_layers_chans_always(
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
        """
        Callback when the training batch end of layers is called.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            outputs: write your description
            batch: write your description
            batch_idx: write your description
            dataloader_idx: write your description
            layer: write your description
            layer_idx: write your description
            chan_input: write your description
            chan_output: write your description
        """
        pass

    def on_train_batch_end_layers_chan_output_always(
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
        """
        Callback when the training batch end with layers_chan_output_always.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            outputs: write your description
            batch: write your description
            batch_idx: write your description
            int: write your description
            dataloader_idx: write your description
            int: write your description
            layer: write your description
            layer_idx: write your description
            int: write your description
            chan_output: write your description
            int: write your description
        """
        pass
