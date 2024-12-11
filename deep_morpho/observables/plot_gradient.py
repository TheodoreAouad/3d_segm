import matplotlib.pyplot as plt
import itertools
from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch.nn as nn

from deep_morpho.models.bisel import BiSEL
from .observable_layers import ObservableLayersChans
from general.utils import max_min_norm
from ..models import BiSE, BiSEC, COBiSEC, COBiSE



class PlotGradientBise(ObservableLayersChans):

    def __init__(self, *args, freq: int = 100, **kwargs):
        """
        Frequency is in milliseconds.

        Args:
            self: write your description
            freq: write your description
        """
        super().__init__(*args, freq=freq, **kwargs)

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
        At the end of training the layers are the channels.

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
        grad_weights = layer.bises[chan_input].weight.grad[chan_output].squeeze()
        trainer.logger.experiment.add_figure(
            f"weights_gradient/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            self.get_figure_gradient(grad_weights),
            trainer.global_step
        )
        trainer.logger.experiment.add_histogram(
            f"weights_gradient_hist/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            grad_weights,
            trainer.global_step
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
        This function is called after the training step of the batch_end method. It grads the

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
        # if isinstance(layer, (BiSE, BiSEC, COBiSEC, COBiSE, BiSEL)):
        grad_bise_bias = layer.bises[chan_input].bias.grad[chan_output]
        if grad_bise_bias is not None:
            trainer.logger.experiment.add_scalars(
                f"weights/bisel/bias_gradient/layer_{layer_idx}_chout_{chan_output}",
                {f"chin_{chan_input}": grad_bise_bias},
                trainer.global_step
            )

        grad_bise_weights = layer.bises[chan_output].weight.grad[chan_input]
        if grad_bise_weights is not None:
            trainer.logger.experiment.add_scalars(
                f"weights/bisel/weights_gradient_mean/layer_{layer_idx}_chout_{chan_output}",
                {f"chin_{chan_input}": grad_bise_weights.mean()},
                trainer.global_step
            )

        for chan in [chan_input, chan_input + layer.in_channels]:
            grad_lui_weight = layer.luis[chan_output].weight.grad[0, chan]
            if grad_lui_weight is not None:
                trainer.logger.experiment.add_scalars(
                    f"weights/lui/weights_gradient/layer_{layer_idx}_chout_{chan_output}",
                    {f"chin_{chan}": grad_lui_weight},
                    trainer.global_step
                )

    def on_train_batch_end_layers_chan_output_always(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
        chan_output: int,
    ):
        """
        At the end of training the layer channel output is always the bias.

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
            chan_output: write your description
        """
        grad_lui_bias = layer.luis[chan_output].bias.grad[0]
        if grad_lui_bias is not None:
            trainer.logger.experiment.add_scalars(
                f"weights/lui/bias_gradient/layer_{layer_idx}",
                {f"chout_{chan_output}": grad_lui_bias},
                trainer.global_step
            )

    @staticmethod
    def get_figure_gradient(gradient):
        """
        Returns a matplotlib figure with the gradient normed.

        Args:
            gradient: write your description
        """
        gradient = gradient.cpu().detach()
        gradient_normed = max_min_norm(gradient)
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(gradient_normed, interpolation='nearest', cmap=plt.cm.gray)
        plt.colorbar()
        # plt.clim(gradient.min(), gradient.max())

        # Use white text if squares are dark; otherwise black.
        threshold = gradient_normed.max() / 2.

        for i, j in itertools.product(range(gradient.shape[0]), range(gradient.shape[1])):
            color = "white" if gradient_normed[i, j] < threshold else "black"
            # plt.text(j, i, round(gradient[i, j].item(), 2), horizontalalignment="center", color=color)
            plt.text(j, i, f"{gradient[i, j].item():.2e}", horizontalalignment="center", color=color)

        plt.tight_layout()
        return figure
