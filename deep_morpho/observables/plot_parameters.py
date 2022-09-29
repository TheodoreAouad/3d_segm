import pathlib
from os.path import join
import itertools
from typing import Any


import matplotlib.pyplot as plt
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch.nn as nn


from .observable_layers import ObservableLayersChans
from general.utils import max_min_norm, save_json

from ..models import BiSE, BiSEL


class PlotWeightsBiSE(ObservableLayersChans):

    def __init__(self, *args, freq: int = 100, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)
        self.last_weights = []

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
        weights = layer.weights[chan_output, chan_input]
        weights_norm = layer._normalized_weight[chan_output, chan_input]
        trainer.logger.experiment.add_figure(
            f"weights_normalized/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            self.get_figure_raw_weights(weights_norm, layer.bias_bise[chan_output, chan_input], layer.activation_P_bise[chan_output, chan_input]),
            # self.get_figure_normalized_weights(weights_norm, layer.bias_bise[chan_output, chan_input], layer.activation_P_bise[chan_output, chan_input]),
            trainer.global_step
        )
        trainer.logger.experiment.add_figure(
            f"weights_raw/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}",
            self.get_figure_raw_weights(weights, layer.bias_bise[chan_output, chan_input], layer.activation_P_bise[chan_output, chan_input]),
            trainer.global_step
        )



    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        for layer_idx, layer in enumerate(pl_module.model.layers):
            to_add = {"weights": layer.weights, "bias_bise": layer.bias_bise, "activation_P_bise": layer.activation_P_bise}
            if isinstance(layer, (BiSE, BiSEL)):
                to_add["normalized_weights"] = layer._normalized_weight
            self.last_weights.append(to_add)

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        for layer_idx, layer_dict in enumerate(self.last_weights):
            for key, weight in layer_dict.items():
                if key not in ["normalized_weights", "weights"]:
                    continue
                for chan_output in range(weight.shape[0]):
                    for chan_input in range(weight.shape[1]):
                        if key == "normalized_weights":
                            # fig = self.get_figure_normalized_weights(weight[chan_output, chan_input],
                            fig = self.get_figure_raw_weights(weight[chan_output, chan_input],
                            bias=layer_dict['bias_bise'][chan_output, chan_input], activation_P=layer_dict['activation_P_bise'][chan_output, chan_input])
                        elif key == "weights":
                            fig = self.get_figure_raw_weights(weight[chan_output, chan_input],
                            bias=layer_dict['bias_bise'][chan_output, chan_input], activation_P=layer_dict['activation_P_bise'][chan_output, chan_input])

                        fig.savefig(join(final_dir, f"{key}_layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))

        return self.last_weights


    @staticmethod
    def get_figure_normalized_weights(weights, bias, activation_P):
        weights = weights.cpu().detach()
        figure = plt.figure(figsize=(8, 8))
        plt.title(f"bias={bias.item():.3f}  act_P={activation_P.item():.3f}  sum={weights.sum():.3f}")
        plt.imshow(weights, interpolation='nearest',)
        plt.colorbar()
        plt.clim(0, 1)

        # Use white text if squares are dark; otherwise black.

        for i, j in itertools.product(range(weights.shape[0]), range(weights.shape[1])):
            color = "white" if weights[i, j] < .5 else "black"
            plt.text(j, i, round(weights[i, j].item(), 2), horizontalalignment="center", color=color)

        plt.tight_layout()
        return figure

    @staticmethod
    def get_figure_raw_weights(weights, bias, activation_P):
        weights = weights.cpu().detach()
        weights_normed = max_min_norm(weights)
        figure = plt.figure(figsize=(8, 8))
        plt.title(f"bias={bias.item():.3f}  act_P={activation_P.item():.3f}  sum={weights.sum():.3f}")
        plt.imshow(weights_normed, interpolation='nearest',)
        plt.colorbar()
        # plt.clim(0, 1)

        # Use white text if squares are dark; otherwise black.
        threshold = weights_normed.max() / 2.

        for i, j in itertools.product(range(weights.shape[0]), range(weights.shape[1])):
            color = "white" if weights_normed[i, j] < threshold else "black"
            plt.text(j, i, round(weights[i, j].item(), 2), horizontalalignment="center", color=color)

        plt.tight_layout()
        return figure


class PlotParametersBiSE(ObservableLayersChans):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_params = {}

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
        metrics = {}
        last_params = {}

        metrics[f'params/activation_P/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}'] = layer.activation_P_bise[chan_output, chan_input]
        metrics[f'params/bias_bise/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}'] = layer.bias_bise[chan_output, chan_input]

        trainer.logger.log_metrics(metrics, trainer.global_step)
        self.last_params[layer_idx] = last_params

        trainer.logger.experiment.add_scalars(
            f"comparative/weight_P/layer_{layer_idx}_chout_{chan_output}",
            {f"chin_{chan_input}": layer.weight_P_bise[chan_output, chan_input]},
            trainer.global_step
        )
        trainer.logger.experiment.add_scalars(
            f"comparative/activation_P/layer_{layer_idx}_chout_{chan_output}",
            {f"chin_{chan_input}": layer.activation_P_bise[chan_output, chan_input]},
            trainer.global_step
        )
        trainer.logger.experiment.add_scalars(
            f"comparative/bias_bise/layer_{layer_idx}_chout_{chan_output}",
            {f"chin_{chan_input}": layer.bias_bise[chan_output, chan_input]},
            trainer.global_step
        )

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json({k1: {k2: str(v2) for k2, v2 in v1.items()} for k1, v1 in self.last_params.items()}, join(final_dir, "parameters.json"))
        return self.last_params
