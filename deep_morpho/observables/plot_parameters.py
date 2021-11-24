import pathlib
from os.path import join

import matplotlib.pyplot as plt
import itertools

from .observable_layers import ObservableLayers
from general.utils import max_min_norm, save_json

from ..models import BiSE, BiSEC, COBiSE, COBiSEC, MaxPlusAtom


class PlotWeightsBiSE(ObservableLayers):

    def __init__(self, *args, freq: int = 100, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)
        self.last_weights = []

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
        # trainer.logger.experiment.add_image(f"weights/Normalized_{layer_idx}", layer._normalized_weight[0], trainer.global_step)
        # trainer.logger.experiment.add_image(f"weights/Raw_{layer_idx}", max_min_norm(layer.weight[0]), trainer.global_step)
        if isinstance(layer, (BiSE, COBiSE, BiSEC, COBiSEC)):
            trainer.logger.experiment.add_figure(f"weights/Normalized_{layer_idx}", self.get_figure_normalized_weights(
                layer._normalized_weight, layer.bias, layer.activation_P), trainer.global_step)
        trainer.logger.experiment.add_figure(f"weights/Raw_{layer_idx}", self.get_figure_raw_weights(layer.weight), trainer.global_step)


    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        for layer_idx, layer in enumerate(pl_module.model.layers):
            to_add = {"weights": layer.weights, "bias": layer.bias, "activation_P": layer.activation_P}
            if isinstance(layer, (BiSE, BiSEC, COBiSE, COBiSEC)):
                to_add["normalized_weights"] = layer._normalized_weight
            self.last_weights.append(to_add)

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        for layer_idx, layer_dict in enumerate(self.last_weights):
            for key, weight in layer_dict.items():
                if key == "normalized_weights":
                    fig = self.get_figure_normalized_weights(weight, bias=layer_dict['bias'], activation_P=layer_dict['activation_P'])
                elif key == "weights":
                    fig = self.get_figure_raw_weights(weight)
                fig.savefig(join(final_dir, f"{key}_{layer_idx}.png"))

        return self.last_weights


    @staticmethod
    def get_figure_normalized_weights(weights, bias, activation_P):
        weights = weights.cpu().detach().squeeze()
        figure = plt.figure(figsize=(8, 8))
        plt.title(f"bias={bias.item()}  act_P={activation_P.item()}")
        plt.imshow(weights, interpolation='nearest', cmap=plt.cm.gray)
        plt.colorbar()
        plt.clim(0, 1)

        # Use white text if squares are dark; otherwise black.

        for i, j in itertools.product(range(weights.shape[0]), range(weights.shape[1])):
            color = "white" if weights[i, j] < .5 else "black"
            plt.text(j, i, round(weights[i, j].item(), 2), horizontalalignment="center", color=color)

        plt.tight_layout()
        return figure

    @staticmethod
    def get_figure_raw_weights(weights):
        weights = weights.cpu().detach().squeeze()
        weights_normed = max_min_norm(weights)
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(weights_normed, interpolation='nearest', cmap=plt.cm.gray)
        plt.colorbar()
        # plt.clim(0, 1)

        # Use white text if squares are dark; otherwise black.
        threshold = weights_normed.max() / 2.

        for i, j in itertools.product(range(weights.shape[0]), range(weights.shape[1])):
            color = "white" if weights_normed[i, j] < threshold else "black"
            plt.text(j, i, round(weights[i, j].item(), 2), horizontalalignment="center", color=color)

        plt.tight_layout()
        return figure


class PlotParametersBiSE(ObservableLayers):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_params = {}

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
        metrics = {}
        last_params = {}

        if isinstance(layer, (BiSE, COBiSE, BiSEC, COBiSEC)):
            metrics.update({
                f"weights/sum_norm_weights_{layer_idx}": layer._normalized_weight.sum(),
                f"params/weight_P_{layer_idx}": layer.weight_P,
                f"params/activation_P_{layer_idx}": layer.activation_P,
            })

            last_params.update({
                f"weight_P": layer.weight_P.item(),
                f"activation_P": layer.activation_P.item(),
            })

        if isinstance(layer, (BiSEC, COBiSEC, MaxPlusAtom)):
            metrics[f"weights/norm_alpha_{layer_idx}"] = layer.thresholded_alpha
            last_params["norm_alpha"] = layer.thresholded_alpha.item()

        if isinstance(layer, (BiSE, BiSEC, COBiSEC)):
            metrics[f"weights/bias_{layer_idx}"] = layer.bias
            last_params["bias"] = layer.bias.item()

        elif isinstance(layer, COBiSE):
            for bise_idx, bise_layer in enumerate(layer.bises):
                metrics[f"weights/bias_{layer_idx}_{bise_idx}"] = layer.bias
                last_params[f"bias_{bise_idx}"] = layer.bias.item()

        trainer.logger.log_metrics(metrics, trainer.global_step)
        self.last_params[layer_idx] = last_params

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json({k1: {k2: str(v2) for k2, v2 in v1.items()} for k1, v1 in self.last_params.items()}, join(final_dir, "parameters.json"))
        return self.last_params
