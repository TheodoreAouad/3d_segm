import matplotlib.pyplot as plt
import itertools

from .observable_layers import ObservableLayers
from general.utils import max_min_norm

from ..models.bise import BiSE, LogicalNotBiSE


class PlotWeightsDilation(ObservableLayers):

    def __init__(self, *args, freq: int = 100, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)

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

        trainer.logger.experiment.add_figure(f"weights/Normalized_{layer_idx}", self.get_figure_normalized_weights(layer._normalized_weight[0]), trainer.global_step)
        trainer.logger.experiment.add_figure(f"weights/Raw_{layer_idx}", self.get_figure_raw_weights(layer.weight[0]), trainer.global_step)


    def get_figure_normalized_weights(self, weights):
        weights = weights[0].cpu().detach()
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(weights, interpolation='nearest', cmap=plt.cm.gray)
        plt.colorbar()
        plt.clim(0, 1)

        # Use white text if squares are dark; otherwise black.

        for i, j in itertools.product(range(weights.shape[0]), range(weights.shape[1])):
            color = "white" if weights[i, j] < .5 else "black"
            plt.text(j, i, round(weights[i, j].item(), 2), horizontalalignment="center", color=color)

        plt.tight_layout()
        return figure

    def get_figure_raw_weights(self, weights):
        weights = weights[0].cpu().detach()
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


class PlotParametersDilation(ObservableLayers):

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
        # trainer.logger.experiment.add_scalar(f"weights/sum_norm_weights_{layer_idx}", layer._normalized_weight.sum(), trainer.global_step)
        # trainer.logger.experiment.add_scalar(f"params/weight_P_{layer_idx}", layer.weight_P, trainer.global_step)
        # trainer.logger.experiment.add_scalar(f"params/activation_P_{layer_idx}", layer.activation_P, trainer.global_step)
        #
        # if isinstance(layer, LogicalNotBiSE):
        #     trainer.logger.experiment.add_scalar(f"weights/norm_alpha_{layer_idx}", layer.thresholded_alpha, trainer.global_step)
        #     trainer.logger.experiment.add_scalar(f"weights/bias_{layer_idx}", layer.bias, trainer.global_step)
        # elif isinstance(layer, BiSE):
        #     trainer.logger.experiment.add_scalar(f"weights/bias_{layer_idx}", layer.bias, trainer.global_step)
        #     trainer.logger.experiment.add_scalar(f"weights/bias+weights_{layer_idx}", layer._normalized_weight.sum() + layer.bias, trainer.global_step)
        #
        metrics = {
            f"weights/sum_norm_weights_{layer_idx}": layer._normalized_weight.sum(),
            f"params/weight_P_{layer_idx}": layer.weight_P,
            f"params/activation_P_{layer_idx}": layer.activation_P,
        }
        # trainer.logger.log_metrics(f"weights/sum_norm_weights_{layer_idx}", layer._normalized_weight.sum(), trainer.global_step)
        # trainer.logger.log_metrics(f"params/weight_P_{layer_idx}", layer.weight_P, trainer.global_step)
        # trainer.logger.log_metrics(f"params/activation_P_{layer_idx}", layer.activation_P, trainer.global_step)

        if isinstance(layer, LogicalNotBiSE):
            metrics[f"weights/norm_alpha_{layer_idx}"] = layer.thresholded_alpha
            metrics[f"weights/bias_{layer_idx}"] = layer.bias
        elif isinstance(layer, BiSE):
            metrics[f"weights/bias_{layer_idx}"] = layer.bias
            metrics[f"weights/bias+weights_{layer_idx}"] = layer._normalized_weight.sum() + layer.bias
        trainer.logger.log_metrics(metrics, trainer.global_step)
