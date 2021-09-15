import matplotlib.pyplot as plt
import itertools

from .observable_layers import ObservableLayers
from general.utils import max_min_norm

from ..models.bise import BiSE, LogicalNotBiSE


class PlotGradientBise(ObservableLayers):

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
        trainer.logger.experiment.add_figure(f"weights_gradient/{layer_idx}", self.get_figure_gradient(layer.weight.grad[0]), trainer.global_step)
        trainer.logger.experiment.add_scalar(f"weights/bias_gradient_{layer_idx}", layer.bias.grad, trainer.global_step)

    def get_figure_gradient(self, gradient):
        gradient = gradient[0].cpu().detach()
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
