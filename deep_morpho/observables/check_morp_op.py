from torch.utils.tensorboard.summary import custom_scalars
import matplotlib.pyplot as plt

from .observable_layers import ObservableLayers
from general.nn.observables import Observable

from ..models import COBiSE, BiSE


class CheckMorpOperation(ObservableLayers):

    def __init__(self, selems, operations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selems = selems
        self.operations = operations


    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        layers = self._get_layers(pl_module)
        default_layout = self.get_layout(layers)
        layout = {"default": default_layout}
        trainer.logger.experiment._get_file_writer().add_summary(custom_scalars(layout))


    @staticmethod
    def get_layout(layers):
        default_layout = {}
        for layer_idx, layer in enumerate(layers):
            if isinstance(layer, COBiSE):
                for bise_idx, bise_layer in enumerate(layer.bises):
                    tags_dilation = [
                        f'comparative/weights/bias_{layer_idx}_{bise_idx}/bias',
                        f'comparative/weights/bias_{layer_idx}_{bise_idx}/dilation_lb',
                        f'comparative/weights/bias_{layer_idx}_{bise_idx}/dilation_ub'
                    ]
                    tags_erosion = [
                        f'comparative/weights/bias_{layer_idx}_{bise_idx}/bias',
                        f'comparative/weights/bias_{layer_idx}_{bise_idx}/erosion_lb',
                        f'comparative/weights/bias_{layer_idx}_{bise_idx}/erosion_ub'
                    ]
                    default_layout.update({
                        f"dilation_{layer_idx}_{bise_idx}": ['Margin', tags_dilation],
                        f"erosion_{layer_idx}_{bise_idx}": ['Margin', tags_erosion],
                    })

            else:
                tags_dilation = [
                    f'comparative/weights/bias_{layer_idx}/bias',
                    f'comparative/weights/bias_{layer_idx}/dilation_lb',
                    f'comparative/weights/bias_{layer_idx}/dilation_ub'
                ]
                tags_erosion = [
                    f'comparative/weights/bias_{layer_idx}/bias',
                    f'comparative/weights/bias_{layer_idx}/erosion_lb',
                    f'comparative/weights/bias_{layer_idx}/erosion_ub'
                ]
                default_layout.update({
                    f"dilation_{layer_idx}": ['Margin', tags_dilation],
                    f"erosion_{layer_idx}": ['Margin', tags_erosion],
                })

        return default_layout

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
        if isinstance(layer, COBiSE):
            for bise_idx, bise_layer in enumerate(layer.bises):
                self.write_scalars_and_metrics(trainer, bise_layer, f'{layer_idx}_{bise_idx}', 2*layer_idx + bise_idx)

        elif isinstance(layer, BiSE):
            self.write_scalars_and_metrics(trainer, layer, layer_idx, layer_idx)


    def write_scalars_and_metrics(self, trainer, layer, current_name, op_idx):
        erosion_lb, erosion_ub = layer.bias_bounds_erosion(self.selems[op_idx])
        dilation_lb, dilation_ub = layer.bias_bounds_dilation(self.selems[op_idx])

        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{current_name}/bias", -layer.bias, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{current_name}/erosion_lb", erosion_lb, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{current_name}/erosion_ub", erosion_ub, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{current_name}/dilation_lb", dilation_lb, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{current_name}/dilation_ub", dilation_ub, trainer.global_step)

        if self.operations[op_idx] == 'dilation':
            metrics = {
                f"metrics/bias - lb(op)_{current_name}": -layer.bias - dilation_lb,
                f"metrics/ub(op) - bias_{current_name}": dilation_ub - (-layer.bias),
            }
        elif self.operations[op_idx] == 'erosion':
            metrics = {
                f"metrics/bias - lb(op)_{current_name}": -layer.bias - erosion_lb,
                f"metrics/ub(op) - bias_{current_name}": erosion_ub - (-layer.bias),
            }
        else:
            raise NotImplementedError('operation must be dilation or erosion.')

        trainer.logger.log_metrics(metrics, trainer.global_step)


class ShowSelem(Observable):

    def __init__(self, freq=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
    ):
        if self.freq_idx % self.freq == 0:
            selems, operations = pl_module.model.get_bise_selems()
            for layer_idx, layer in enumerate(pl_module.model.layers):
                if not isinstance(layer, BiSE):
                    fig = self.default_figure("Not BiSE")

                elif selems[layer_idx] is None:
                    fig = self.default_figure("No operation found.")

                else:
                    fig = self.selem_fig(selems[layer_idx], operations[layer_idx])

                trainer.logger.experiment.add_figure(f"weights/learned_selem_{layer_idx}", fig, trainer.global_step)
        self.freq_idx += 1

    @staticmethod
    def default_figure(text):
        fig = plt.figure(figsize=(5, 5))
        plt.text(2, 2, text, horizontalalignment="center")
        return fig

    @staticmethod
    def selem_fig(selem, operation):
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(selem, interpolation="nearest")
        plt.title(operation)
        return fig
