import pathlib
from os.path import join

from torch.utils.tensorboard.summary import custom_scalars
import matplotlib.pyplot as plt

from .observable_layers import ObservableLayers, ObservableLayersChans
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


class ShowSelemAlmostBinary(Observable):

    def __init__(self, freq=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.freq_idx = 0
        self.last_selem_and_op = {}

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
                    # fig = self.default_figure("Not BiSE")
                    continue

                elif selems[layer_idx] is None:
                    continue
                    # fig = self.default_figure("No operation found.")

                else:
                    fig = self.selem_fig(selems[layer_idx], operations[layer_idx])

                trainer.logger.experiment.add_figure(f"learned_selem/almost_binary_{layer_idx}", fig, trainer.global_step)
                self.last_selem_and_op[layer_idx] = (selems[layer_idx], operations[layer_idx])
        self.freq_idx += 1

    @staticmethod
    def default_figure(text):
        fig = plt.figure(figsize=(5, 5))
        plt.text(2, 2, text, horizontalalignment="center")
        return fig

    @staticmethod
    def selem_fig(selem, operation):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(selem, interpolation="nearest")
        plt.title(operation)
        return fig

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for layer_idx, (selem, operation) in self.last_selem_and_op.items():
            fig = self.selem_fig(selem, operation)
            fig.savefig(join(final_dir, f"layer_{layer_idx}.png"))
            saved.append(fig)

        return saved


class ShowSelemBinary(ObservableLayersChans):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_selem_and_op = {}

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
        selem, operation = layer.bises[chan_input].find_selem_and_operation_chan(chan_output, v1=0, v2=1)
        if selem is None:
            return

        fig = self.selem_fig(selem, operation)
        trainer.logger.experiment.add_figure(f"learned_selem/binary/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig, trainer.global_step)
        self.last_selem_and_op[(layer_idx, chan_input, chan_output)] = (selem, operation)


    @staticmethod
    def selem_fig(selem, operation):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(selem, interpolation="nearest")
        plt.title(operation)
        return fig


    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for (layer_idx, chan_input, chan_output), (selem, operation) in self.last_selem_and_op.items():
            fig = self.selem_fig(selem, operation)
            fig.savefig(join(final_dir, f"layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
            saved.append(fig)

        return saved
