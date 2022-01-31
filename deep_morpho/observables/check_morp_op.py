import pathlib
from os.path import join

from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.tensorboard.summary import custom_scalars
import matplotlib.pyplot as plt

from .observable_layers import ObservableLayers, ObservableLayersChans
from general.nn.observables import Observable

from ..models import COBiSE, BiSE


class CheckMorpOperation(ObservableLayers):

    def __init__(self, selems, operations, *args, **kwargs):
        """
        Initialize the SDTree.

        Args:
            self: write your description
            selems: write your description
            operations: write your description
        """
        super().__init__(*args, **kwargs)
        self.selems = selems
        self.operations = operations


    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """
        Add the summary to the experiment.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
        """
        layers = self._get_layers(pl_module)
        default_layout = self.get_layout(layers)
        layout = {"default": default_layout}
        trainer.logger.experiment._get_file_writer().add_summary(custom_scalars(layout))


    @staticmethod
    def get_layout(layers):
        """
        Generate a default layout for the layers.

        Args:
            layers: write your description
        """
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
        """
        Writes scalars and metrics for the layers at the batch end.

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
        if isinstance(layer, COBiSE):
            for bise_idx, bise_layer in enumerate(layer.bises):
                self.write_scalars_and_metrics(trainer, bise_layer, f'{layer_idx}_{bise_idx}', 2*layer_idx + bise_idx)

        elif isinstance(layer, BiSE):
            self.write_scalars_and_metrics(trainer, layer, layer_idx, layer_idx)


    def write_scalars_and_metrics(self, trainer, layer, current_name, op_idx):
        """
        Write scalars and metrics to the log file.

        Args:
            self: write your description
            trainer: write your description
            layer: write your description
            current_name: write your description
            op_idx: write your description
        """
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
        """
        Initialize the SArray with a specific frequency.

        Args:
            self: write your description
            freq: write your description
        """
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
        """
        Calculates the selem and operation figs for each layer.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            outputs: write your description
            batch: write your description
            batch_idx: write your description
            dataloader_idx: write your description
        """
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
        """
        Create a default figure with a text.

        Args:
            text: write your description
        """
        fig = plt.figure(figsize=(5, 5))
        plt.text(2, 2, text, horizontalalignment="center")
        return fig

    @staticmethod
    def selem_fig(selem, operation):
        """
        Plot a selem image.

        Args:
            selem: write your description
            operation: write your description
        """
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(selem, interpolation="nearest")
        plt.title(operation)
        return fig

    def save(self, save_path: str):
        """
        Save the figure to a directory.

        Args:
            self: write your description
            save_path: write your description
        """
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
        """
        Initialize the state of the SArray.

        Args:
            self: write your description
        """
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
        """
        Add selem and operation to self. last_selem_and_op.

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
        selem, operation = layer.bises[chan_input].find_selem_and_operation_chan(chan_output, v1=0, v2=1)
        if selem is None:
            return

        fig = self.selem_fig(selem, operation)
        trainer.logger.experiment.add_figure(f"learned_selem/binary/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig, trainer.global_step)
        self.last_selem_and_op[(layer_idx, chan_input, chan_output)] = (selem, operation)


    @staticmethod
    def selem_fig(selem, operation):
        """
        Plot a selem image.

        Args:
            selem: write your description
            operation: write your description
        """
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(selem, interpolation="nearest")
        plt.title(operation)
        return fig


    def save(self, save_path: str):
        """
        Save the channel elements to a directory.

        Args:
            self: write your description
            save_path: write your description
        """
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for (layer_idx, chan_input, chan_output), (selem, operation) in self.last_selem_and_op.items():
            fig = self.selem_fig(selem, operation)
            fig.savefig(join(final_dir, f"layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
            saved.append(fig)

        return saved


class ShowLUISetBinary(ObservableLayersChans):

    def __init__(self, *args, **kwargs):
        """
        Initialize the state of the iterator.

        Args:
            self: write your description
        """
        super().__init__(*args, **kwargs)
        self.last_set_and_op = {}

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
        Set the figure at the end of each layer channel output.

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
        C, operation = layer.luis[chan_output].find_set_and_operation_chan(0, v1=None, v2=None)
        if C is None:
            return

        fig = self.set_fig(C, operation)
        trainer.logger.experiment.add_figure(f"learned_set_lui/binary/layer_{layer_idx}_chout_{chan_output}", fig, trainer.global_step)
        self.last_set_and_op[(layer_idx, chan_output)] = (C, operation)


    @staticmethod
    def set_fig(C, operation):
        """
        Set figure and figure for given operation.

        Args:
            C: write your description
            operation: write your description
        """
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(C[:, None].astype(int), interpolation="nearest", vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks(range(len(C)))
        plt.title(operation)
        return fig


    def save(self, save_path: str):
        """
        Saves the current state of the channel graph to a file.

        Args:
            self: write your description
            save_path: write your description
        """
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for (layer_idx, chan_output), (C, operation) in self.last_set_and_op.items():
            fig = self.set_fig(C, operation)
            fig.savefig(join(final_dir, f"layer_{layer_idx}_chout_{chan_output}.png"))
            saved.append(fig)

        return saved
