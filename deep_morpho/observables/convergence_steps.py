from typing import Any
import pathlib
from os.path import join

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np

from deep_morpho.observables.observable_layers import ObservableLayersChans
from general.nn.observables import Observable
from ..models import BiSE
from general.utils import save_json



class ConvergenceMetrics(Observable):
    """
    class used to calculate and track metrics in the tensorboard
    """
    def __init__(self, metrics, eps=1e-3):
        """
        Initialize the metrics object with the given metrics object.

        Args:
            self: write your description
            metrics: write your description
            eps: write your description
        """
        self.metrics = metrics
        self.cur_value = {
            "train": {k: None for k in metrics.keys()},
            "val": {k: None for k in metrics.keys()},
            "test": {k: None for k in metrics.keys()},
        }
        self.convergence_step = {
            "train": {k: None for k in metrics.keys()},
            "val": {k: None for k in metrics.keys()},
            "test": {k: None for k in metrics.keys()},
        }
        self.eps = eps


    def on_train_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        """
        Calculates and logs the metrics at the end of the training batch.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            outputs: write your description
            batch: write your description
            batch_idx: write your description
            preds: write your description
        """
        inputs, targets = batch
        self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='train')

    def on_validation_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        """
        Calculates and logs the metrics at the end of a batch of validation.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            outputs: write your description
            batch: write your description
            batch_idx: write your description
            preds: write your description
        """
        inputs, targets = batch
        self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='val')

    def on_test_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        """
        Calculates and logs the metrics at the end of a test batch.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            outputs: write your description
            batch: write your description
            batch_idx: write your description
            preds: write your description
        """
        inputs, targets = batch
        self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='test')

    def _calculate_and_log_metrics(self, trainer, pl_module, targets, preds, state='train'):
        """
        Calculates and logs the metrics.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            targets: write your description
            preds: write your description
            state: write your description
        """
        for metric_name in self.metrics:
            metric = self.metrics[metric_name](targets, preds)
            self.update_step(metric_name, metric, state, trainer.global_step)


            trainer.logger.experiment.add_scalars(
                f"comparative/convergence/metric_{metric_name}",
                {state: self.convergence_step[state][metric_name]}, trainer.global_step
            )

            trainer.logger.log_metrics(
                {f"convergence/metric_{metric_name}_{state}": self.convergence_step[state][metric_name]}, trainer.global_step
            )


            # f"metrics_multi_label_{batch_or_epoch}/{metric_name}/{state}", {f'label_{name_label}': metric}, step
            # trainer.logger.experiment.add_scalars(metric_name, {f'{metric_name}_{state}': metric})

    def update_step(self, metric_name, metric_value, state, step):
        """
        Update the step of the convergence metric.

        Args:
            self: write your description
            metric_name: write your description
            metric_value: write your description
            state: write your description
            step: write your description
        """
        if self.cur_value[state][metric_name] is None or self.cur_value[state][metric_name] < metric_value:
            self.cur_value[state][metric_name] = metric_value
            self.convergence_step[state][metric_name] = step
            return

        diff_rate = (self.cur_value[state][metric_name] - metric_value) / self.cur_value[state][metric_name]
        if diff_rate > self.eps:
            self.cur_value[state][metric_name] = metric_value
            self.convergence_step[state][metric_name] = step

    def save(self, save_path: str):
        """
        Saves the convergence step to a JSON file.

        Args:
            self: write your description
            save_path: write your description
        """
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json(self.convergence_step, join(final_dir, "convergence_step.json"))
        return self.convergence_step


class ConvergenceAlmostBinary(Observable):

    def __init__(self, freq=200, *args, **kwargs):
        """
        Initialize the filter.

        Args:
            self: write your description
            freq: write your description
        """
        super().__init__(*args, **kwargs)
        self.convergence_step = {}
        self.has_converged = {}
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
        """
        Run on batch end.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            outputs: write your description
            batch: write your description
            batch_idx: write your description
            dataloader_idx: write your description
        """
        # trainer.logger.experiment.add_image(f"weights/Normalized_{layer_idx}", layer._normalized_weight[0], trainer.global_step)
        # trainer.logger.experiment.add_image(f"weights/Raw_{layer_idx}", max_min_norm(layer.weight[0]), trainer.global_step)

        if self.freq_idx % self.freq == 0:
            selems, operations = pl_module.model.get_bise_selems()
            for layer_idx, layer in enumerate(pl_module.model.layers):
                if not isinstance(layer, BiSE):
                    continue

                self.update_step(layer_idx, selems[layer_idx] is not None, trainer.global_step)

                value = self.convergence_step.get(layer_idx, -1)

                trainer.logger.experiment.add_scalar(
                    f"comparative/convergence/almost_binary_{layer_idx}",
                    value, trainer.global_step
                )

                trainer.logger.log_metrics(
                    {f"convergence/almost_binary_{layer_idx}": value}, trainer.global_step
                )
        self.freq_idx += 1



    def update_step(self, layer_idx, is_converged, step):
        """
        Update the step of the convergence layer.

        Args:
            self: write your description
            layer_idx: write your description
            is_converged: write your description
            step: write your description
        """

        if not self.has_converged.get(layer_idx, False) and is_converged:
            self.convergence_step[layer_idx] = step

        if not is_converged:
            self.convergence_step[layer_idx] = -1

        self.has_converged[layer_idx] = is_converged

    def save(self, save_path: str):
        """
        Saves the convergence step to a JSON file.

        Args:
            self: write your description
            save_path: write your description
        """
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json(self.convergence_step, join(final_dir, "convergence_step.json"))
        return self.convergence_step



class ConvergenceBinary(ObservableLayersChans):

    def __init__(self, *args, **kwargs):
        """
        Initializes the state of the object.

        Args:
            self: write your description
        """
        super().__init__(*args, **kwargs)
        self.convergence_step = {"lui": {}, "bisel": {}}
        self.has_converged = {
            "lui": {"intersection": {}, "union": {}},
            "bisel": {"erosion": {}, "dilation": {}}
        }
        self.converged_value = {"lui": {}, "bisel": {}}

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
        At the end of training the layer channel outputs are set to 0 and the operation is performed on

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
        key = str((layer_idx, chan_output))
        C, operation = layer.luis[chan_output].find_set_and_operation_chan(0, v1=None, v2=None)

        self.update_step("lui", key, C, operation, trainer.global_step)

        step_value = self.convergence_step["lui"].get(key, self.default_value)
        step_value = -step_value if operation == "intersection" else step_value

        trainer.logger.experiment.add_scalars(
            f"comparative/convergence/binary/lui/layer_{layer_idx}/",
            {f"chout_{chan_output}": step_value}, trainer.global_step
        )

        trainer.logger.log_metrics(
            {f"convergence/binary/lui/layer_{layer_idx}_chout_{chan_output}": step_value}, trainer.global_step
        )

    @property
    def default_value(self):
        """
        Default value for the placeholder.

        Args:
            self: write your description
        """
        # return torch.tensor([np.nan])
        return np.nan

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
        At the end of training the layers channels.

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
        key = str((layer_idx, chan_input, chan_output))
        selem, operation = layer.bises[chan_input].find_selem_and_operation_chan(chan_output, v1=0, v2=1)

        self.update_step("bisel", key, selem, operation, trainer.global_step)

        step_value = self.convergence_step["bisel"].get(key, self.default_value)
        step_value = -step_value if operation == "erosion" else step_value

        trainer.logger.experiment.add_scalars(
            f"comparative/convergence/binary/bisel/layer_{layer_idx}_chout_{chan_output}/",
            {f"chin_{chan_input}": step_value}, trainer.global_step
        )

        trainer.logger.log_metrics(
            {f"convergence/binary/bisel/layer_{layer_idx}_chout_{chan_output}_chin_{chan_input}": step_value}, trainer.global_step
        )

    def update_step(self, layer_key, key, value, operation, step):
        """
        Update the step if it has converged.

        Args:
            self: write your description
            layer_key: write your description
            key: write your description
            value: write your description
            operation: write your description
            step: write your description
        """
        is_converged = operation is not None

        if (
            is_converged and
            (
                (not self.has_converged[layer_key][operation].get(key, False)) or   # it has not converged before
                (self.converged_value[layer_key].get(key, None) != value).any()   # its convergence value changed
            )
        ):
            self.convergence_step[layer_key][key] = step

        # we prepare to set only the right operation to True
        for other_op in self.has_converged[layer_key].keys():
            self.has_converged[layer_key][other_op][key] = False

        if not is_converged:
            self.convergence_step[layer_key][key] = self.default_value
        else:
            self.has_converged[layer_key][operation][key] = True
            self.converged_value[layer_key][key] = value


    def save(self, save_path: str):
        """
        Saves the convergence step to a JSON file.

        Args:
            self: write your description
            save_path: write your description
        """
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json(self.convergence_step, join(final_dir, "convergence_step.json"))
        return self.convergence_step
