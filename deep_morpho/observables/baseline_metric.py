import pathlib
from os.path import join

import torch

from general.nn.observables.observable import Observable
from general.utils import save_json



class InputAsPredMetric(Observable):
    """
    class used to calculate and track metrics in the tensorboard
    """
    def __init__(self, metrics, ):
        """
        Initialize the metrics object.

        Args:
            self: write your description
            metrics: write your description
        """
        self.metrics = metrics
        self.tb_steps = {metric: {"train": 0, "val": 0, "test": 0} for metric in self.metrics.keys()}
        self.last_value = {k: 0 for k in metrics.keys()}

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
        if inputs.shape[1] < targets.shape[1]:
            inputs = torch.cat([inputs for _ in range(targets.shape[1])], axis=1)
        elif inputs.shape[1] > targets.shape[1]:
            inputs = torch.cat([inputs[:, 0:1, ...] for _ in range(targets.shape[1])], axis=1)
        self._calculate_and_log_metrics(trainer, pl_module, targets, inputs.squeeze(), state='train')

    def _calculate_and_log_metrics(self, trainer, pl_module, targets, preds, state='train', batch_or_epoch='batch'):
        """
        Calculates and logs the metrics for the given targets and preds.

        Args:
            self: write your description
            trainer: write your description
            pl_module: write your description
            targets: write your description
            preds: write your description
            state: write your description
            batch_or_epoch: write your description
        """
        for metric_name in self.metrics:
            metric = self.metrics[metric_name](targets, preds)
            self.last_value[metric_name] = metric
            # pl_module.log(f"mean_metrics_{batch_or_epoch}/{metric_name}/{state}", metric)
            if batch_or_epoch == 'batch':
                step = self.tb_steps[metric_name][state]
            else:
                step = trainer.current_epoch

            trainer.logger.experiment.add_scalars(
                f"comparative/metrics_{batch_or_epoch}/{metric_name}", {f"input_as_pred": metric}, step
            )

            if batch_or_epoch == 'batch':
                self.tb_steps[metric_name][state] = step + 1

            # f"metrics_multi_label_{batch_or_epoch}/{metric_name}/{state}", {f'label_{name_label}': metric}, step
            # trainer.logger.experiment.add_scalars(metric_name, {f'{metric_name}_{state}': metric})

    def save(self, save_path: str):
        """
        Saves the current state of the object to a JSON file.

        Args:
            self: write your description
            save_path: write your description
        """
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json({k: str(v) for k, v in self.last_value.items()}, join(final_dir, "baseline_metrics.json"))
        return self.last_value
