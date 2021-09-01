import torch
from general.nn.observables.observable import Observable


class InputAsPredMetric(Observable):
    """
    class used to calculate and track metrics in the tensorboard
    """
    def __init__(self, metrics, ):
        self.metrics = metrics
        self.tb_steps = {metric: {"train": 0, "val": 0, "test": 0} for metric in self.metrics.keys()}

    def on_train_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        inputs, targets = batch
        self._calculate_and_log_metrics(trainer, pl_module, targets, inputs.squeeze(), state='train')

    def _calculate_and_log_metrics(self, trainer, pl_module, targets, preds, state='train', batch_or_epoch='batch'):
        for metric_name in self.metrics:
            metric = self.metrics[metric_name](targets, preds)
            # pl_module.log(f"mean_metrics_{batch_or_epoch}/{metric_name}/{state}", metric)
            if batch_or_epoch == 'batch':
                step = self.tb_steps[metric_name][state]
            else:
                step = trainer.current_epoch

            trainer.logger.experiment.add_scalars(
                f"mean_metrics_{batch_or_epoch}/{metric_name}", {f"input_as_pred": metric}, step
            )

            if batch_or_epoch == 'batch':
                self.tb_steps[metric_name][state] = step + 1

            # f"metrics_multi_label_{batch_or_epoch}/{metric_name}/{state}", {f'label_{name_label}': metric}, step
            # trainer.logger.experiment.add_scalars(metric_name, {f'{metric_name}_{state}': metric})
