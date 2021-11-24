import pathlib
from os.path import join


from deep_morpho.observables.observable_layers import ObservableLayers
from general.nn.observables import Observable
from ..models import BiSE
from general.utils import save_json



class ConvergenceMetrics(Observable):
    """
    class used to calculate and track metrics in the tensorboard
    """
    def __init__(self, metrics, eps=1e-3):
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
        inputs, targets = batch
        self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='train')

    def on_validation_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        inputs, targets = batch
        self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='val')

    def on_test_batch_end_with_preds(self, trainer, pl_module, outputs, batch, batch_idx, preds):
        inputs, targets = batch
        self._calculate_and_log_metrics(trainer, pl_module, targets, preds, state='test')

    def _calculate_and_log_metrics(self, trainer, pl_module, targets, preds, state='train'):
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
        if self.cur_value[state][metric_name] is None or self.cur_value[state][metric_name] < metric_value:
            self.cur_value[state][metric_name] = metric_value
            self.convergence_step[state][metric_name] = step
            return

        diff_rate = (self.cur_value[state][metric_name] - metric_value) / self.cur_value[state][metric_name]
        if diff_rate > self.eps:
            self.cur_value[state][metric_name] = metric_value
            self.convergence_step[state][metric_name] = step

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json(self.convergence_step, join(final_dir, "convergence_step.json"))
        return self.convergence_step


class ConvergenceAlmostBinary(Observable):

    def __init__(self, freq=200, *args, **kwargs):
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

        if not self.has_converged.get(layer_idx, False) and is_converged:
            self.convergence_step[layer_idx] = step

        if not is_converged:
            self.convergence_step[layer_idx] = -1

        self.has_converged[layer_idx] = is_converged

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json(self.convergence_step, join(final_dir, "convergence_step.json"))
        return self.convergence_step



class ConvergenceBinary(ObservableLayers):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convergence_step = {}
        self.has_converged = {}

    def on_train_batch_end_layers(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int
    ):
        if not isinstance(layer, BiSE):
            return

        selem, operation = layer.find_selem_and_operation(v1=0, v2=1)

        self.update_step(layer_idx, selem is not None, trainer.global_step)

        value = self.convergence_step.get(layer_idx, -1)

        trainer.logger.experiment.add_scalar(
            f"comparative/convergence/binary_{layer_idx}",
            value, trainer.global_step
        )

        trainer.logger.log_metrics(
            {f"convergence/binary_{layer_idx}": value}, trainer.global_step
        )

    def update_step(self, layer_idx, is_converged, step):

        if not self.has_converged.get(layer_idx, False) and is_converged:
            self.convergence_step[layer_idx] = step

        if not is_converged:
            self.convergence_step[layer_idx] = -1

        self.has_converged[layer_idx] = is_converged


    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json(self.convergence_step, join(final_dir, "convergence_step.json"))
        return self.convergence_step
