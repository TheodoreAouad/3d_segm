import pathlib
from os.path import join
from typing import Dict

import torch
import matplotlib.pyplot as plt

from general.nn.observables.observable import Observable
from .plot_pred import PlotPredsClassif
from general.utils import save_json



class BinaryModeMetric(Observable):

    def __init__(self, metrics, freq=100, do_plot_figure: bool = True):
        self.metrics = metrics
        self.freq = freq
        self.freq_idx = 0
        self.last_value = {}
        self.do_plot_figure = do_plot_figure


    def on_train_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
    ):
        if self.freq_idx % self.freq != 0:
            self.freq_idx += 1
            return
        self.freq_idx += 1

        with torch.no_grad():
            pl_module.model.binary()

            inputs, targets = batch
            inputs = (inputs > 0).float()  # handle both {0, 1} and {-1, 1}
            targets = (targets > 0).float()  # handle both {0, 1} and {-1, 1}

            preds = pl_module.model(inputs)
            for metric_name in self.metrics:
                metric = self.metrics[metric_name](targets, preds)
                self.last_value[metric_name] = metric
                # pl_module.log(f"mean_metrics_{batch_or_epoch}/{metric_name}/{state}", metric)

                trainer.logger.experiment.add_scalars(
                    f"comparative/binary_mode/{metric_name}", {'train': metric}, trainer.global_step
                )

                trainer.logger.log_metrics(
                    {f"binary_mode/{metric_name}_{'train'}": metric}, trainer.global_step
                )
                trainer.logged_metrics.update(
                    {f"binary_mode/{metric_name}_{'train'}": metric}
                )

            img, pred, target = inputs[0], preds[0], targets[0]
            if self.do_plot_figure:
                fig = self.plot_three(*[k.cpu().detach().numpy() for k in [img, pred, target]], title='train')
                trainer.logger.experiment.add_figure("preds/train/binary_mode/input_pred_target", fig, trainer.global_step)

            pl_module.model.binary(False)


    @staticmethod
    def plot_three(img, pred, target, title=''):
        ncols = max(img.shape[0], pred.shape[0])
        fig, axs = plt.subplots(3, ncols, figsize=(4 * ncols, 4 * 3), squeeze=False)
        fig.suptitle(title)

        for chan in range(img.shape[0]):
            axs[0, chan].imshow(img[chan], cmap='gray', vmin=0, vmax=1)
            axs[0, chan].set_title(f'input_{chan}')

        for chan in range(pred.shape[0]):
            axs[1, chan].imshow(pred[chan], cmap='gray', vmin=0, vmax=1)
            axs[1, chan].set_title(f'pred_{chan} vmin={pred[chan].min():.2} vmax={pred[chan].max():.2}')

        for chan in range(target.shape[0]):
            axs[2, chan].imshow(target[chan], cmap='gray', vmin=0, vmax=1)
            axs[2, chan].set_title(f'target_{chan}')

        return fig

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json({k: str(v) for k, v in self.last_value.items()}, join(final_dir, "metrics.json"))
        return self.last_value


class BinaryModeMetricClassif(Observable):

    def __init__(self, metrics, freq: Dict = {"train": 100, "val": 10, }, do_plot_figure: bool = True, figsize_atom=(4, 4), n_imgs=10, ):
        self.metrics = metrics
        self.freq = freq
        self.freq_idx = {"train": 0, "val": 0}
        self.last_value = {}
        self.do_plot_figure = do_plot_figure
        self.val_step = 0
        self.figsize_atom = figsize_atom
        self.n_imgs = n_imgs

    def on_train_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
    ):
        if self.freq_idx["train"] % self.freq["train"] != 0:
            self.freq_idx["train"] += 1
            return
        self.freq_idx["train"] += 1

        self._compute_metric_and_plot(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            preds=preds,
            state='train'
        )


    def on_validation_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
    ):
        if self.freq_idx["val"] % self.freq["val"] != 0:
            self.freq_idx["val"] += 1
            return
        self.freq_idx["val"] += 1

        self._compute_metric_and_plot(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            preds=preds,
            state='val'
        )

    def on_test_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
    ):
        # if self.freq_idx["test"] % self.freq["test"] != 0:
        #     self.freq_idx["test"] += 1
        #     return
        # self.freq_idx["test"] += 1

        self._compute_metric_and_plot(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
            preds=preds,
            state='test'
        )

    def _compute_metric_and_plot(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
        state: str,
    ):
        step = trainer.global_step
        # if state == "train":
        #     step = trainer.global_step
        # elif state == "val":
        #     step = self.val_step
        #     self.val_step += 1

        with torch.no_grad():
            pl_module.model.binary()

            inputs, targets = batch
            inputs = (inputs > 0).float()  # handle both {0, 1} and {-1, 1}
            targets = (targets > 0).float()  # handle both {0, 1} and {-1, 1}

            preds = pl_module.model(inputs)
            for metric_name in self.metrics:
                metric = self.metrics[metric_name](targets, preds)
                self.last_value[metric_name] = metric
                # pl_module.log(f"mean_metrics_{batch_or_epoch}/{metric_name}/{state}", metric)

                trainer.logger.experiment.add_scalars(
                    f"comparative/binary_mode/{metric_name}", {state: metric}, step
                )

                trainer.logger.log_metrics(
                    {f"binary_mode/{metric_name}_{state}": metric}, step
                )
                trainer.logged_metrics.update(
                    {f"binary_mode/{metric_name}_{state}": metric}
                )

            # img, pred, target = inputs[0], preds[0], targets[0]
            if self.do_plot_figure:
                fig = self.plot_pred(
                    *[k.cpu().detach().numpy() for k in [inputs, preds, targets]],
                    figsize_atom=self.figsize_atom,
                    n_imgs=self.n_imgs,
                    title=state,
                )
                # fig = self.plot_pred(*[k.cpu().detach().numpy() for k in [img, pred, target]], title=state)
                trainer.logger.experiment.add_figure(f"preds/{state}/binary_mode/input_pred_target", fig, step)

            pl_module.model.binary(False)


    @staticmethod
    def plot_pred(*args, **kwargs):
        return PlotPredsClassif.plot_pred(*args, **kwargs)


    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
        save_json({k: str(v) for k, v in self.last_value.items()}, join(final_dir, "metrics.json"))
        return self.last_value
