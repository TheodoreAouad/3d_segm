import pathlib
from os.path import join

import torch
import matplotlib.pyplot as plt

from general.nn.observables.observable import Observable
from general.utils import save_json



class BinaryModeMetric(Observable):

    def __init__(self, metrics, freq=100):
        self.metrics = metrics
        self.freq = freq
        self.freq_idx = 0
        self.last_value = {}


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

            img, pred, target = inputs[0], preds[0], targets[0]
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
