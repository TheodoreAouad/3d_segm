from typing import Any
import pathlib
from os.path import join

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from general.nn.observables import Observable


class PlotPreds(Observable):

    def __init__(self, freq: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.idx = 0
        self.saved_fig = None


    def on_train_batch_end_with_preds(
            self,
            trainer: 'pl.Trainer',
            pl_module: 'pl.LightningModule',
            outputs: 'STEP_OUTPUT',
            batch: 'Any',
            batch_idx: int,
            preds: 'Any',
    ) -> None:
        if self.idx % self.freq == 0:
            # img, target = batch[0][0], batch[1][0].unsqueeze(0)
            # # print(preds)
            # pred = preds[0].unsqueeze(0)
            # input_pred_target = torch.cat([img, pred, target], dim=2)
            # trainer.logger.experiment.add_image("preds/input_pred_target", input_pred_target, trainer.global_step)
            img, target = batch[0][0], batch[1][0]
            pred = preds[0]
            fig = self.plot_three(*[k.cpu().detach() for k in [img, pred, target]])

            # if img.ndim == 3:
            #     fig = self.plot_channels(*[k.cpu().detach() for k in [img, pred, target]])
            # elif img.ndim == 2:
            #     fig = self.plot_three(*[k.cpu().detach() for k in [img, pred, target]])
            # else:
            #     raise ValueError(f'Image has invalid dimension. Expected (W x L x H), got {img.shape}.')
            trainer.logger.experiment.add_figure("preds/input_pred_target", fig, trainer.global_step)
            self.saved_fig = fig

        self.idx += 1

    @staticmethod
    def plot_three(img, pred, target):
        ncols = max(img.shape[0], pred.shape[0])
        fig, axs = plt.subplots(3, ncols, figsize=(4 * ncols, 4 * 3))

        for chan in range(img.shape[0]):
            axs[0, chan].imshow(img[chan], cmap='gray')
            axs[0, chan].set_title(f'input_{chan}')

        for chan in range(pred.shape[0]):
            axs[1, chan].imshow(pred[chan], cmap='gray')
            axs[1, chan].set_title(f'pred_{chan}')

        for chan in range(target.shape[0]):
            axs[2, chan].imshow(target[chan], cmap='gray')
            axs[2, chan].set_title(f'target_{chan}')

        return fig

    @staticmethod
    def plot_channels(img, pred, target):
        ncols = max(img.shape[-1], 2)
        fig, axs = plt.subplots(2, ncols, figsize=(ncols*7, 2*7))

        for chan in range(img.shape[-1]):
            axs[0, chan].imshow(img[..., chan], cmap='gray')
            axs[0, chan].set_title(f'input_{chan}')

        axs[1, 0].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axs[1, 0].set_title('pred')

        axs[1, 1].imshow(target, cmap='gray')
        axs[1, 1].set_title('target')

        return fig

    def save(self, save_path: str):
        if self.saved_fig is not None:
            final_dir = join(save_path, self.__class__.__name__)
            pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
            self.saved_fig.savefig(join(final_dir, "input_pred_target.png"))

        return self.saved_fig
