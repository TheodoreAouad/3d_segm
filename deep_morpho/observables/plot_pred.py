import pathlib
from os.path import join

import torch
import matplotlib.pyplot as plt

from general.nn.observables import Observable
from general.utils import max_min_norm


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
            img, target = batch[0][0].squeeze(), batch[1][0].squeeze()
            pred = preds[0].squeeze()

            if img.ndim == 3:
                fig = self.plot_channels(*[k.cpu().detach() for k in [img, pred, target]])
            elif img.ndim == 2:
                fig = self.plot_three(*[k.cpu().detach() for k in [img, pred, target]])
            else:
                raise ValueError(f'Image has invalid dimension. Expected (W x L x H), got {img.shape}.')
            trainer.logger.experiment.add_figure("preds/input_pred_target", fig, trainer.global_step)
            self.saved_fig = fig

        self.idx += 1

    @staticmethod
    def plot_three(img, pred, target):
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('input')

        axs[1].imshow(pred, cmap='gray')
        axs[1].set_title('pred')

        axs[2].imshow(target, cmap='gray')
        axs[2].set_title('target')

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
