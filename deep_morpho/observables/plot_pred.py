from typing import Any, Dict
import pathlib
from os.path import join
import random

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt

from general.nn.observables import Observable


class PlotPreds(Observable):

    def __init__(self, freq: Dict = {"train": 100, "val": 10}, fig_kwargs={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.idx = {"train": 0, "val": 0}
        self.saved_fig = {"train": None, "val": None}
        self.fig_kwargs = fig_kwargs
        self.fig_kwargs['vmin'] = self.fig_kwargs.get('vmin', 0)
        self.fig_kwargs['vmax'] = self.fig_kwargs.get('vmax', 1)

    def on_validation_batch_end_with_preds(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        preds: Any
    ) -> None:
        if self.idx['val'] % self.freq['val'] == 0:
            idx = random.choice(range(len(batch[0])))
            img, target = batch[0][idx], batch[1][idx]
            pred = preds[idx]
            fig = self.plot_three(*[k.cpu().detach().numpy() for k in [img, pred, target]], title=f'val | epoch {trainer.current_epoch}')
            trainer.logger.experiment.add_figure("preds/val/input_pred_target", fig, self.idx['val'])
            self.saved_fig['val'] = fig

        self.idx['val'] += 1


    def on_train_batch_end_with_preds(
            self,
            trainer: 'pl.Trainer',
            pl_module: 'pl.LightningModule',
            outputs: 'STEP_OUTPUT',
            batch: 'Any',
            batch_idx: int,
            preds: 'Any',
    ) -> None:
        if self.idx['train'] % self.freq["train"] == 0:
            with torch.no_grad():
                # idx = random.choice(range(len(batch[0])))
                idx = 0
                img, target = batch[0][idx], batch[1][idx]
                pred = preds[idx]
                fig = self.plot_three(*[k.cpu().detach().numpy() for k in [img, pred, target]], title='train', fig_kwargs=self.fig_kwargs)
                trainer.logger.experiment.add_figure("preds/train/input_pred_target", fig, trainer.global_step)
                self.saved_fig['train'] = fig

        self.idx['train'] += 1

    @staticmethod
    def plot_three(img, pred, target, title='', fig_kwargs={"vmin": 0, "vmax": 1}):
        ncols = max(img.shape[0], pred.shape[0])
        fig, axs = plt.subplots(3, ncols, figsize=(4 * ncols, 4 * 3), squeeze=False)
        fig.suptitle(title)

        for chan in range(img.shape[0]):
            axs[0, chan].imshow(img[chan], cmap='gray', **fig_kwargs)
            axs[0, chan].set_title(f'input_{chan}')

        for chan in range(pred.shape[0]):
            axs[1, chan].imshow(pred[chan], cmap='gray', **fig_kwargs)
            axs[1, chan].set_title(f'pred_{chan} vmin={pred[chan].min():.2} vmax={pred[chan].max():.2}')

        for chan in range(target.shape[0]):
            axs[2, chan].imshow(target[chan], cmap='gray', **fig_kwargs)
            axs[2, chan].set_title(f'target_{chan}')

        return fig

    def save(self, save_path: str):
        for state in ['train', 'val']:
            if self.saved_fig[state] is not None:
                final_dir = join(save_path, self.__class__.__name__)
                pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
                self.saved_fig[state].savefig(join(final_dir, f"input_pred_target_{state}.png"))
                plt.close(self.saved_fig[state])

        return self.saved_fig


class PlotPredsClassif(Observable):

    def __init__(self, freq: Dict = {"train": 100, "val": 10}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.idx = {"train": 0, "val": 0}
        self.saved_fig = {"train": None, "val": None}

    def on_validation_batch_end_with_preds(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        preds: Any
    ) -> None:
        if self.idx['val'] % self.freq['val'] == 0:
            idx = 0
            img, target = batch[0][idx], batch[1][idx]
            pred = preds[idx]
            fig = self.plot_pred(*[k.cpu().detach().numpy() for k in [img, pred, target]], title=f'val | epoch {trainer.current_epoch}')
            trainer.logger.experiment.add_figure("preds/val/input_pred_target", fig, self.idx['val'])
            self.saved_fig['val'] = fig

        self.idx['val'] += 1


    def on_train_batch_end_with_preds(
            self,
            trainer: 'pl.Trainer',
            pl_module: 'pl.LightningModule',
            outputs: 'STEP_OUTPUT',
            batch: 'Any',
            batch_idx: int,
            preds: 'Any',
    ) -> None:
        if self.idx['train'] % self.freq["train"] == 0:
            with torch.no_grad():
                # idx = random.choice(range(len(batch[0])))
                idx = 0
                img, target = batch[0][idx], batch[1][idx]
                pred = preds[idx]
                fig = self.plot_pred(*[k.cpu().detach().numpy() for k in [img, pred, target]], title='train')
                trainer.logger.experiment.add_figure("preds/train/input_pred_target", fig, trainer.global_step)
                self.saved_fig['train'] = fig

        self.idx['train'] += 1

    @staticmethod
    def plot_pred(img, pred, target, title=''):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        n_classes = len(pred)

        axs[0].imshow(img[0])
        axs[0].set_title(target.argmax())

        pred_label = pred.argmax()

        colors = ["red" for _ in range(n_classes)]
        colors[pred_label] = "green"

        axs[1].barh(range(n_classes), pred, tick_label=range(n_classes), color=colors)
        axs[1].set_xlim(0, 1)
        axs[1].set_title(f'pred: {pred.argmax()}')

        for idx, value in enumerate(pred):
            axs[1].text(value.item(), idx, f'{value.item():.2f}')

        fig.suptitle(title)

        return fig

    def save(self, save_path: str):
        for state in ['train', 'val']:
            if self.saved_fig[state] is not None:
                final_dir = join(save_path, self.__class__.__name__)
                pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
                self.saved_fig[state].savefig(join(final_dir, f"input_pred_target_{state}.png"))
                plt.close(self.saved_fig[state])

        return self.saved_fig
