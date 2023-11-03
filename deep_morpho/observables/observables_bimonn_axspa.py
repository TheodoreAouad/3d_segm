import pathlib
from os.path import join
from typing import Any, Dict

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
import numpy as np

from general.nn.observables import Observable
from general.utils import save_json
from .binary_mode_activatedness import ActivatednessObservable
from .plot_model import PlotBimonn
from .binary_mode_metric import BinaryModeMetricBase
from .observable_layers import ObservableLayersChans
from .weight_histogram import WeightsHistogramBiSE
from .plot_lui_parameters import PlotLUIParametersBiSEL
from .plot_parameters import PlotParametersBiSE
from .plot_forward import PlotBimonnForward




def prep_segm(cur_segm):
        """From cur_segm in shape (channels, W, H) to segm in shape (W, H) with multiple values"""
        segm = np.zeros((cur_segm.shape[1], cur_segm.shape[2]))
        segm[cur_segm[0] == 1] = 1
        segm[cur_segm[1] == 1] = 2
        return segm


class PlotPredsBimonnAxspa(Observable):
    def __init__(
        self,
        freq_batch: Dict = {"train": 100, "val": 10, "test": 10},
        figsize_atom=(4, 4),
        n_imgs=10,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.freq_batch = freq_batch
        self.batch_idx = {"train": 0, "val": 0, "test": 0}
        self.saved_fig = {"train": None, "val": None}
        self.figsize_atom = figsize_atom
        self.n_imgs = n_imgs

    def on_validation_batch_end_with_preds(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: "STEP_OUTPUT",
        batch: Any,
        batch_idx: int,
        preds: Any
    ) -> None:
        if self.freq_batch["val"] is not None and self.batch_idx["val"] % self.freq_batch["val"] == 0:
            self.plot_pred_state(
                trainer=trainer, pl_module=pl_module, batch=batch, preds=preds, state="batch_val",
                title=f'val | epoch {trainer.current_epoch}', step=self.batch_idx["val"])
        self.batch_idx["val"] += 1

    def on_train_batch_end_with_preds(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: 'STEP_OUTPUT',
        batch: 'Any',
        batch_idx: int,
        preds: 'Any',
    ) -> None:
        if self.freq_batch["train"] is not None and self.batch_idx["train"] % self.freq_batch["train"] == 0:
            self.plot_pred_state(
                trainer=trainer, pl_module=pl_module, batch=batch, preds=preds, state="batch_train", title="train",
                step=trainer.global_step
            )
        self.batch_idx["train"] += 1

    def on_test_batch_end_with_preds(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: 'STEP_OUTPUT',
        batch: 'Any',
        batch_idx: int,
        preds: 'Any',
    ) -> None:
        if self.freq_batch["test"] is not None and self.batch_idx["test"] % self.freq_batch["test"] == 0:
            self.plot_pred_state(
                trainer=trainer, pl_module=pl_module, batch=batch, preds=preds, state="batch_test", title="test",
                step=self.batch_idx["test"]
            )
        self.batch_idx["test"] += 1


    def plot_pred_state(self, trainer, pl_module, batch, preds, state, title, step):
        with torch.no_grad():
            (imgs, segms), targets = batch
            # imgs = [k.cpu().detach().numpy().transpose(1, 2, 0) for k in imgs[0]]
            img = imgs[0, 0].cpu().detach().numpy()

            segm = prep_segm(segms[0].cpu().detach().numpy())
            pred_segm = pl_module.model.current_output["bimonn"][0, 0].cpu().detach().numpy()
            pred_label = pl_module.model.current_output["pred"][0].item()

            target = targets[0].item()
            fig = self.plot_pred(
                img=img,
                # *[k.cpu().detach().numpy() for k in [preds, batch[1]]],
                segm=segm,
                pred_segm=pred_segm,
                pred_label=pred_label,
                target=target,
                figsize_atom=self.figsize_atom,
                n_imgs=self.n_imgs,
            )
            trainer.logger.experiment.add_figure(f"preds/{state}/input_pred_target", fig, step)
            self.saved_fig[state] = fig


    @staticmethod
    def plot_pred(img, segm, pred_segm, pred_label, target, figsize_atom, n_imgs, ):
        W, L = figsize_atom
        fig, axs = plt.subplots(3, 1, figsize=(1*W, 3*L))

        axs[0].imshow(img, cmap="gray")
        axs[0].imshow(np.ma.masked_where(segm == 0, segm), alpha=0.5)
        axs[0].set_title("Input Img")

        axs[1].imshow(pred_segm, cmap="gray", vmin=0, vmax=1)
        axs[1].set_title(f"Bimonn Output. Min: {pred_segm.min():.2e}, Max: {pred_segm.max():.2e}")

        classif_input = pred_segm * img

        axs[2].imshow(classif_input, cmap="gray")
        axs[2].set_title(f"Classif Input. Min: {classif_input.min():.2e}, Max: {classif_input.max():.2e}")

        fig.suptitle(f"Target: {target}, Pred: {pred_label:.2e}")

        return fig

    def save(self, save_path: str):
        for state in ['train', 'val']:
            if self.saved_fig[state] is not None:
                final_dir = join(save_path, self.__class__.__name__)
                pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
                self.saved_fig[state].savefig(join(final_dir, f"input_pred_target_{state}.png"))
                plt.close(self.saved_fig[state])

        return self.saved_fig


class PlotPredsBimonnAxspaMerged(PlotPredsBimonnAxspa):

    def plot_pred_state(self, trainer, pl_module, batch, preds, state, title, step):
        with torch.no_grad():
            imgs, targets = batch
            # imgs = [k.cpu().detach().numpy().transpose(1, 2, 0) for k in imgs[0]]
            img = imgs[0, 0].cpu().detach().numpy()

            pred_label = preds[0].item()

            target = targets[0].item()
            fig = self.plot_pred(
                img=img,
                # *[k.cpu().detach().numpy() for k in [preds, batch[1]]],
                pred_label=pred_label,
                target=target,
                figsize_atom=self.figsize_atom,
                n_imgs=self.n_imgs,
            )
            trainer.logger.experiment.add_figure(f"preds/{state}/input_pred_target", fig, step)
            self.saved_fig[state] = fig


    @staticmethod
    def plot_pred(img, pred_label, target, figsize_atom, n_imgs, ):
        W, L = figsize_atom
        fig, ax = plt.subplots(1, 1, figsize=(W, L))

        ax.imshow(img, cmap="gray")

        fig.suptitle(f"Target: {target}, Pred: {pred_label:.2e}")

        return fig

    def save(self, save_path: str):
        for state in ['train', 'val']:
            if self.saved_fig[state] is not None:
                final_dir = join(save_path, self.__class__.__name__)
                pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)
                self.saved_fig[state].savefig(join(final_dir, f"input_pred_target_{state}.png"))
                plt.close(self.saved_fig[state])

        return self.saved_fig


class ActivatednessObservableBimonnAxspa(ActivatednessObservable):
    def _get_layers(self, pl_module):
        return pl_module.model.bimonn.layers


class PlotBimonnBimonAxspa(PlotBimonn):
    def get_model(self, pl_module):
        return pl_module.model.bimonn


class BinaryModeMetricBimonnAxspa(BinaryModeMetricBase):
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
        if self.freq[state] is None and self.plot_freq[state] is None:
            return

        if not self.do_metric_update(state) and not self.do_plot_update(state):
            self.plot_freq_idx[state] += 1
            self.freq_idx[state] += 1
            return

        if state == "test":
            step = self.test_step
            self.test_step += 1
        else:
            step = trainer.global_step

        with torch.no_grad():
            pl_module.model.binary(update_binaries=self.update_binaries)

            inputs, targets = batch

            preds = pl_module.model(inputs)
            self.n_inputs[state] += targets.shape[0]

            if self.freq_idx[state] % self.freq[state] == 0:
                for metric_name in self.metrics:
                    metric = self.metrics[metric_name](targets, preds)
                    self.metrics_sum[state][metric_name] += metric * targets.shape[0]

                    # pl_module.log(f"mean_metrics_{batch_or_epoch}/{metric_name}/{state}", metric)
                    pl_module.log(f"binary_mode/{metric_name}_{state}", metric)

                    trainer.logger.experiment.add_scalars(
                        f"comparative/binary_mode/{metric_name}", {state: metric}, step
                    )

                    trainer.logger.log_metrics(
                        {f"binary_mode/{metric_name}_{state}": metric}, step
                    )
                    trainer.logged_metrics.update(
                        {f"binary_mode/{metric_name}_{state}": metric}
                    )

            # if self.plot_freq[state] is not None and self.plot_freq_idx[state] % self.plot_freq[state] == 0:
            if self.do_plot_update(state):
                (imgs, segms), targets = batch

                img = imgs[0, 0].cpu().detach().numpy()

                segm = prep_segm(segms[0].cpu().detach().numpy())
                pred_segm = pl_module.model.current_output["bimonn"][0, 0].cpu().detach().numpy()
                pred_label = pl_module.model.current_output["pred"][0].item()

                target = targets[0].item()

                fig = fig = self.plot_step(
                    img=img,
                    # *[k.cpu().detach().numpy() for k in [preds, batch[1]]],
                    segm=segm,
                    pred_segm=pred_segm,
                    pred_label=pred_label,
                    target=target,
                    figsize_atom=self.figsize_atom,
                    n_imgs=self.n_imgs,
                )
                trainer.logger.experiment.add_figure(f"preds/{state}/binary_mode/input_pred_target", fig, step)

            self.plot_freq_idx[state] += 1
            self.freq_idx[state] += 1

            pl_module.model.binary(False)

    @staticmethod
    def plot_step(img, segm, pred_segm, pred_label, target, figsize_atom, n_imgs, ):
        return PlotPredsBimonnAxspa.plot_pred(img, segm, pred_segm, pred_label, target, figsize_atom, n_imgs)

    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        dict_str = {}
        for k1, v1 in self.last_value.items():
            dict_str[k1] = {}
            for k2, v2 in v1.items():
                dict_str[k1][k2] = str(v2)
        save_json(dict_str, join(final_dir, "metrics.json"))
        return self.last_value



class ObservableLayersChansBimonnAxspa(ObservableLayersChans):

    def _get_layers(self, pl_module):
        return pl_module.model.bimonn.layers


class WeightsHistogramBiSEBimonnAxspa(ObservableLayersChansBimonnAxspa, WeightsHistogramBiSE):
    pass


class PlotLUIParametersBiSELBimonnAxspa(ObservableLayersChansBimonnAxspa, PlotLUIParametersBiSEL):
    pass


class PlotParametersBiSEBimonnAxspa(ObservableLayersChansBimonnAxspa, PlotParametersBiSE):
    pass


class PlotBimonnForwardBimonnAxspa(PlotBimonnForward):
    def _get_model(self, pl_module):
        return pl_module.model.bimonn

    def _get_input(self, pl_module: "pl.LightningModule", batch: "Any"):
        return batch[0][1][0].unsqueeze(0).to(pl_module.device)
