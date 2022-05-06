import pathlib
from os.path import join

import torch

from general.nn.observables import Observable
from deep_morpho.viz.bimonn_viz import BimonnForwardVizualiser, BimonnHistogramVizualiser


class PlotBimonnForward(Observable):

    def __init__(self, freq: int = 300, figsize=None, do_plot={"float": True, "binary": True}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.freq_idx = 0
        self.figsize = figsize
        self.do_plot = do_plot

        self.last_figs = {}

    def on_train_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
    ) -> None:
        with torch.no_grad():
            inpt = batch[0][0].unsqueeze(0)
            if self.freq_idx % self.freq == 0:
                for key, do_key in self.do_plot.items():
                    if do_key:
                        if key == "binary":
                            pl_module.model.binary(True)
                        else:
                            pl_module.model.binary(False)

                        vizualiser = BimonnForwardVizualiser(pl_module.model, mode=key, inpt=inpt)
                        fig = vizualiser.get_fig(figsize=self.figsize)
                        trainer.logger.experiment.add_figure(f"forward/{key}", fig, trainer.global_step)
                        self.last_figs[key] = fig

        self.freq_idx += 1
        pl_module.model.binary(False)


    def save(self, save_path: str):
        if len(self.last_figs) == 0:
            return

        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        for key, fig in self.last_figs.items():
            fig.savefig(join(final_dir, f"forward_{key}.png"))

        return self.last_figs


class PlotBimonnHistogram(Observable):

    def __init__(self, freq: int = 300, figsize=None, do_plot={"float": True, "binary": True}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.freq_idx = 0
        self.figsize = figsize
        self.do_plot = do_plot

        self.last_figs = {}

    def on_train_batch_end_with_preds(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        preds: "Any",
    ) -> None:
        with torch.no_grad():
            inpt = batch[0][0].unsqueeze(0)
            if self.freq_idx % self.freq == 0:
                for key, do_key in self.do_plot.items():
                    if do_key:
                        if key == "binary":
                            pl_module.model.binary(True)
                        else:
                            pl_module.model.binary(False)

                        vizualiser = BimonnHistogramVizualiser(pl_module.model, mode=key, inpt=inpt)
                        fig = vizualiser.get_fig(figsize=self.figsize)
                        trainer.logger.experiment.add_figure(f"histogram/{key}", fig, trainer.global_step)
                        self.last_figs[key] = fig

        self.freq_idx += 1
        pl_module.model.binary(False)


    def save(self, save_path: str):
        if len(self.last_figs) == 0:
            return

        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        for key, fig in self.last_figs.items():
            fig.savefig(join(final_dir, f"histogram_{key}.png"))

        return self.last_figs
