import pathlib
from os.path import join

from general.nn.observables import Observable
from deep_morpho.viz.bimonn_viz import BimonnVizualiser


class PlotBimonn(Observable):

    def __init__(self, freq: int = 300, figsize=None, dpi=None, do_plot={"weights": True, "learned": True, "closest": True,}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.freq_idx = 0
        self.figsize = figsize
        self.dpi = dpi
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

        if self.freq_idx % self.freq == 0:
            self.save_figs(trainer, pl_module)
            # for key, do_key in self.do_plot.items():
            #     if do_key:
            #         vizualiser = BimonnVizualiser(pl_module.model, mode=key)
            #         fig = vizualiser.get_fig(figsize=self.figsize, dpi=self.dpi)
            #         trainer.logger.experiment.add_figure(f"model/{key}", fig, trainer.global_step)
            #         self.last_figs[key] = fig

        self.freq_idx += 1

    def on_train_end(self, trainer, pl_module):
        self.save_figs(trainer, pl_module)

    def save_figs(self, trainer, pl_module):
        for key, do_key in self.do_plot.items():
            if do_key:
                vizualiser = BimonnVizualiser(pl_module.model, mode=key)
                fig = vizualiser.get_fig(figsize=self.figsize, dpi=self.dpi)
                trainer.logger.experiment.add_figure(f"model/{key}", fig, trainer.global_step)
                self.last_figs[key] = fig

    def save(self, save_path: str):
        if len(self.last_figs) == 0:
            return

        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        for key, fig in self.last_figs.items():
            fig.savefig(join(final_dir, f"model_{key}.png"))

        return self.last_figs
