from general.nn.observables import Observable
from deep_morpho.viz.bimonn_viz import BimonnVizualiser


class PlotBimonn(Observable):

    def __init__(self, freq: int = 300, figsize=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.freq_idx = 0
        self.figsize = figsize

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

            vizualiser = BimonnVizualiser(pl_module.model, mode="weight")
            fig = vizualiser.get_fig(figsize=self.figsize)
            trainer.logger.experiment.add_figure("model/weight", fig, trainer.global_step)

            vizualiser = BimonnVizualiser(pl_module.model, mode="selem")
            fig = vizualiser.get_fig(figsize=self.figsize)
            trainer.logger.experiment.add_figure("model/selem", fig, trainer.global_step)

        self.freq_idx += 1
