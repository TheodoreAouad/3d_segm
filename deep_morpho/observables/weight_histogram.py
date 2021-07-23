import torch
from general.nn.observables import Observable
from general.utils import max_min_norm


class WeightsHistogramDilation(Observable):

    def __init__(self, freq: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.idx = 0

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""

        if self.idx % self.freq == 0:
            trainer.logger.experiment.add_histogram("weights_hist/Normalized", pl_module.model._normalized_weight[0],
                                                trainer.global_step)
            trainer.logger.experiment.add_histogram("weights_hist/Raw", max_min_norm(pl_module.model.weight[0]), trainer.global_step)
            trainer.logger.experiment.add_histogram("weights_hist/Sigmoid", torch.sigmoid(pl_module.model.weight[0]),
                                                trainer.global_step)
        self.idx += 1


class WeightsHistogramMultipleDilations(Observable):

    def __init__(self, freq: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.idx = 0

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""

        if self.idx % self.freq == 0:
            for idx, model in enumerate(pl_module.model.dilations):
                trainer.logger.experiment.add_histogram(f"weights_hist_{idx}/Normalized", model._normalized_weight[0],
                                                    trainer.global_step)
                trainer.logger.experiment.add_histogram(f"weights_hist_{idx}/Raw", max_min_norm(model.weight[0]), trainer.global_step)
                trainer.logger.experiment.add_histogram(f"weights_hist_{idx}/Sigmoid", torch.sigmoid(model.weight[0]),
                                                    trainer.global_step)
        self.idx += 1
