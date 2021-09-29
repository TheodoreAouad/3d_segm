import torch
from general.nn.observables import Observable
from general.utils import max_min_norm
from .observable_layers import ObservableLayers
from ..models import BiSE, BiSEC, COBiSEC, COBiSE


class WeightsHistogramBiSE(ObservableLayers):

    def __init__(self, *args, freq: int = 100, **kwargs):
        super().__init__(*args, freq=freq, **kwargs)

    def on_train_batch_end_layers(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
    ):
        if isinstance(layer, (BiSE, BiSEC, COBiSEC, COBiSE)):
            trainer.logger.experiment.add_histogram(f"weights_hist/Normalized_{layer_idx}", layer._normalized_weight, trainer.global_step)
        trainer.logger.experiment.add_histogram(f"weights_hist/Raw_{layer_idx}", layer.weight, trainer.global_step)

# class WeightsHistogramDilation(Observable):

#     def __init__(self, freq: int = 100, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.freq = freq
#         self.idx = 0

#     def on_train_batch_end(
#         self,
#         trainer: 'pl.Trainer',
#         pl_module: 'pl.LightningModule',
#         outputs: "STEP_OUTPUT",
#         batch: "Any",
#         batch_idx: int,
#         dataloader_idx: int,
#     ) -> None:
#         """Called when the train batch ends."""

#         if self.idx % self.freq == 0:
#             trainer.logger.experiment.add_histogram("weights_hist/Normalized", pl_module.model._normalized_weight[0],
#                                                 trainer.global_step)
#             trainer.logger.experiment.add_histogram("weights_hist/Raw", pl_module.model.weight[0], trainer.global_step)
#         self.idx += 1


# class WeightsHistogramMultipleDilations(Observable):

#     def __init__(self, freq: int = 100, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.freq = freq
#         self.idx = 0

#     def on_train_batch_end(
#         self,
#         trainer: 'pl.Trainer',
#         pl_module: 'pl.LightningModule',
#         outputs: "STEP_OUTPUT",
#         batch: "Any",
#         batch_idx: int,
#         dataloader_idx: int,
#     ) -> None:
#         """Called when the train batch ends."""

#         if self.idx % self.freq == 0:
#             for idx, model in enumerate(pl_module.model.dilations):
#                 trainer.logger.experiment.add_histogram(f"weights_hist_{idx}/Normalized", model._normalized_weight[0],
#                                                     trainer.global_step)
#                 trainer.logger.experiment.add_histogram(f"weights_hist_{idx}/Raw", model.weight[0], trainer.global_step)
#         self.idx += 1
