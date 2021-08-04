from general.nn.observables import Observable
from .observable_layers import ObservableLayers
from general.utils import max_min_norm


class PlotWeightsDilation(ObservableLayers):

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
        trainer.logger.experiment.add_image(f"weights/Normalized_{layer_idx}", layer._normalized_weight[0], trainer.global_step)
        trainer.logger.experiment.add_image(f"weights/Raw_{layer_idx}", max_min_norm(layer.weight[0]), trainer.global_step)


class PlotParametersDilation(ObservableLayers):

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
        trainer.logger.experiment.add_scalar(f"weights/bias_{layer_idx}", layer.bias, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"weights/sum_norm_weights_{layer_idx}", layer._normalized_weight.sum(), trainer.global_step)
        trainer.logger.experiment.add_scalar(f"params/weight_P_{layer_idx}", layer.weight_P, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"params/activation_P_{layer_idx}", layer.activation_P, trainer.global_step)



# class PlotParametersDilation(Observable):

#     def __init__(self, freq_imgs: int = 100, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.freq_imgs = freq_imgs
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
#         # weights = torch.stack(pl_module)
#         if self.idx % self.freq_imgs == 0:
#             trainer.logger.experiment.add_image("weights/Normalized", pl_module.model._normalized_weight[0], trainer.global_step)
#             trainer.logger.experiment.add_image("weights/Raw", max_min_norm(pl_module.model.weight[0]), trainer.global_step)
#         self.idx += 1

#         trainer.logger.experiment.add_scalar("weights/bias_", pl_module.model.bias, trainer.global_step)
#         trainer.logger.experiment.add_scalar("weights/sum_norm_weights", pl_module.model._normalized_weight.sum(), trainer.global_step)
#         trainer.logger.experiment.add_scalar("params/weight_P", pl_module.model.weight_P, trainer.global_step)
#         trainer.logger.experiment.add_scalar("params/activation_P", pl_module.model.activation_P, trainer.global_step)


# class PlotParametersMultipleDilations(Observable):

#     def __init__(self, freq_imgs: int = 100, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.freq_imgs = freq_imgs
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
#         # weights = torch.stack(pl_module)

#         if self.idx % self.freq_imgs == 0:
#             for idx, model in enumerate(pl_module.model.dilations):
#                 trainer.logger.experiment.add_image(f"weights_{idx}/Normalized", model._normalized_weight[0], trainer.global_step)
#                 trainer.logger.experiment.add_image(f"weights_{idx}/Raw", max_min_norm(model.weight[0]), trainer.global_step)
#         self.idx += 1

#         for idx, model in enumerate(pl_module.model.dilations):
#             trainer.logger.experiment.add_scalar(f"weights/bias_{idx}", model.bias, trainer.global_step)
#             trainer.logger.experiment.add_scalar(f"params/weight_P_{idx}", model.weight_P, trainer.global_step)
#             trainer.logger.experiment.add_scalar(f"params/activation_P_{idx}", model.activation_P, trainer.global_step)
