import torch
from general.nn.observables import Observable
from general.utils import max_min_norm


class PlotParameters(Observable):

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
        # weights = torch.stack(pl_module)
        trainer.logger.experiment.add_image("weights/Normalized", pl_module.model._normalized_weight[0], trainer.global_step)
        trainer.logger.experiment.add_image("weights/Raw", max_min_norm(pl_module.model.weight[0]), trainer.global_step)
        trainer.logger.experiment.add_image("weights/Sigmoid", torch.sigmoid(pl_module.model.weight[0]), trainer.global_step)
        trainer.logger.experiment.add_scalar("params/P_", pl_module.model.P_, trainer.global_step)
