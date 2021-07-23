from general.nn.observables import Observable


class PlotParametersDilation(Observable):

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
        trainer.logger.experiment.add_image("weights/Raw", pl_module.model.weight[0], trainer.global_step)

        trainer.logger.experiment.add_scalar("weights/bias_", pl_module.model.bias, trainer.global_step)
        trainer.logger.experiment.add_scalar("params/weight_P", pl_module.model.weight_P, trainer.global_step)
        trainer.logger.experiment.add_scalar("params/activation_P", pl_module.model.activation_P, trainer.global_step)


class PlotParametersMultipleDilation(Observable):

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
        for idx, model in enumerate(pl_module.model.dilations):
            trainer.logger.experiment.add_image(f"weights_{idx}/Normalized", model._normalized_weight[0], trainer.global_step)
            trainer.logger.experiment.add_image(f"weights_{idx}/Raw", model.weight[0], trainer.global_step)

            trainer.logger.experiment.add_scalar(f"weights/bias_{idx}", model.bias, trainer.global_step)
            trainer.logger.experiment.add_scalar(f"params/weight_P_{idx}", model.weight_P, trainer.global_step)
            trainer.logger.experiment.add_scalar(f"params/activation_P_{idx}", model.activation_P, trainer.global_step)
