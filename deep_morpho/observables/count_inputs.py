from general.nn.observables import Observable


class CountInputs(Observable):

    def __init__(self):
        super().__init__()
        self.n_inputs = 0

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: 'STEP_OUTPUT',
        batch: 'Any',
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.n_inputs += len(batch[0])
        trainer.logger.experiment.add_scalar("n_inputs", self.n_inputs, trainer.global_step)
