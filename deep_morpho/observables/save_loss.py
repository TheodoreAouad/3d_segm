from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl

from general.nn.observables import Observable


class SaveLoss(Observable):

    def __init__(self, freq: int = 100, *args, **kwargs):
        super().__init__(freq=freq, *args, **kwargs)

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
        self.freq_idx += 1
        if self.freq_idx % self.freq == 0:
            trainer.logger.experiment.add_scalars(
                "loss/train", {k: v for k, v in outputs.items() if 'loss' in k}, trainer.global_step
            )
            trainer.logged_metrics.update(
                **{f"loss/train/{k}": v for k, v in outputs.items() if 'loss' in k}
            )
        # for k, v in outputs.items():
        #     if 'loss' in k:
        #         trainer.log(f"loss/train/{k}", v)
