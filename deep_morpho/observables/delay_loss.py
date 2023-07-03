from typing import List
import pytorch_lightning as pl

from general.nn.observables import Observable
from general.nn.loss import LossHandler


class DelayLossBatchStep(Observable):
    """ Activates a loss term after a given number of batch.
    """
    def __init__(self, delay_steps: int, keys: List[str]):
        self.delay_steps = delay_steps
        self.keys = keys
        self._done = False
        self.loss_coefs = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not isinstance(pl_module.loss, LossHandler):
            return

        self.loss_coefs = pl_module.loss.coefs.copy()

        for key in self.keys:
            pl_module.loss.coefs[key] = 0


    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx: int, dataloader_idx: int
    ) -> None:
        if trainer.global_step < self.delay_steps:
            return

        if not self._done:
            self._done = True
            if not isinstance(pl_module.loss, LossHandler):
                return

            for key in self.keys:
                pl_module.loss.do_compute[key] = True
                pl_module.loss.loss_coefs[key] = self.loss_coefs[key]
