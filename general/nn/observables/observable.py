import pytorch_lightning as pl
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT


class Observable(pl.callbacks.Callback):
    """
        Abstract base class for training, validation and testing hooks.
        Same logic as pytorch_lightning callbacks. But as of version 1.3.0 of PL,
        these methods do not take the predictions as arguments, so we have to recompute them.
        To avoid this, we create additional methods to catch them (see class MODULE2 #TODO)

    """

    def on_train_batch_end_with_preds(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            preds: Any,
    ) -> None:
        pass

    def on_validation_batch_end_with_preds(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            preds: Any,
    ) -> None:
        pass

    def on_test_batch_end_with_preds(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            preds: Any,
    ) -> None:
        pass
