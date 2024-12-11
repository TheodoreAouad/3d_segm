from pytorch_lightning import LightningModule
from typing import Any, List, Optional, Callable, Dict
from ..observables.observable import Observable
from pytorch_lightning.utilities.types import EPOCH_OUTPUT


class ObsLightningModule(LightningModule):

    def __init__(self, observables=None, *args, **kwargs):
        """
        Initialize the list of observables.

        Args:
            self: write your description
            observables: write your description
        """
        super().__init__(*args, **kwargs)
        self.observables: Optional[List[Observable]] = observables

    def training_step(self, batch: Any, batch_idx: int):
        """
        Run the batch - level training step.

        Args:
            self: write your description
            batch: write your description
            batch_idx: write your description
        """
        outputs, preds = self.obs_training_step(batch, batch_idx)

        for obs in self.observables:
            obs.on_train_batch_end_with_preds(
                self.trainer,
                self.trainer.lightning_module,
                outputs,
                batch,
                batch_idx,
                preds
            )

        return outputs

    def validation_step(self, batch: Any, batch_idx: int):
        """
        Run the batch validation step.

        Args:
            self: write your description
            batch: write your description
            batch_idx: write your description
        """
        outputs, preds = self.obs_validation_step(batch, batch_idx)

        for obs in self.observables:
            obs.on_validation_batch_end_with_preds(
                self.trainer,
                self.trainer.lightning_module,
                outputs,
                batch,
                batch_idx,
                preds
            )
        return outputs

    def test_step(self, batch: Any, batch_idx: int):
        """
        Run a test step.

        Args:
            self: write your description
            batch: write your description
            batch_idx: write your description
        """
        outputs, preds = self.obs_test_step(batch, batch_idx)

        for obs in self.observables:
            obs.on_test_batch_end_with_preds(
                self.trainer,
                self.trainer.lightning_module,
                outputs,
                batch,
                batch_idx,
                preds
            )

        return outputs

    def test_epoch_end(self, outputs: EPOCH_OUTPUT):
        """
        Test the end of each epoch.

        Args:
            self: write your description
            outputs: write your description
        """
        self.obs_test_epoch_end(outputs)
        for obs in self.observables:
            obs.on_test_epoch_end(self.trainer, self.trainer.lightning_module)

    def obs_training_step(self, batch: Any, batch_idx: int):
        """
        One observation training step.

        Args:
            self: write your description
            batch: write your description
            batch_idx: write your description
        """
        raise NotImplementedError

    def obs_validation_step(self, batch: Any, batch_idx: int):
        """
        One observation validation step.

        Args:
            self: write your description
            batch: write your description
            batch_idx: write your description
        """
        raise NotImplementedError

    def obs_test_step(self, batch: Any, batch_idx: int):
        """
        Test the observation function.

        Args:
            self: write your description
            batch: write your description
            batch_idx: write your description
        """
        raise NotImplementedError

    def obs_test_epoch_end(self, outputs: EPOCH_OUTPUT):
        """
        End of the test epoch for observing test data.

        Args:
            self: write your description
            outputs: write your description
        """
        pass


class NetLightning(ObsLightningModule):
    def __init__(
            self,
            model: "nn.Module",
            learning_rate: float,
            loss: Callable,
            optimizer: Callable,
            output_dir: str,
            optimizer_args: Dict = {},
            observables: Optional[List[Observable]] = [],
    ):
        """
        Initialize the hyperparameters of the optimizer.

        Args:
            self: write your description
            model: write your description
            learning_rate: write your description
            loss: write your description
            optimizer: write your description
            output_dir: write your description
            optimizer_args: write your description
            observables: write your description
            Optional: write your description
            List: write your description
            Observable: write your description
        """

        super().__init__(observables)
        self.model = model
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.output_dir = output_dir
        # self.save_hyperparameters()

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: write your description
            x: write your description
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        Configures the optimizers for the model.

        Args:
            self: write your description
        """
        return self.optimizer(self.parameters(), lr=self.learning_rate, **self.optimizer_args)

    def obs_training_step(self, batch, batch_idx):
        """
        Perform obs training step.

        Args:
            self: write your description
            batch: write your description
            batch_idx: write your description
        """
        x, y = batch
        predictions = self.forward(x)

        loss = self.loss(predictions, y)
        self.log('loss/train_loss', loss)

        return {'loss': loss}, predictions

    def obs_validation_step(self, batch, batch_idx):
        """
        One step of obs validation.

        Args:
            self: write your description
            batch: write your description
            batch_idx: write your description
        """
        x, y = batch
        predictions = self.forward(x)

        loss = self.loss(predictions, y)
        self.log('loss/val_loss', loss)

        return {'val_loss': loss}, predictions

    def obs_test_step(self, batch, batch_idx):
        """
        One step of the observation test loop.

        Args:
            self: write your description
            batch: write your description
            batch_idx: write your description
        """
        x, y = batch
        predictions = self.forward(x)

        loss = self.loss(predictions, y)
        self.log('loss/test_loss', loss)

        return {'test_loss': loss}, predictions
