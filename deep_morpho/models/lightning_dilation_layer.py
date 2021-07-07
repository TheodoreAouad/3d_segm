from typing import Dict, Callable, List

from .dilation_layer import DilationLayer
from general.nn.pytorch_lightning_module.obs_lightning_module import NetLightning


class LightningDilationLayer(NetLightning):

    def __init__(
        self,
        model_args: Dict,
        learning_rate: float,
        loss: Callable,
        optimizer: Callable,
        output_dir: str,
        optimizer_args: Dict = {},
        observables: [List["Observable"]] = [],
    ):
        super().__init__(
            model=DilationLayer(**model_args),
            learning_rate=learning_rate,
            loss=loss,
            optimizer=optimizer,
            output_dir=output_dir,
            optimizer_args=optimizer_args,
            observables=observables,
        )
        self.save_hyperparameters()

    def obs_training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.forward(x).squeeze()

        loss = self.loss
        loss = loss(predictions, y)
        self.log('loss/train_loss', loss)

        return {'loss': loss}, predictions