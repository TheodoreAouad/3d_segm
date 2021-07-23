from typing import Dict, Callable, List

from .opening_net import OpeningNet
from general.nn.pytorch_lightning_module.obs_lightning_module import NetLightning


class LightningOpeningNet(NetLightning):

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
            model=OpeningNet(**model_args),
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