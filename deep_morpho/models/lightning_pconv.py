from typing import Dict, Callable, List

from .pconv import PConv2d
from general.nn.pytorch_lightning_module.obs_lightning_module import NetLightning


class LightningPConv2d(NetLightning):

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
        """
        Explicitly save the hyperparameters of the model.

        Args:
            self: write your description
            model_args: write your description
            learning_rate: write your description
            loss: write your description
            optimizer: write your description
            output_dir: write your description
            optimizer_args: write your description
            observables: write your description
            List: write your description
        """
        super().__init__(
            model=PConv2d(**model_args),
            learning_rate=learning_rate,
            loss=loss,
            optimizer=optimizer,
            output_dir=output_dir,
            optimizer_args=optimizer_args,
            observables=observables,
        )
        self.save_hyperparameters()
