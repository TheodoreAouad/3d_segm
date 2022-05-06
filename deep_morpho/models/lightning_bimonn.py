from typing import Dict, Callable, List

from .bimonn import BiMoNN, BiMoNNClassifier
from general.nn.pytorch_lightning_module.obs_lightning_module import NetLightning


class LightningBiMoNN(NetLightning):

    def __init__(
        self,
        model_args: Dict,
        learning_rate: float,
        loss: Callable,
        optimizer: Callable,
        optimizer_args: Dict = {},
        observables: List["Observable"] = [],
    ):
        super().__init__(
            model=BiMoNN(**model_args),
            learning_rate=learning_rate,
            loss=loss,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            observables=observables,
        )

        self.save_hyperparameters()


class LightningBiMoNNClassifier(NetLightning):

    def __init__(
        self,
        model_args: Dict,
        learning_rate: float,
        loss: Callable,
        optimizer: Callable,
        optimizer_args: Dict = {},
        observables: List["Observable"] = [],
    ):
        super().__init__(
            model=BiMoNNClassifier(**model_args),
            learning_rate=learning_rate,
            loss=loss,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            observables=observables,
        )

        self.save_hyperparameters()
