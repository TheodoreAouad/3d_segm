from typing import Dict, Callable, List

from .bimonn import (
    BiMoNN, BiMoNNClassifierMaxPool, BiMoNNClassifierMaxPoolNotBinary, BiMoNNClassifierLastLinearNotBinary,
    BiMoNNClassifierLastLinear
)
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
        **kwargs
    ):
        super().__init__(
            model=BiMoNN(**model_args),
            learning_rate=learning_rate,
            loss=loss,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            observables=observables,
            **kwargs
        )

        self.save_hyperparameters(ignore="observables")
        self.hparams["model_type"] = self.__class__.__name__



class LightningBiMoNNClassifierMaxPool(NetLightning):

    def __init__(
        self,
        model_args: Dict,
        learning_rate: float,
        loss: Callable,
        optimizer: Callable,
        optimizer_args: Dict = {},
        observables: List["Observable"] = [],
        **kwargs
    ):
        super().__init__(
            model=BiMoNNClassifierMaxPool(**model_args),
            learning_rate=learning_rate,
            loss=loss,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            observables=observables,
            **kwargs
        )

        self.save_hyperparameters(ignore="observables")
        self.hparams["model_type"] = self.__class__.__name__



class LightningBiMoNNClassifierMaxPoolNotBinary(NetLightning):

    def __init__(
        self,
        model_args: Dict,
        learning_rate: float,
        loss: Callable,
        optimizer: Callable,
        optimizer_args: Dict = {},
        observables: List["Observable"] = [],
        **kwargs
    ):
        super().__init__(
            model=BiMoNNClassifierMaxPoolNotBinary(**model_args),
            learning_rate=learning_rate,
            loss=loss,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            observables=observables,
            **kwargs
        )

        self.save_hyperparameters(ignore="observables")
        self.hparams["model_type"] = self.__class__.__name__



class LightningBiMoNNClassifierLastLinearNotBinary(NetLightning):

    def __init__(
        self,
        model_args: Dict,
        learning_rate: float,
        loss: Callable,
        optimizer: Callable,
        optimizer_args: Dict = {},
        observables: List["Observable"] = [],
        **kwargs
    ):
        super().__init__(
            model=BiMoNNClassifierLastLinearNotBinary(**model_args),
            learning_rate=learning_rate,
            loss=loss,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            observables=observables,
            **kwargs
        )

        self.save_hyperparameters(ignore="observables")
        self.hparams["model_type"] = self.__class__.__name__



class LightningBiMoNNClassifierLastLinear(NetLightning):

    def __init__(
        self,
        model_args: Dict,
        learning_rate: float,
        loss: Callable,
        optimizer: Callable,
        optimizer_args: Dict = {},
        observables: List["Observable"] = [],
        **kwargs
    ):
        super().__init__(
            model=BiMoNNClassifierLastLinear(**model_args),
            learning_rate=learning_rate,
            loss=loss,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            observables=observables,
            **kwargs
        )

        self.save_hyperparameters(ignore="observables")
        self.hparams["model_type"] = self.__class__.__name__

