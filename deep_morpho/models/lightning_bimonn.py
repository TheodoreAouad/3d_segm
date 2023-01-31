from typing import Dict, Callable, List

from .bimonn import (
    BiMoNN, BiMoNNClassifierMaxPool, BiMoNNClassifierMaxPoolNotBinary, BiMoNNClassifierLastLinearNotBinary,
    BiMoNNClassifierLastLinear
)
from .specific_bimonn import BimonnDense, BimonnDenseNotBinary, BimonnBiselDenseNotBinary
from general.nn.pytorch_lightning_module.obs_lightning_module import NetLightning
from general.nn.observables import Observable


# def lightning_init(cls):
#     def __init__(self, model_args, **kwargs):
#         super(type(self), self).__init__(model=cls(**model_args), **kwargs)
#         self.save_hyperparameters(ignore="observables")
#         self.hparams["model_type"] = self.__class__.__name__
#     return __init__

# TODO: Factorize all this code

class GenericLightningModel(NetLightning):

    def __init__(
        self,
        model_args: Dict,
        learning_rate: float,
        loss: Callable,
        optimizer: Callable,
        optimizer_args: Dict = {},
        observables: List[Observable] = [],
        **kwargs
    ):
        super().__init__(
            model=self.model_class(**model_args),
            learning_rate=learning_rate,
            loss=loss,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            observables=observables,
            **kwargs
        )

        self.save_hyperparameters(ignore="observables")
        self.hparams["model_type"] = self.__class__.__name__


class LightningBiMoNN(GenericLightningModel):
    model_class = BiMoNN


class LightningBiMoNNClassifierMaxPool(GenericLightningModel):
    model_class = BiMoNNClassifierMaxPool


class LightningBiMoNNClassifierMaxPoolNotBinary(GenericLightningModel):
    model_class = BiMoNNClassifierMaxPoolNotBinary


class LightningBiMoNNClassifierLastLinearNotBinary(GenericLightningModel):
    model_class = BiMoNNClassifierLastLinearNotBinary


class LightningBiMoNNClassifierLastLinear(GenericLightningModel):
    model_class = BiMoNNClassifierLastLinear


class LightningBimonnDense(GenericLightningModel):
    model_class = BimonnDense


class LightningBimonnDenseNotBinary(GenericLightningModel):
    model_class = BimonnDenseNotBinary


class LightningBimonnBiselDenseNotBinary(GenericLightningModel):
    model_class = BimonnBiselDenseNotBinary



# class LightningBiMoNN(NetLightning):

#     def __init__(
#         self,
#         model_args: Dict,
#         learning_rate: float,
#         loss: Callable,
#         optimizer: Callable,
#         optimizer_args: Dict = {},
#         observables: List["Observable"] = [],
#         **kwargs
#     ):
#         super().__init__(
#             model=BiMoNN(**model_args),
#             learning_rate=learning_rate,
#             loss=loss,
#             optimizer=optimizer,
#             optimizer_args=optimizer_args,
#             observables=observables,
#             **kwargs
#         )

#         self.save_hyperparameters(ignore="observables")
#         self.hparams["model_type"] = self.__class__.__name__



# class LightningBiMoNNClassifierMaxPool(NetLightning):

#     def __init__(
#         self,
#         model_args: Dict,
#         learning_rate: float,
#         loss: Callable,
#         optimizer: Callable,
#         optimizer_args: Dict = {},
#         observables: List["Observable"] = [],
#         **kwargs
#     ):
#         super().__init__(
#             model=BiMoNNClassifierMaxPool(**model_args),
#             learning_rate=learning_rate,
#             loss=loss,
#             optimizer=optimizer,
#             optimizer_args=optimizer_args,
#             observables=observables,
#             **kwargs
#         )

#         self.save_hyperparameters(ignore="observables")
#         self.hparams["model_type"] = self.__class__.__name__



# class LightningBiMoNNClassifierMaxPoolNotBinary(NetLightning):

#     def __init__(
#         self,
#         model_args: Dict,
#         learning_rate: float,
#         loss: Callable,
#         optimizer: Callable,
#         optimizer_args: Dict = {},
#         observables: List["Observable"] = [],
#         **kwargs
#     ):
#         super().__init__(
#             model=BiMoNNClassifierMaxPoolNotBinary(**model_args),
#             learning_rate=learning_rate,
#             loss=loss,
#             optimizer=optimizer,
#             optimizer_args=optimizer_args,
#             observables=observables,
#             **kwargs
#         )

#         self.save_hyperparameters(ignore="observables")
#         self.hparams["model_type"] = self.__class__.__name__



# class LightningBiMoNNClassifierLastLinearNotBinary(NetLightning):

#     def __init__(
#         self,
#         model_args: Dict,
#         learning_rate: float,
#         loss: Callable,
#         optimizer: Callable,
#         optimizer_args: Dict = {},
#         observables: List["Observable"] = [],
#         **kwargs
#     ):
#         super().__init__(
#             model=BiMoNNClassifierLastLinearNotBinary(**model_args),
#             learning_rate=learning_rate,
#             loss=loss,
#             optimizer=optimizer,
#             optimizer_args=optimizer_args,
#             observables=observables,
#             **kwargs
#         )

#         self.save_hyperparameters(ignore="observables")
#         self.hparams["model_type"] = self.__class__.__name__



# class LightningBiMoNNClassifierLastLinear(NetLightning):

#     def __init__(
#         self,
#         model_args: Dict,
#         learning_rate: float,
#         loss: Callable,
#         optimizer: Callable,
#         optimizer_args: Dict = {},
#         observables: List["Observable"] = [],
#         **kwargs
#     ):
#         super().__init__(
#             model=BiMoNNClassifierLastLinear(**model_args),
#             learning_rate=learning_rate,
#             loss=loss,
#             optimizer=optimizer,
#             optimizer_args=optimizer_args,
#             observables=observables,
#             **kwargs
#         )

#         self.save_hyperparameters(ignore="observables")
#         self.hparams["model_type"] = self.__class__.__name__


# class LightningBimonnDense(NetLightning):

#     def __init__(
#         self,
#         model_args: Dict,
#         learning_rate: float,
#         loss: Callable,
#         optimizer: Callable,
#         optimizer_args: Dict = {},
#         observables: List["Observable"] = [],
#         **kwargs
#     ):
#         super().__init__(
#             model=BimonnDense(**model_args),
#             learning_rate=learning_rate,
#             loss=loss,
#             optimizer=optimizer,
#             optimizer_args=optimizer_args,
#             observables=observables,
#             **kwargs
#         )

#         self.save_hyperparameters(ignore="observables")
#         self.hparams["model_type"] = self.__class__.__name__


# class LightningBimonnDenseNotBinary(NetLightning):

#     def __init__(
#         self,
#         model_args: Dict,
#         learning_rate: float,
#         loss: Callable,
#         optimizer: Callable,
#         optimizer_args: Dict = {},
#         observables: List["Observable"] = [],
#         **kwargs
#     ):
#         super().__init__(
#             model=BimonnDenseNotBinary(**model_args),
#             learning_rate=learning_rate,
#             loss=loss,
#             optimizer=optimizer,
#             optimizer_args=optimizer_args,
#             observables=observables,
#             **kwargs
#         )

#         self.save_hyperparameters(ignore="observables")
#         self.hparams["model_type"] = self.__class__.__name__


# class LightningBimonnBiselDenseNotBinary(NetLightning):

#     def __init__(
#         self,
#         model_args: Dict,
#         learning_rate: float,
#         loss: Callable,
#         optimizer: Callable,
#         optimizer_args: Dict = {},
#         observables: List["Observable"] = [],
#         **kwargs
#     ):
#         super().__init__(
#             model=BimonnBiselDenseNotBinary(**model_args),
#             learning_rate=learning_rate,
#             loss=loss,
#             optimizer=optimizer,
#             optimizer_args=optimizer_args,
#             observables=observables,
#             **kwargs
#         )

#         self.save_hyperparameters(ignore="observables")
#         self.hparams["model_type"] = self.__class__.__name__
