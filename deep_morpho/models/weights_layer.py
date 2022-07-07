from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .bise import BiSE

import torch
import torch.nn as nn

from .bise_module_container import BiseModuleContainer
from .threshold_layer import dispatcher


class WeightsBise(nn.Module):
    """Base class to deal with weights in BiSE like neurons. We suppose that the weights = f(param).
    """

    def __init__(self, bise_module: "BiSE", *args, **kwargs):
        super().__init__()
        self._bise_module: BiseModuleContainer = BiseModuleContainer(bise_module)
        self.param = self.init_param(*args, **kwargs)

    def forward(self,) -> torch.Tensor:
        return self.from_param_to_weights(self.param)

    def forward_inverse(self, weights: torch.Tensor,) -> torch.Tensor:
        return self.from_weights_to_param(weights)

    def from_param_to_weights(self, param: torch.Tensor,) -> torch.Tensor:
        raise NotImplementedError

    def from_weights_to_param(self, weights: torch.Tensor,) -> torch.Tensor:
        raise NotImplementedError

    @property
    def bise_module(self):
        return self._bise_module.bise_module

    @property
    def conv(self):
        return self.bise_module.conv

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def shape(self):
        return self.conv.weight.shape

    def init_param(self, *args, **kwargs) -> torch.Tensor:
        return nn.Parameter(torch.FloatTensor(size=self.shape))

    def set_weights(self, new_weights: torch.Tensor) -> torch.Tensor:
        new_param = self.from_weights_to_param(new_weights)
        self.set_param(new_param)
        return new_param

    def set_param(self, new_param: torch.Tensor) -> torch.Tensor:
        self.param.data = new_param
        return new_param

    @property
    def grad(self):
        return self.param.grad


class WeightsThresholdedBise(WeightsBise):

    def __init__(self, threshold_mode: str, P_=1, constant_P: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold_mode = threshold_mode
        self.threshold_layer = dispatcher[threshold_mode](P_=P_, constant_P=constant_P, axis_channels=0, n_channels=self.out_channels,)

    def from_param_to_weights(self, param: torch.Tensor) -> torch.Tensor:
        return self.threshold_layer.forward(param)

    def from_weights_to_param(self, weights: torch.Tensor) -> torch.Tensor:
        return self.threshold_layer.forward_inverse(weights)


class WeightsNormalizedBiSE(WeightsThresholdedBise):

    def __init__(self, factor: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor

    def from_param_to_weights(self, param: torch.Tensor) -> torch.Tensor:
        normalized_weights = super().from_param_to_weights(param)
        return self.factor * normalized_weights / normalized_weights.sum()


class WeightsRaw(WeightsThresholdedBise):

    def __init__(self, *args, **kwargs):
        super().__init__(threshold_mode='identity', *args, **kwargs)
