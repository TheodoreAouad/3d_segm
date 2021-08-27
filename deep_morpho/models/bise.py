from typing import Tuple

from skimage.morphology import disk
import torch
import torch.nn as nn

from general.utils import max_min_norm
from deep_morpho.threshold_fn import arctan_threshold, tanh_threshold, sigmoid_threshold, erf_threshold
from .threshold_layer import dispatcher
from .logical_not_layer import LogicalNotLayer


class BiSE(nn.Module):

    def __init__(
        self,
        kernel_size: Tuple,
        weight_P: float = 1,
        weight_threshold_mode: str = "sigmoid",
        activation_P: float = 10,
        activation_threshold_mode: str = "sigmoid",
        shared_weights: torch.tensor = None,
        shared_weight_P: torch.tensor = None,
        init_bias_value: float = -2,
        init_weight_identity: bool = True,
        out_channels=1,
        *args,
        **kwargs
    ):
        super().__init__()

        self.weight_threshold_mode = weight_threshold_mode.lower()
        self.activation_threshold_mode = activation_threshold_mode.lower()
        self._weight_P = nn.Parameter(torch.tensor([weight_P]).float())
        self.activation_P_init = activation_P
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            # bias=0,
            padding=kernel_size[0]//2,
            *args,
            **kwargs
        )
        with torch.no_grad():
            self.conv.bias.fill_(init_bias_value)
        if init_weight_identity:
            self._init_as_identity()

        self.shared_weights = shared_weights
        self.shared_weight_P = shared_weight_P

        self.weight_threshold_layer = dispatcher[self.weight_threshold_mode](P_=self.weight_P)
        self.activation_threshold_layer = dispatcher[self.activation_threshold_mode](P_=activation_P)

    def activation_threshold_fn(self, x):
        return self.activation_threshold_layer.threshold_fn(x)

    def weight_threshold_fn(self, x):
        return self.weight_threshold_layer.threshold_fn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv._conv_forward(x, self._normalized_weight, self.bias, )
        output = self.activation_threshold_layer(output)
        return output

    def _init_as_identity(self):
        self.conv.weight.data.fill_(-1)
        shape = self.conv.weight.shape
        self.conv.weight.data[..., shape[-2]//2, shape[-1]//2] = 1

    @property
    def _normalized_weight(self):
        conv_weight = self.weight_threshold_layer(self.weight)
        return conv_weight

    @property
    def weight(self):
        if self.shared_weights is not None:
            return self.shared_weights
        return self.conv.weight

    @property
    def weight_P(self):
        if self.shared_weight_P is not None:
            return self.shared_weight_P
        return self._weight_P

    @property
    def activation_P(self):
        return self.activation_threshold_layer.P_

    @property
    def weights(self):
        return self.weight

    @property
    def bias(self):
        return self.conv.bias

    # def sigmoid_weight(self, x):
    #     return sigmoid_threshold(x, self.weight_P)
    #
    # def sigmoid_activation(self, x):
    #     return sigmoid_threshold(x, self.activation_P)
    #
    # def arctan_weight(self, x):
    #     return arctan_threshold(x, self.weight_P)
    #
    # def arctan_activation(self, x):
    #     return arctan_threshold(x, self.activation_P)
    #
    # def weight_max_min(self, x):
    #     return max_min_norm(x)
    #
    # def activation_max_min(self, x):
    #     return max_min_norm(x)
    #
    # def tanh_weight(self, x):
    #     return tanh_threshold(x, self.weight_P)
    #
    # def tanh_activation(self, x):
    #     return tanh_threshold(x, self.activation_P)
    #
    # def erf_weight(self, x):
    #     return erf_threshold(x, self.weight_P)
    #
    # def erf_activation(self, x):
    #     return erf_threshold(x, self.activation_P)


class LogicalNotBiSE(BiSE):

    def __init__(self, *args, logical_not_threshold_mode: str = 'sigmoid', alpha_init=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv.bias = None
        self.logical_not_threshold_mode = logical_not_threshold_mode
        self.logical_not_layer = LogicalNotLayer(logical_not_threshold_mode, alpha_init=alpha_init)

        self._bias = nn.Parameter(torch.tensor([-0.5]).float(), requires_grad=False)

        # test: we only learn the alpha parameter. We fix the selem
        # self.__normalized_weight = torch.zeros((1, 1, 5, 5)).float()
        # self.__normalized_weight[0, 0, ...] = torch.tensor(disk(2)).float()
        # self.__normalized_weight = nn.Parameter(self.__normalized_weight, requires_grad=False)
        # self.activation_threshold_layer.P_.requires_grad = False


    # @property  # test: we only learn the alpha parameter. We fix the selem
    # def _normalized_weight(self):
    #     return self.__normalized_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.logical_not_layer(x)
        output = super().forward(output)
        output = self.logical_not_layer(output)
        return output

    @property
    def bias(self):
        return self._bias

    @property
    def alpha(self):
        return self.logical_not_layer.alpha

    @property
    def thresholded_alpha(self):
        return self.logical_not_layer.thresholded_alpha
