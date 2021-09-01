from typing import Tuple

from skimage.morphology import disk
import torch
import torch.nn as nn

from .threshold_layer import dispatcher
from .logical_not_layer import LogicalNotLayer


class BiSE(nn.Module):

    def __init__(
        self,
        kernel_size: Tuple,
        weight_P: float = 1,
        threshold_mode: str = "sigmoid",
        activation_P: float = 10,
        shared_weights: torch.tensor = None,
        shared_weight_P: torch.tensor = None,
        init_bias_value: float = -2,
        init_weight_identity: bool = True,
        out_channels=1,
        *args,
        **kwargs
    ):
        super().__init__()
        self.threshold_mode = self._init_threshold_mode(threshold_mode)
        self._weight_P = nn.Parameter(torch.tensor([weight_P]).float())
        self.activation_P_init = activation_P
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
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

    @staticmethod
    def _init_threshold_mode(threshold_mode):
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["weight", "activation"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode

    @property
    def weight_threshold_mode(self):
        return self.threshold_mode["weight"]

    @property
    def activation_threshold_mode(self):
        return self.threshold_mode["activation"]

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


class LogicalNotBiSE(BiSE):

    def __init__(self, *args, alpha_init=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv.bias = None
        self.logical_not_layer = LogicalNotLayer(self.logical_not_threshold_mode, alpha_init=alpha_init)

        self._bias = nn.Parameter(torch.tensor([-0.5]).float(), requires_grad=False)

        # exp 11 logical not
        # self.activation_threshold_layer.P_.requires_grad = False
        # self.activation_threshold_layer.P_.fill_(10)

    # exp 11 logical not
    # @property
    # def _normalized_weight(self):
    #     return torch.FloatTensor(disk(3)).unsqueeze(0).unsqueeze(0).cuda()

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

    @property
    def logical_not_threshold_mode(self):
        return self.threshold_mode["logical_not"]

    @staticmethod
    def _init_threshold_mode(threshold_mode):
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["weight", "activation", "logical_not"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode