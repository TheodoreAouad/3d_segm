from typing import Tuple

import torch
import torch.nn as nn

from general.utils import max_min_norm
from deep_morpho.threshold_fn import arctan_threshold, tanh_threshold, sigmoid_threshold, erf_threshold


class DilationLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        weight_P: float = 1,
        weight_threshold_mode: str = "sigmoid",
        activation_P: float = 10,
        activation_threshold_mode: str = "sigmoid",
        shared_weights: torch.tensor = None,
        shared_weight_P: torch.tensor = None,
        init_bias_value: float = -2,
        *args,
        **kwargs
    ):
        super().__init__()

        self.weight_threshold_mode = weight_threshold_mode.lower()
        self.activation_threshold_mode = activation_threshold_mode.lower()
        self._weight_P = weight_P
        self.activation_P = activation_P
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            # bias=0,
            padding=kernel_size[0]//2,
            *args,
            **kwargs
        )
        with torch.no_grad():
            self.conv.bias.fill_(init_bias_value)

        self.shared_weights = shared_weights
        self.shared_weight_P = shared_weight_P

        # for to_threshold in ['weight', 'activation']:
        #     attr_mode = getattr(self, f'{to_threshold}_threshold_mode')
        #     attr_fn = f'{to_threshold}_threshold_fn'
        #     attr_P = f'{to_threshold}_P'

        #     if attr_mode == "sigmoid":
        #         setattr(self, attr_P, nn.Parameter(torch.tensor([getattr(self, attr_P)]).float()))
        #         setattr(self, attr_fn, getattr(self, f"sigmoid_{to_threshold}"))
        #     elif attr_mode == "max_min":
        #         setattr(self, attr_fn, max_min_norm)

        # if weight_threshold_mode == "sigmoid":
        #     if shared_weight_P is None:
        #         self._weight_P = nn.Parameter(torch.tensor([weight_P]).float())
        #     self.weight_threshold_fn = self.sigmoid_weight
        # elif weight_threshold_mode == "max_min":
        #     self.weight_threshold_fn = max_min_norm

        if self.weight_threshold_mode != "max_min" and shared_weight_P is None:
            self._weight_P = nn.Parameter(torch.tensor([weight_P]).float())
        self.weight_threshold_fn = getattr(self, f'{self.weight_threshold_mode}_weight')

        if self.activation_threshold_mode != "max_min":
            self.activation_P = nn.Parameter(torch.tensor([activation_P]).float())
        self.activation_threshold_fn = getattr(self, f'{self.activation_threshold_mode}_activation')

        # if activation_threshold_mode == "sigmoid":
        #     self.activation_P = nn.Parameter(torch.tensor([activation_P]).float())
        #     self.activation_threshold_fn = self.sigmoid_activation
        # elif activation_threshold_mode == 'max_min':
        #     self.activation_threshold_fn = max_min_norm

    def forward(self, x: torch.Tensor):
        output = self.conv._conv_forward(x, self._normalized_weight, self.bias, )
        output = self.activation_threshold_fn(output)
        return output

    @property
    def _normalized_weight(self):
        conv_weight = self.weight_threshold_fn(self.weight)
        # return conv_weight / conv_weight.sum()
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
    def weights(self):
        return self.weight

    @property
    def bias(self):
        return self.conv.bias

    def sigmoid_weight(self, x):
        return sigmoid_threshold(x, self.weight_P)

    def sigmoid_activation(self, x):
        return sigmoid_threshold(x, self.activation_P)

    def arctan_weight(self, x):
        return arctan_threshold(x, self.weight_P)

    def arctan_activation(self, x):
        return arctan_threshold(x, self.activation_P)

    def weight_max_min(self, x):
        return max_min_norm(x)

    def activation_max_min(self, x):
        return max_min_norm(x)

    def tanh_weight(self, x):
        return tanh_threshold(x, self.weight_P)

    def tanh_activation(self, x):
        return tanh_threshold(x, self.activation_P)

    def erf_weight(self, x):
        return erf_threshold(x, self.weight_P)

    def erf_activation(self, x):
        return erf_threshold(x, self.activation_P)
