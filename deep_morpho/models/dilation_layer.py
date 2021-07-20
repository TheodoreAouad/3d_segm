from typing import Tuple

import torch
import torch.nn as nn

from general.utils import max_min_norm


class DilationLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        weight_P: float = 1,
        weight_threshold_mode: str = "sigmoid",
        activation_P: float = 1,
        activation_threshold_mode: str = "sigmoid",
        *args,
        **kwargs
    ):
        super().__init__()

        self.weight_threshold_mode = weight_threshold_mode
        self.activation_threshold_mode = activation_threshold_mode
        self.weight_P = weight_P
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

        for to_threshold in ['weight', 'activation']:
            attr_mode = getattr(self, f'{to_threshold}_threshold_mode')
            attr_fn = f'{to_threshold}_threshold_fn'
            attr_P = f'{to_threshold}_P'

            if attr_mode == "sigmoid":
                setattr(self, attr_P, nn.Parameter(torch.tensor([getattr(self, attr_P)]).float()))
                setattr(self, attr_fn, lambda x: torch.sigmoid(getattr(self, attr_P) * x))
            elif attr_mode == "max_min":
                setattr(self, attr_fn, max_min_norm)
        # if weight_threshold_mode == "sigmoid":
        #     self.weight_P = nn.Parameter(torch.tensor([weight_P]).float())
        #     self.weight_threshold_fn = lambda x: torch.sigmoid(self.weight_P * x)
        # elif weight_threshold_mode == "max_min":
        #     self.weight_threshold_fn = max_min_norm

    def forward(self, x: torch.Tensor):
        output = self.conv._conv_forward(x, self._normalized_weight, self.conv.bias, )
        output = self.activation_threshold_fn(output)
        return output

    @property
    def _normalized_weight(self):
        conv_weight = self.weight_threshold_fn(self.conv.weight)
        # return conv_weight / conv_weight.sum()
        return conv_weight

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias
