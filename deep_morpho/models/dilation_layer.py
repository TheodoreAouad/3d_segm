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
        P_: float = 1,
        *args,
        **kwargs
    ):
        super().__init__()
        self.P_ = nn.Parameter(torch.tensor([P_]).float())
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=0,
            padding=kernel_size[0]//2,
            *args,
            **kwargs
        )

    def forward(self, x: torch.Tensor):
        # conv_weight = torch.sigmoid(self.P_ * self.conv.weight)
        # conv_weight = conv_weight / conv_weight.sum()
        output = self.conv._conv_forward(x, self._normalized_weight, self.conv.bias, )
        return output

    @property
    def _normalized_weight(self):
        # conv_weight = torch.sigmoid(self.P_ * self.conv.weight)
        conv_weight = max_min_norm(self.conv.weight)
        return conv_weight / conv_weight.sum()

    @property
    def weight(self):
        return self.conv.weight
