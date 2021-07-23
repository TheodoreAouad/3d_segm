from typing import Tuple

import torch
import torch.nn as nn

from .dilation_layer import DilationLayer


class OpeningNet(nn.Module):

    def __init__(self, share_weights=False, **dilation_layer_args):
        super().__init__()
        self.share_weights = share_weights
        self.dilations1 = DilationLayer(**dilation_layer_args)
        if self.share_weights:
            self.dilations2 = DilationLayer(shared_weights=self.dilations1.weights, **dilation_layer_args)
        else:
            self.dilations2 = DilationLayer(**dilation_layer_args)

        self.dilations = [self.dilations1, self.dilations2]

    def forward(self, x):
        output = self.dilations[0](x)
        output = self.dilations[1](output)
        return output
