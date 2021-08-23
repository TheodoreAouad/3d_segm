import torch
import torch.nn as nn

from ..threshold_fn import *


class LogicalNotLayer(nn.Module):

    def __init__(self, threshold_mode='sigmoid', alpha_init: float = 0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha_init]).float())
        self.threshold_fn = getattr(self, threshold_mode + '_threshold')


    def forward(self, x):
        return self.thresholded_alpha * x + (1 - self.thresholded_alpha) * (1 - x)


    @property
    def thresholded_alpha(self):
        return self.threshold_fn(self.alpha)

    def sigmoid_threshold(self, x):
        return sigmoid_threshold(x)

    def arctan_threshold(self, x):
        return arctan_threshold(x)

    def tanh_threshold(self, x):
        return tanh_threshold(x)

    def erf_threshold(self, x):
        return erf_threshold(x)
