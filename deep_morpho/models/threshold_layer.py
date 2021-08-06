import torch
import torch.nn as nn

from ..threshold_fn import *


class ThresholdLayer(nn.Module):

    def __init__(self, threshold_fn, P_: float = 1, threshold_name: str = ''):
        super().__init__()
        if isinstance(P_, nn.Parameter):
            self.P_ = P_
        else:
            self.P_ = nn.Parameter(torch.tensor([P_]).float())
        self.threshold_name = threshold_name
        self.threshold_fn = threshold_fn

    def forward(self, x):
        return self.threshold_fn(x * self.P_)


class SigmoidLayer(ThresholdLayer):
    def __init__(self, P_: float = 1):
        super().__init__(threshold_fn=sigmoid_threshold,  P_=P_, threshold_name='sigmoid')


class ArctanLayer(ThresholdLayer):
    def __init__(self, P_: float = 1):
        super().__init__(threshold_fn=arctan_threshold,  P_=P_, threshold_name='arctan')


class TanhLayer(ThresholdLayer):
    def __init__(self, P_: float = 1):
        super().__init__(threshold_fn=tanh_threshold,  P_=P_, threshold_name='tanh')


class ErfLayer(ThresholdLayer):
    def __init__(self, P_: float = 1):
        super().__init__(threshold_fn=erf_threshold,  P_=P_, threshold_name='erf')


dispatcher = {
    'sigmoid': SigmoidLayer, 'arctan': ArctanLayer, 'tanh': TanhLayer, 'erf': ErfLayer
}

