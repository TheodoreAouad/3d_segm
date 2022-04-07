import torch
import torch.nn as nn
import torch.nn.functional as F

from ..threshold_fn import *


class ThresholdLayer(nn.Module):

    def __init__(
        self,
        threshold_fn,
        threshold_inverse_fn=None,
        P_: float = 1,
        n_channels: int = 1,
        axis_channels: int = 1,
        threshold_name: str = '',
        bias: float = 0,
        constant_P: bool = False,
        binary_mode: bool = False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.axis_channels = axis_channels
        self.threshold_name = threshold_name
        self.threshold_fn = threshold_fn
        self.threshold_inverse_fn = threshold_inverse_fn
        self.bias = bias
        self.binary_mode = binary_mode

        if isinstance(P_, nn.Parameter):
            self.P_ = P_
        else:
            self.P_ = nn.Parameter(torch.tensor([P_ for _ in range(n_channels)]).float())
        if constant_P:
            self.P_.requires_grad = False

    def forward(self, x, binary_mode=None):
        # print((x + self.bias).shape)
        # print(self.P_.view(*([len(self.P_)] + [1 for _ in range(x.ndim - 1)])).shape)
        # return self.threshold_fn(
        #     (x + self.bias) * self.P_.view(*([1 for _ in range(self.axis_channels)] + [len(self.P_)] + [1 for _ in range(self.axis_channels, x.ndim - 1)]))
        # )
        if binary_mode is None:
            binary_mode = self.binary_mode

        if binary_mode:
            return x > 0

        return self.apply_threshold(x, self.P_, self.bias)

    def apply_threshold(self, x, P_, bias):
        return self.threshold_fn(
            (x + bias) * P_.view(*([1 for _ in range(self.axis_channels)] + [len(P_)] + [1 for _ in range(self.axis_channels, x.ndim - 1)]))
        )

    def forward_inverse(self, y):
        return self.apply_threshold_inverse(y, self.P_, self.bias)

    def apply_threshold_inverse(self, y, P_, bias):
        assert self.threshold_inverse_fn is not None
        return (
            1 / P_.view(*([1 for _ in range(self.axis_channels)] + [len(P_)] + [1 for _ in range(self.axis_channels, y.ndim - 1)])) *
            self.threshold_inverse_fn(y) - bias
        )


class SigmoidLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=sigmoid_threshold, threshold_inverse_fn=sigmoid_threshold_inverse, threshold_name='sigmoid', *args, **kwargs)


class ArctanLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=arctan_threshold, threshold_inverse_fn=arctan_threshold_inverse, threshold_name='arctan', *args, **kwargs)


class TanhLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=tanh_threshold, threshold_inverse_fn=tanh_threshold_inverse, threshold_name='tanh', *args, **kwargs)


class ErfLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=erf_threshold, threshold_inverse_fn=erf_threshold_inverse, threshold_name='erf', *args, **kwargs)


class ClampLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=lambda x: clamp_threshold(x, 0, 1), threshold_name='clamp', *args, **kwargs)


class IdentityLayer(ThresholdLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(threshold_fn=lambda x: x, threshold_inverse_fn=lambda x: x, threshold_name='identity', *args, **kwargs)


class SoftplusThresholdLayer(ThresholdLayer):
    def __init__(self, beta: int = 1, threshold: int = 20, *args, **kwargs):
        super().__init__(threshold_fn=lambda x: F.softplus(x, beta, threshold), threshold_inverse_fn=lambda x: softplus_threshold_inverse(x, beta))
        self.beta = beta
        self.threshold = threshold


dispatcher = {
    'sigmoid': SigmoidLayer, 'arctan': ArctanLayer, 'tanh': TanhLayer, 'erf': ErfLayer, 'clamp': ClampLayer, 'identity': IdentityLayer,
    'softplus': SoftplusThresholdLayer,
}
