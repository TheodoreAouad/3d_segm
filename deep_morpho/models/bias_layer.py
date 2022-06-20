import torch
import torch.nn as nn

from .softplus import Softplus


class BiasBise(nn.Module):
    """Base class to deal with bias in BiSE like neurons. We suppose that the bias = f(param).

    Args:
        nn (_type_): _description_
    """

    def __init__(self, bise_module: "BiSE"):
        super().__init__()
        self.bise_module = bise_module
        self.param = self.init_param()

    def forward(self):
        return self.from_param_to_bias(self.param)

    def forward_inverse(self, bias):
        return self.from_bias_to_param(bias)

    def from_param_to_bias(self, param):
        raise NotImplementedError

    def from_bias_to_param(self, bias):
        raise NotImplementedError

    @property
    def conv(self):
        return self.bise_module.conv

    @property
    def shape(self):
        return (self.conv.weight.shape[0], )

    def init_param(self, *args, **kwargs):
        return nn.Parameter(torch.FloatTensor(size=self.shape, requires_grad=True))

    def set_bias(self, new_bias: torch.Tensor) -> torch.Tensor:
        new_param = self.from_bias_to_param(new_bias)
        self.set_param(new_param)
        return new_param

    def set_param(self, new_param: torch.Tensor) -> torch.Tensor:
        self.param.data = new_param
        return new_param

    @property
    def grad(self):
        return self.param.grad

class BiasRaw(BiasBise):

    def from_param_to_bias(self, param):
        raise param

    def from_bias_to_param(self, bias):
        raise bias


class BiasSoftplus(BiasBise):

    def __init__(self, offset=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = offset
        self.softplus_layer = Softplus()

    def from_param_to_bias(self, param):
        return -self.softplus_layer(param) - self.offset

    def from_bias_to_param(self, bias):
        return self.softplus_layer.forward_inverse(-bias - self.offset)


class BiasBiseSoftplusReparametrized(BiasSoftplus):

    def __init__(self, *args, offset=0, **kwargs):
        super().__init__(offset=offset, *args, **kwargs)

    def get_min_max_intrinsic_bias_values(self):
        """We compute the min and max values that are really useful for the bias. We suppose that if the bias is out of
        these bounds, then it is nonsense.
        The bounds are computed to avoid having always a negative convolution output or a positive convolution output.
        """
        weights_aligned = self.bise_module._normalized_weight.flatten(start_dim=1)
        # weights_aligned = self.bise_module._normalized_weight.reshape(self.bise_module._normalized_weight.shape[0], -1)
        weights_min = weights_aligned.min(1).values
        weights_2nd_min = weights_aligned.kthvalue(2, 1).values
        weights_sum = weights_aligned.sum(1)

        bmin = 1/2 * (weights_min + weights_2nd_min)
        bmax = weights_sum - 1/2 * weights_min

        return bmin, bmax

    def from_param_to_bias(self, param):
        # with torch.no_grad():
        bmin, bmax = self.get_min_max_intrinsic_bias_values()
        return torch.clamp(self.softplus_layer(param), bmin, bmax)
        # return -bias

    def from_bias_to_param(self, bias):
        return bias


class BiasBiseSoftplusProjected(BiasBiseSoftplusReparametrized):

    def from_param_to_bias(self, param):
        with torch.no_grad():
            bias = super().from_param_to_bias(param)
        return bias
