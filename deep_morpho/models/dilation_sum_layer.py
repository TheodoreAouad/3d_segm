from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complementation_layer import ComplementationLayer
from .threshold_layer import dispatcher


class Unfolder(nn.Module):

    def __init__(self, kernel_size, padding=0, device='cpu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (padding, padding, padding, padding) if not isinstance(padding, tuple) else padding
        self._device_tensor = nn.Parameter(torch.FloatTensor(device=device))

        self._right_operator = {}
        self._left_operator = {}


    @property
    def device(self):
        return self._device_tensor.device

    @property
    def ndim(self):
        return len(self.kernel_size)

    def unfold(self, x):
        x = F.pad(x, self.padding)
        return self.get_left_operator(x.shape[-self.ndim:]) @ x @ self.get_right_operator(x.shape[-self.ndim:])

    @staticmethod
    def create_right_operator(size, k, device):
        right_operator = torch.zeros((size[0], k * (size[1] - k+1)), device=device)
        for i in range(right_operator.shape[0] - k + 1):
            right_operator[i:i+k, k*i:k*(i+1)] = torch.eye(k)
        return right_operator

    def get_right_operator(self, size):
        if size not in self._right_operator.keys():
            self._right_operator[size] = self.create_right_operator(size, self.kernel_size[1], device=self.device)
            setattr(self, '_right_operator_' + self.add_size_string(size), self._right_operator[size])
        return self._right_operator[size]

    def get_left_operator(self, size):
        if size not in self._left_operator.keys():
            self._left_operator[size] = self.create_right_operator(size[::-1], self.kernel_size[0], device=self.device).T
            setattr(self, '_left_operator_' + self.add_size_string(size), self._left_operator[size])
        return self._left_operator[size]

    def __call__(self, x):
        return self.unfold(x)

    @staticmethod
    def add_size_string(size, sep="x"):
        return sep.join([f'{s}' for s in size])


class DilationSumLayer(nn.Module):

    def __init__(
        self,
        kernel_size,
        activation_P: float = 10,
        threshold_mode: str = 'tanh',
        padding: Union[int, str] = 'same',
        init_value: float = -10,
    ):
        super().__init__()
        self.threshold_mode = self._init_threshold_mode(threshold_mode)
        self.activation_P_init = activation_P
        self.kernel_size = kernel_size
        self.init_value = init_value
        if padding == 'same':
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = tuple(sum([[k//2, k // 2] for k in kernel_size], start=[]))
        self.padding = tuple([padding for _ in range(self.ndim * 2)]) if not isinstance(padding, tuple) else padding

        self.unfolder = Unfolder(kernel_size=kernel_size, padding=padding)
        self.weight = nn.Parameter(self.init_weights(init_value)).float()
        self.maxpool = nn.MaxPool2d(kernel_size)

        self.activation_threshold_layer = dispatcher[self.activation_threshold_mode](P_=activation_P)

    def forward(self, x):
        output = self.unfolder(x)
        output = output + self.kron_weight(x.shape[-self.ndim:])
        output = self.maxpool(output)
        return self.activation_threshold_layer(output)

    def init_weights(self, init_value):
        weights = torch.zeros(self.kernel_size) + init_value
        weights[self.kernel_size[0]//2, self.kernel_size[1]//2] = 0
        return weights

    @property
    def activation_threshold_mode(self):
        return self.threshold_mode["activation"]

    @property
    def activation_P(self):
        return self.activation_threshold_layer.P_

    def kron_weight(self, size):
        return torch.kron(torch.ones([self.padding[2*k] + size[k] + self.padding[2*k + 1] - self.kernel_size[k] + 1 for k in range(self.ndim)], device=self.weight.device), self.weight)

    @property
    def ndim(self):
        return len(self.kernel_size)

    @staticmethod
    def _init_threshold_mode(threshold_mode):
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["activation"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode


class MaxPlusAtom(nn.Module):

    def __init__(self, *args, alpha_init=0, threshold_mode='tanh', **kwargs):
        super().__init__()
        self.threshold_mode = self._init_threshold_mode(threshold_mode)
        self.dilation_sum_layer = DilationSumLayer(threshold_mode=threshold_mode, *args, **kwargs)
        self.complementation_layer = ComplementationLayer(self.complementation_threshold_mode, alpha_init=alpha_init)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.complementation_layer(x)
        # output = super().forward(output)
        output = self.dilation_sum_layer.forward(output)
        if self.thresholded_alpha < 1/2:
            return 1 - output
        # output = self.complementation_layer(output)
        return output

    @property
    def alpha(self):
        return self.complementation_layer.alpha

    @property
    def thresholded_alpha(self):
        return self.complementation_layer.thresholded_alpha

    @property
    def complementation_threshold_mode(self):
        return self.threshold_mode["complementation"]

    @property
    def activation_threshold_mode(self):
        return self.dilation_sum_layer.activation_threshold_mode

    @staticmethod
    def _init_threshold_mode(threshold_mode):
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["complementation"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode

    def kron_weight(self, size):
        return self.dilation_sum_layer.kron_weight(size)

    @property
    def weight(self):
        return self.dilation_sum_layer.weight

    @property
    def weights(self):
        return self.weight

    @property
    def ndim(self):
        return self.dilation_sum_layer.ndim
