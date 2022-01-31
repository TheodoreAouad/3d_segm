from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complementation_layer import ComplementationLayer
from .threshold_layer import dispatcher


class Unfolder(nn.Module):

    def __init__(self, kernel_size, padding=0, device='cpu'):
        """
        Initialize the operator.

        Args:
            self: write your description
            kernel_size: write your description
            padding: write your description
            device: write your description
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (padding, padding, padding, padding) if not isinstance(padding, tuple) else padding
        self._device_tensor = nn.Parameter(torch.FloatTensor(device=device))

        self._right_operator = {}
        self._left_operator = {}


    @property
    def device(self):
        """
        Device of the device tensor.

        Args:
            self: write your description
        """
        return self._device_tensor.device

    @property
    def ndim(self):
        """
        Returns the number of dimensions of the kernel

        Args:
            self: write your description
        """
        return len(self.kernel_size)

    def unfold(self, x):
        """
        Unfold a tensor.

        Args:
            self: write your description
            x: write your description
        """
        x = F.pad(x, self.padding)
        return self.get_left_operator(x.shape[-self.ndim:]) @ x @ self.get_right_operator(x.shape[-self.ndim:])

    @staticmethod
    def create_right_operator(size, k, device):
        """
        Create a right operator for the given size and k

        Args:
            size: write your description
            k: write your description
            device: write your description
        """
        right_operator = torch.zeros((size[0], k * (size[1] - k+1)), device=device)
        for i in range(right_operator.shape[0] - k + 1):
            right_operator[i:i+k, k*i:k*(i+1)] = torch.eye(k)
        return right_operator

    def get_right_operator(self, size):
        """
        Returns the right operator for the specified size.

        Args:
            self: write your description
            size: write your description
        """
        if size not in self._right_operator.keys():
            self._right_operator[size] = self.create_right_operator(size, self.kernel_size[1], device=self.device)
            setattr(self, '_right_operator_' + self.add_size_string(size), self._right_operator[size])
        return self._right_operator[size]

    def get_left_operator(self, size):
        """
        Returns the left operator for the given size.

        Args:
            self: write your description
            size: write your description
        """
        if size not in self._left_operator.keys():
            self._left_operator[size] = self.create_right_operator(size[::-1], self.kernel_size[0], device=self.device).T
            setattr(self, '_left_operator_' + self.add_size_string(size), self._left_operator[size])
        return self._left_operator[size]

    def __call__(self, x):
        """
        Call unfold and return the result.

        Args:
            self: write your description
            x: write your description
        """
        return self.unfold(x)

    @staticmethod
    def add_size_string(size, sep="x"):
        """
        Return size string.

        Args:
            size: write your description
            sep: write your description
        """
        return sep.join([f'{s}' for s in size])


class DilationSumLayer(nn.Module):

    def __init__(
        self,
        kernel_size,
        activation_P: float = 10,
        threshold_mode: str = 'tanh',
        padding: Union[int, str] = 'same',
        init_value: float = -2,
    ):
        """
        Initialize the layer.

        Args:
            self: write your description
            kernel_size: write your description
            activation_P: write your description
            threshold_mode: write your description
            padding: write your description
            init_value: write your description
        """
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
        """
        Apply the unfolder and kronecker product to x.

        Args:
            self: write your description
            x: write your description
        """
        output = self.unfolder(x)
        output = output + self.kron_weight(x.shape[-self.ndim:])
        output = self.maxpool(output)
        # return self.activation_threshold_layer(output)
        return output

    def init_weights(self, init_value):
        """
        Initialize the weights for the kernel

        Args:
            self: write your description
            init_value: write your description
        """
        # weights = torch.zeros(self.kernel_size) + init_value
        # weights[self.kernel_size[0]//2, self.kernel_size[1]//2] = 0
        weights = torch.randn(self.kernel_size)
        return weights

    @property
    def activation_threshold_mode(self):
        """
        Returns the activation threshold mode.

        Args:
            self: write your description
        """
        return self.threshold_mode["activation"]

    @property
    def activation_P(self):
        """
        The activation threshold P_i.

        Args:
            self: write your description
        """
        return self.activation_threshold_layer.P_

    def kron_weight(self, size):
        """
        Return kronecker product of padding and kernel weight

        Args:
            self: write your description
            size: write your description
        """
        return torch.kron(torch.ones([self.padding[2*k] + size[k] + self.padding[2*k + 1] - self.kernel_size[k] + 1 for k in range(self.ndim)], device=self.weight.device), self.weight)

    @property
    def ndim(self):
        """
        Returns the number of dimensions of the kernel

        Args:
            self: write your description
        """
        return len(self.kernel_size)

    @staticmethod
    def _init_threshold_mode(threshold_mode):
        """
        Initialize threshold_mode.

        Args:
            threshold_mode: write your description
        """
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["activation"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode


class MaxPlusAtom(nn.Module):

    def __init__(self, *args, alpha_init=0, threshold_mode='tanh', **kwargs):
        """
        Initializes the layer for the DilationSum algorithm

        Args:
            self: write your description
            alpha_init: write your description
            threshold_mode: write your description
        """
        super().__init__()
        self.threshold_mode = self._init_threshold_mode(threshold_mode)
        self.dilation_sum_layer = DilationSumLayer(threshold_mode=threshold_mode, *args, **kwargs)
        self.complementation_layer = ComplementationLayer(self.complementation_threshold_mode, alpha_init=alpha_init)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complementary sum layer

        Args:
            self: write your description
            x: write your description
            torch: write your description
            Tensor: write your description
        """
        output = self.complementation_layer(x)
        # output = super().forward(output)
        output = self.dilation_sum_layer.forward(output)
        if self.thresholded_alpha < 1/2:
            return 1 - output
        # output = self.complementation_layer(output)
        return output

    @property
    def alpha(self):
        """
        The alpha of the complementation layer.

        Args:
            self: write your description
        """
        return self.complementation_layer.alpha

    @property
    def thresholded_alpha(self):
        """
        The thresholded alpha of the complementation layer.

        Args:
            self: write your description
        """
        return self.complementation_layer.thresholded_alpha

    @property
    def complementation_threshold_mode(self):
        """
        Returns the threshold_mode of the complementation algorithm.

        Args:
            self: write your description
        """
        return self.threshold_mode["complementation"]

    @property
    def activation_threshold_mode(self):
        """
        The activation threshold mode of the layer.

        Args:
            self: write your description
        """
        return self.dilation_sum_layer.activation_threshold_mode

    @staticmethod
    def _init_threshold_mode(threshold_mode):
        """
        Initialize threshold_mode.

        Args:
            threshold_mode: write your description
        """
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["complementation"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode

    def kron_weight(self, size):
        """
        Returns the kron weight of the layer.

        Args:
            self: write your description
            size: write your description
        """
        return self.dilation_sum_layer.kron_weight(size)

    @property
    def weight(self):
        """
        The weight of the layer.

        Args:
            self: write your description
        """
        return self.dilation_sum_layer.weight

    @property
    def weights(self):
        """
        Weights of the node.

        Args:
            self: write your description
        """
        return self.weight

    @property
    def ndim(self):
        """
        Dimension of the dilation sum layer.

        Args:
            self: write your description
        """
        return self.dilation_sum_layer.ndim
