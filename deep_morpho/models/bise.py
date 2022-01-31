from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


from .threshold_layer import dispatcher
from .complementation_layer import ComplementationLayer


class BiSE(nn.Module):

    def __init__(
        self,
        kernel_size: Tuple,
        weight_P: float = 1,
        threshold_mode: str = "sigmoid",
        activation_P: float = 10,
        constant_activation_P: bool = False,
        constant_weight_P: bool = False,
        shared_weights: torch.tensor = None,
        shared_weight_P: torch.tensor = None,
        init_bias_value: float = -2,
        init_weight_identity: bool = True,
        out_channels=1,
        *args,
        **kwargs
    ):
        """
        Initialize the layer

        Args:
            self: write your description
            kernel_size: write your description
            weight_P: write your description
            threshold_mode: write your description
            activation_P: write your description
            constant_activation_P: write your description
            constant_weight_P: write your description
            shared_weights: write your description
            torch: write your description
            tensor: write your description
            shared_weight_P: write your description
            torch: write your description
            tensor: write your description
            init_bias_value: write your description
            init_weight_identity: write your description
            out_channels: write your description
        """
        super().__init__()
        self.threshold_mode = self._init_threshold_mode(threshold_mode)
        self._weight_P = nn.Parameter(torch.tensor([weight_P for _ in range(out_channels)]).float())
        self.activation_P_init = activation_P
        self.kernel_size = self._init_kernel_size(kernel_size)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            # padding="same",
            padding=self.kernel_size[0]//2,
            padding_mode='replicate',
            *args,
            **kwargs
        )
        with torch.no_grad():
            self.conv.bias.fill_(init_bias_value)
        if init_weight_identity:
            self._init_as_identity()

        self.shared_weights = shared_weights
        self.shared_weight_P = shared_weight_P

        self.weight_threshold_layer = dispatcher[self.weight_threshold_mode](P_=self.weight_P, constant_P=constant_weight_P, n_channels=out_channels, axis_channels=0)
        self.activation_threshold_layer = dispatcher[self.activation_threshold_mode](P_=activation_P, constant_P=constant_activation_P, n_channels=out_channels, axis_channels=1)

    @staticmethod
    def bise_from_selem(selem: np.ndarray, operation: str, threshold_mode: str = "tanh", weight_P=10, **kwargs):
        """
        Constructs a BiSE net from a given selem.

        Args:
            selem: write your description
            np: write your description
            ndarray: write your description
            operation: write your description
            threshold_mode: write your description
            weight_P: write your description
        """
        net = BiSE(kernel_size=selem.shape, threshold_mode=threshold_mode, **kwargs)
        assert set(np.unique(selem)).issubset([0, 1])
        net.set_weights((torch.tensor(selem) - .5)[None, None, ...])
        net._weight_P.data = torch.FloatTensor([weight_P])
        bias_value = -.5 if operation == "dilation" else -float(selem.sum()) + .5
        net.set_bias(torch.FloatTensor([bias_value]))
        return net

    @staticmethod
    def _init_kernel_size(kernel_size: Union[Tuple, int]):
        """
        Initializes the kernel size.

        Args:
            kernel_size: write your description
        """
        if isinstance(kernel_size, int):
            return (kernel_size, kernel_size)
        return kernel_size

    def activation_threshold_fn(self, x):
        """
        Returns the activation threshold function for the given state.

        Args:
            self: write your description
            x: write your description
        """
        return self.activation_threshold_layer.threshold_fn(x)

    def weight_threshold_fn(self, x):
        """
        Returns the weight threshold function for the given input.

        Args:
            self: write your description
            x: write your description
        """
        return self.weight_threshold_layer.threshold_fn(x)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward computation.

        Args:
            self: write your description
            x: write your description
        """
        output = self.conv._conv_forward(x, self._normalized_weight, self.bias, )
        output = self.activation_threshold_layer(output)
        return output

    def _init_as_identity(self):
        """
        Initialize the convolution weight as identity.

        Args:
            self: write your description
        """
        self.conv.weight.data.fill_(-1)
        shape = self.conv.weight.shape
        self.conv.weight.data[..., shape[-2]//2, shape[-1]//2] = 1

    @staticmethod
    def is_erosion_by(normalized_weights: torch.Tensor, bias: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1):
        """
        Determines if a bias is a dominant oesion by a given bias

        Args:
            normalized_weights: write your description
            torch: write your description
            Tensor: write your description
            bias: write your description
            torch: write your description
            Tensor: write your description
            S: write your description
            np: write your description
            ndarray: write your description
            v1: write your description
            v2: write your description
        """
        assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
        lb, ub = BiSE.bias_bounds_erosion(normalized_weights=normalized_weights, S=S, v1=v1, v2=v2)
        return lb < -bias < ub

    @staticmethod
    def is_dilation_by(normalized_weights: torch.Tensor, bias: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1):
        """
        Determines whether the given dilation is dilation by the given bias

        Args:
            normalized_weights: write your description
            torch: write your description
            Tensor: write your description
            bias: write your description
            torch: write your description
            Tensor: write your description
            S: write your description
            np: write your description
            ndarray: write your description
            v1: write your description
            v2: write your description
        """
        assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
        lb, ub = BiSE.bias_bounds_dilation(normalized_weights=normalized_weights, S=S, v1=v1, v2=v2)
        return lb < -bias < ub

    @staticmethod
    def bias_bounds_erosion(normalized_weights: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1):
        """
        Computes the bias bounds for a given set of normalized weights

        Args:
            normalized_weights: write your description
            torch: write your description
            Tensor: write your description
            S: write your description
            np: write your description
            ndarray: write your description
            v1: write your description
            v2: write your description
        """
        S = S.astype(bool)
        W = normalized_weights.squeeze().cpu().detach().numpy()
        return W.sum() - (1 - v1) * W[S].min(), v2 * W[S].sum()

    @staticmethod
    def bias_bounds_dilation(normalized_weights: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1):
        """
        Computes the bias bounds dilation of normalized weights

        Args:
            normalized_weights: write your description
            torch: write your description
            Tensor: write your description
            S: write your description
            np: write your description
            ndarray: write your description
            v1: write your description
            v2: write your description
        """
        S = S.astype(bool)
        W = normalized_weights.squeeze().cpu().detach().numpy()
        return W[~S].sum() + v1 * W[S].sum(), v2 * W[S].min()

    def find_selem_for_operation_chan(self, idx: int, operation: str, v1: float = 0, v2: float = 1):
        """
        We find the selem for either a dilation or an erosion. In theory, there is at most one selem that works.
        We verify this theory.

        Args:
            operation (str): 'dilation' or 'erosion', the operation we want to check for
            v1 (float): the lower value of the almost binary
            v2 (float): the upper value of the almost binary (input not in ]v1, v2[)

        Returns:
            np.ndarray if a selem is found
            None if none is found
        """
        weights = self._normalized_weight[idx]
        # weight_values = weights.unique()
        bias = self.bias[idx]
        is_op_fn = {'dilation': self.is_dilation_by, 'erosion': self.is_erosion_by}[operation]
        born = {'dilation': -bias / v2, "erosion": (weights.sum() + bias) / (1 - v1)}[operation]

        # possible_values = weight_values >= born
        selem = (weights > born).squeeze().cpu().detach().numpy()
        if not selem.any():
            return None

        # selem = (weights >= weight_values[possible_values][0]).squeeze().cpu().detach().numpy()

        if is_op_fn(weights, bias, selem, v1, v2):
            return selem
        return None


    def find_selem_dilation_chan(self, idx: int, v1: float = 0, v2: float = 1):
        """
        Find the channel of a particular element in the DIlation waveform.

        Args:
            self: write your description
            idx: write your description
            v1: write your description
            v2: write your description
        """
        return self.find_selem_for_operation(idx, 'dilation', v1=v1, v2=v2)


    def find_selem_erosion_chan(self, idx: int, v1: float = 0, v2: float = 1):
        """
        Find the element of the specified monosion channel.

        Args:
            self: write your description
            idx: write your description
            v1: write your description
            v2: write your description
        """
        return self.find_selem_for_operation(idx, 'erosion', v1=v1, v2=v2)

    def find_selem_and_operation_chan(self, idx: int, v1: float = 0, v2: float = 1):
        """Find the selem and the operation given the almost binary features.

        Args:
            v1 (float): lower bound of almost binary input deadzone. Defaults to 0.
            v2 (float): upper bound of almost binary input deadzone. Defaults to 1.

        Returns:
            (np.ndarray, operation): if the selem is found, returns the selem and the operation
            (None, None): if nothing is found, returns None
        """
        for operation in ['dilation', 'erosion']:
            selem = self.find_selem_for_operation_chan(idx, operation, v1=v1, v2=v2)
            if selem is not None:
                return selem, operation
        return None, None

    def get_outputs_bounds(self, v1: float = 0, v2: float = 1):
        """If the BiSE is learned, returns the bounds of the deadzone of the almost binary output.

        Args:
            v1 (float): lower bound of almost binary input deadzone. Defaults to 0.
            v2 (float): upper bound of almost binary input deadzone. Defaults to 1.

        Returns:
            (float, float): if the bise is learned, bounds of the deadzone of the almost binary output
            (None, None): if the bise is not learned
        """
        selem, operation = self.find_selem_and_operation(v1=v1, v2=v2)

        if selem is None:
            return None, None

        if operation == 'dilation':
            b1, b2 = self.bias_bounds_dilation(selem, v1=v1, v2=v2)
        if operation == 'erosion':
            b1, b2 = self.bias_bounds_erosion(selem, v1=v1, v2=v2)

        with torch.no_grad():
            res = [self.activation_threshold_layer(b1 + self.bias), self.activation_threshold_layer(b2 + self.bias)]
        res = [i.item() for i in res]
        return res
        # return 0, 1

    @staticmethod
    def _init_threshold_mode(threshold_mode):
        """
        Initialize threshold_mode.

        Args:
            threshold_mode: write your description
        """
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["weight", "activation"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode

    @property
    def weight_threshold_mode(self):
        """
        Weight threshold mode.

        Args:
            self: write your description
        """
        return self.threshold_mode["weight"]

    @property
    def activation_threshold_mode(self):
        """
        Returns the activation threshold mode.

        Args:
            self: write your description
        """
        return self.threshold_mode["activation"]

    @property
    def _normalized_weight(self):
        """
        Return normalized weight.

        Args:
            self: write your description
        """
        conv_weight = self.weight_threshold_layer(self.weight)
        return conv_weight

    @property
    def _normalized_weights(self):
        """
        Weights for the model.

        Args:
            self: write your description
        """
        return self._normalized_weight

    @property
    def weight(self):
        """
        Weight of the convolution.

        Args:
            self: write your description
        """
        if self.shared_weights is not None:
            return self.shared_weights
        return self.conv.weight

    @property
    def out_channels(self):
        """
        The number of out channels.

        Args:
            self: write your description
        """
        return self.conv.out_channels

    def set_weights(self, new_weights: torch.Tensor) -> torch.Tensor:
        """
        Set the weights of the convolutional layer.

        Args:
            self: write your description
            new_weights: write your description
            torch: write your description
            Tensor: write your description
        """
        assert self.weight.shape == new_weights.shape, f"Weights must be of same shape {self.weight.shape}"
        self.conv.weight.data = new_weights
        return new_weights

    def set_bias(self, new_bias: torch.Tensor) -> torch.Tensor:
        """
        Set the bias tensor to a new value.

        Args:
            self: write your description
            new_bias: write your description
            torch: write your description
            Tensor: write your description
        """
        assert self.bias.shape == new_bias.shape
        self.conv.bias.data = new_bias
        return new_bias

    @property
    def weight_P(self):
        """
        Weight in P.

        Args:
            self: write your description
        """
        if self.shared_weight_P is not None:
            return self.shared_weight_P
        return self._weight_P

    @property
    def activation_P(self):
        """
        The activation threshold P_i.

        Args:
            self: write your description
        """
        return self.activation_threshold_layer.P_

    @property
    def weights(self):
        """
        Weights of the node.

        Args:
            self: write your description
        """
        return self.weight

    @property
    def bias(self):
        """
        Bias of the output.

        Args:
            self: write your description
        """
        return self.conv.bias


class BiSEC(BiSE):

    def __init__(self, *args, alpha_init=0, invert_thresholded_alpha=False, **kwargs):
        """
        Initialize the complementation layer

        Args:
            self: write your description
            alpha_init: write your description
            invert_thresholded_alpha: write your description
        """
        super().__init__(*args, **kwargs)
        self.conv.bias = None
        self.complementation_layer = ComplementationLayer(
            self.complementation_threshold_mode, alpha_init=alpha_init, invert_thresholded_alpha=invert_thresholded_alpha
        )

        self._bias = nn.Parameter(torch.tensor([0.5]).float(), requires_grad=False)


    def forward(self, x: Tensor) -> Tensor:
        """
        Returns the complemented complemented forward layer.

        Args:
            self: write your description
            x: write your description
        """
        output = self.complementation_layer(x)
        output = super().forward(output)
        if self.thresholded_alpha < 1/2:
            return 1 - output
        # output = self.complementation_layer(output)
        return output

    @property
    def bias(self):
        """
        The bias of the layer.

        Args:
            self: write your description
        """
        return -(
            min(self.thresholded_alpha, 1 - self.thresholded_alpha) *
            (self._normalized_weight.sum() - 1) + self._bias
        )

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

    @staticmethod
    def _init_threshold_mode(threshold_mode):
        """
        Initialize threshold_mode.

        Args:
            threshold_mode: write your description
        """
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["weight", "activation", "complementation"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode
