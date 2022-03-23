from typing import Tuple, Union, Dict
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


from .threshold_layer import dispatcher
from .complementation_layer import ComplementationLayer
from .softplus import Softplus
from .binary_nn import BinaryNN
from general.utils import set_borders_to


class BiSE(BinaryNN):

    operation_code = {"erosion": 0, "dilation": 1}

    def __init__(
        self,
        kernel_size: Tuple,
        weight_P: float = 1,
        threshold_mode: Union[Dict[str, str], str] = "sigmoid",
        activation_P: float = 10,
        constant_activation_P: bool = False,
        constant_weight_P: bool = False,
        shared_weights: torch.tensor = None,
        shared_weight_P: torch.tensor = None,
        init_bias_value: float = -2,
        init_weight_mode: str = "normal_identity",
        out_channels: int = 1,
        do_mask_output: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.threshold_mode = self._init_threshold_mode(threshold_mode)
        self._weight_P = nn.Parameter(torch.tensor([weight_P for _ in range(out_channels)]).float())
        self.activation_P_init = activation_P
        self.kernel_size = self._init_kernel_size(kernel_size)
        self.init_weight_mode = init_weight_mode
        self.do_mask_output = do_mask_output
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

        self.softplus_layer = Softplus()

        self.shared_weights = shared_weights
        self.shared_weight_P = shared_weight_P

        self.weight_threshold_layer = dispatcher[self.weight_threshold_mode](P_=self.weight_P, constant_P=constant_weight_P, n_channels=out_channels, axis_channels=0)
        self.activation_threshold_layer = dispatcher[self.activation_threshold_mode](P_=activation_P, constant_P=constant_activation_P, n_channels=out_channels, axis_channels=1)

        with torch.no_grad():
            self.conv.bias.fill_(init_bias_value)

        self.init_weights()


        self.closest_selem = np.zeros((*self.kernel_size, out_channels)).astype(bool)
        self.closest_operation = np.zeros(out_channels)
        self.closest_selem_dist = np.zeros(out_channels)

        self.learned_selem = np.zeros((*self.kernel_size, out_channels)).astype(bool)
        self.learned_operation = np.zeros(out_channels)
        self.is_activated = np.zeros(out_channels).astype(bool)

        self.update_learned_selems()

    def update_learned_selems(self):
        for chan in range(self.out_channels):
            self.find_closest_selem_and_operation_chan(chan)
            self.find_selem_and_operation_chan(chan)

    def binary(self, mode: bool = True):
        if mode:
            self.update_learned_selems()
        return super().binary(mode)

    @staticmethod
    def bise_from_selem(selem: np.ndarray, operation: str, threshold_mode: str = "tanh", weight_P=10, **kwargs):
        assert set(np.unique(selem)).issubset([0, 1])
        net = BiSE(kernel_size=selem.shape, threshold_mode=threshold_mode, out_channels=1, **kwargs)

        if threshold_mode == "identity":
            net.set_weights(torch.FloatTensor(selem)[None, None, ...])
        else:
            net.set_weights((torch.tensor(selem) - .5)[None, None, ...])
            net._weight_P.data = torch.FloatTensor([weight_P])
        bias_value = -.5 if operation == "dilation" else -float(selem.sum()) + .5
        net.set_bias(torch.FloatTensor([bias_value]))

        net.update_learned_selems()

        return net

    @staticmethod
    def _init_kernel_size(kernel_size: Union[Tuple, int]):
        if isinstance(kernel_size, int):
            return (kernel_size, kernel_size)
        return kernel_size

    def init_weights(self):
        if self.init_weight_mode == "normal_identity":
            self.set_weights(self._init_normal_identity(self.kernel_size, self.out_channels))
        elif self.init_weight_mode == "identity":
            self._init_as_identity()
        elif self.init_weight_mode == "conv_0.5":
            self.set_weights(self.weight + 0.5)
        else:
            warnings.warn(f"init weight mode {self.init_weight_mode} not recognized. Classical conv init used.")
            pass

    @staticmethod
    def _init_normal_identity(kernel_size, chan_output, std=0.3, mean=1):
        weights = torch.randn((chan_output,) + kernel_size)[:, None, ...] * std - mean
        weights[..., kernel_size[0] // 2, kernel_size[1] // 2] += 2*mean
        return weights

    def _init_as_identity(self):
        self.conv.weight.data.fill_(-1)
        shape = self.conv.weight.shape
        self.conv.weight.data[..., shape[-2]//2, shape[-1]//2] = 1

    def activation_threshold_fn(self, x):
        return self.activation_threshold_layer.threshold_fn(x)

    def weight_threshold_fn(self, x):
        return self.weight_threshold_layer.threshold_fn(x)

    def forward(self, x: Tensor) -> Tensor:
        if self.binary_mode:
            return self.forward_binary(x)

        output = self.conv._conv_forward(x, self._normalized_weight, self.bias, )
        output = self.activation_threshold_layer(output)

        if self.do_mask_output:
            return self.mask_output(output)

        return output

    def forward_binary(self, x: Tensor) -> Tensor:
        """
        Replaces the BiSE with the closest learned operation.
        """
        weights, bias = self.get_binary_weights_and_bias()
        output = (self.activation_P[None, :, None, None] * self.conv._conv_forward(x, weights, bias, ) > 0).float()
        # output = (self.conv._conv_forward(x, weights, bias, ) > 0).float()

        if self.do_mask_output:
            return self.mask_output(output)

        return output

    def get_binary_weights_and_bias(self) -> Tuple[Tensor, Tensor]:
        """
        Get the closest learned selems as well as the bias corresponding to the operation (dilation or erosion).
        """
        weights = torch.zeros_like(self.weight)
        bias = torch.zeros_like(self.bias)

        weights[self.is_activated, 0] = torch.FloatTensor(self.learned_selem[..., self.is_activated].transpose(2, 0, 1)).to(weights.device)
        weights[~self.is_activated, 0] = torch.FloatTensor(self.closest_selem[..., ~self.is_activated].transpose(2, 0, 1)).to(bias.device)

        dil_key = self.operation_code['dilation']
        ero_key = self.operation_code['erosion']
        bias[self.closest_operation == dil_key] = -0.5
        bias[self.closest_operation == ero_key] = -weights[self.closest_operation == ero_key].sum((1, 2, 3)) + 0.5

        return weights, bias

    def mask_output(self, output):
        masker = set_borders_to(torch.ones(output.shape[-2:], requires_grad=False), border=np.array(self.kernel_size) // 2)[None, None, ...]
        return output * masker.to(output.device)

    @staticmethod
    def distance_to_bounds(bound_fn, normalized_weights: torch.Tensor, bias: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1) -> float:
        assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
        lb, ub = bound_fn(normalized_weights=normalized_weights, S=S, v1=v1, v2=v2)
        dist_lb = lb + bias  # if dist_lb < 0 : lower bound respected
        dist_ub = -bias - ub  # if dist_ub < 0 : upper bound respected
        return max(dist_lb, dist_ub, 0)

    @staticmethod
    def distance_to_dilation(normalized_weights: torch.Tensor, bias: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1) -> float:
        return BiSE.distance_to_bounds(BiSE.bias_bounds_dilation, normalized_weights, bias, S, v1, v2)

    @staticmethod
    def distance_to_erosion(normalized_weights: torch.Tensor, bias: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1) -> float:
        return BiSE.distance_to_bounds(BiSE.bias_bounds_erosion, normalized_weights, bias, S, v1, v2)

    @staticmethod
    def is_erosion_by(normalized_weights: torch.Tensor, bias: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1) -> bool:
        assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
        lb, ub = BiSE.bias_bounds_erosion(normalized_weights=normalized_weights, S=S, v1=v1, v2=v2)
        return lb < -bias < ub

    @staticmethod
    def is_dilation_by(normalized_weights: torch.Tensor, bias: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1) -> bool:
        assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
        lb, ub = BiSE.bias_bounds_dilation(normalized_weights=normalized_weights, S=S, v1=v1, v2=v2)
        return lb < -bias < ub

    @staticmethod
    def bias_bounds_erosion(normalized_weights: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1) -> Tuple[float, float]:
        S = S.astype(bool)
        W = normalized_weights.cpu().detach().numpy()
        return W[W > 0].sum() - (1 - v1) * W[S].min(), v2 * W[S & (W > 0)].sum() + W[W < 0].sum()

    @staticmethod
    def bias_bounds_dilation(normalized_weights: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1) -> Tuple[float, float]:
        S = S.astype(bool)
        W = normalized_weights.cpu().detach().numpy()
        return W[(~S) & (W > 0)].sum() + v1 * W[S & (W > 0)].sum(), v2 * W[S].min() + W[W < 0].sum()

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
        weights = self._normalized_weight[idx, 0]
        # weight_values = weights.unique()
        bias = self.bias[idx]
        is_op_fn = {'dilation': self.is_dilation_by, 'erosion': self.is_erosion_by}[operation]
        born = {'dilation': -bias / v2, "erosion": (weights.sum() + bias) / (1 - v1)}[operation]

        # possible_values = weight_values >= born
        selem = (weights > born).cpu().detach().numpy()
        if not selem.any():
            return None

        # selem = (weights >= weight_values[possible_values][0]).squeeze().cpu().detach().numpy()

        if is_op_fn(weights, bias, selem, v1, v2):
            return selem
        return None

    def find_closest_selem_for_operation_chan(self, idx: int, operation: str, v1: float = 0, v2: float = 1):
        """
        We find the selem for either a dilation or an erosion. In theory, there is at most one selem that works.

        Args:
            operation (str): 'dilation' or 'erosion', the operation we want to check for
            v1 (float): the lower value of the almost binary
            v2 (float): the upper value of the almost binary (input not in ]v1, v2[)

        Returns:
            np.ndarray: the closest selem
            float: the distance to the constraint space
        """
        weights = self._normalized_weight[idx, 0]
        weight_values = weights.unique().detach().cpu().numpy()
        bias = self.bias[idx]
        distance_fn = {'dilation': self.distance_to_dilation, 'erosion': self.distance_to_erosion}[operation]

        dists = np.zeros_like(weight_values)
        selems = []
        for value_idx, value in enumerate(weight_values):
            selem = (weights >= value).cpu().detach().numpy()
            dists[value_idx] = distance_fn(weights, bias, selem, v1, v2)
            selems.append(selem)

        idx_min = dists.argmin()
        return selems[idx_min], dists[idx_min]

    def find_closest_selem_and_operation_chan(self, idx: int, v1: float = 0, v2: float = 1) -> Tuple[np.ndarray, str, float]:
        """Find the closest selem and the operation given the almost binary features.

        Args:
            v1 (float): lower bound of almost binary input deadzone. Defaults to 0.
            v2 (float): upper bound of almost binary input deadzone. Defaults to 1.

        Returns:
            (np.ndarray, str, float): if the selem is found, returns the selem and the operation
        """
        final_dist = np.infty
        for operation in ['dilation', 'erosion']:
            new_selem, new_dist = self.find_closest_selem_for_operation_chan(idx, operation, v1=v1, v2=v2)
            if new_dist < final_dist:
                final_dist = new_dist
                final_selem = new_selem
                final_operation = operation  # str array has 1 character

        self.closest_selem[..., idx] = final_selem.astype(bool)
        self.closest_operation[idx] = self.operation_code[final_operation]
        self.closest_selem_dist[idx] = final_dist
        return final_selem, final_operation, final_dist


    def find_selem_dilation_chan(self, idx: int, v1: float = 0, v2: float = 1):
        return self.find_selem_for_operation(idx, 'dilation', v1=v1, v2=v2)


    def find_selem_erosion_chan(self, idx: int, v1: float = 0, v2: float = 1):
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
                self.learned_selem[..., idx] = selem.astype(bool)
                self.learned_operation[idx] = self.operation_code[operation]  # str array has 1 character
                self.is_activated[idx] = True
                return selem, operation

        self.is_activated[idx] = False
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
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["weight", "activation"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode

    @property
    def weight_threshold_mode(self):
        return self.threshold_mode["weight"]

    @property
    def activation_threshold_mode(self):
        return self.threshold_mode["activation"]

    @property
    def _normalized_weight(self):
        conv_weight = self.weight_threshold_layer(self.weight)
        return conv_weight
        # weights = torch.zeros_like(self.weight)
        # weights.data[..., self.kernel_size[0]//2, self.kernel_size[1]//2] = 1
        # return weights

    @property
    def _normalized_weights(self):
        return self._normalized_weight

    @property
    def weight(self):
        if self.shared_weights is not None:
            return self.shared_weights
        return self.conv.weight

    @property
    def out_channels(self):
        return self.conv.out_channels

    def set_weights(self, new_weights: torch.Tensor) -> torch.Tensor:
        assert self.weight.shape == new_weights.shape, f"Weights must be of same shape {self.weight.shape}"
        self.conv.weight.data = new_weights
        return new_weights

    def set_bias(self, new_bias: torch.Tensor) -> torch.Tensor:
        assert self.bias.shape == new_bias.shape
        # self.conv.bias.data = new_bias
        assert (new_bias <= -0.5).all()
        self.conv.bias.data = self.softplus_layer.forward_inverse(-new_bias - 0.5)
        return new_bias

    def set_activation_P(self, new_P: torch.Tensor) -> torch.Tensor:
        assert self.activation_P.shape == new_P.shape
        self.activation_threshold_layer.P_.data = new_P
        return new_P

    @property
    def weight_P(self):
        if self.shared_weight_P is not None:
            return self.shared_weight_P
        return self._weight_P

    @property
    def activation_P(self):
        return self.activation_threshold_layer.P_

    @property
    def weights(self):
        return self.weight

    @property
    def bias(self):
        return -self.softplus_layer(self.conv.bias) - .5
        # return self.conv.bias


class BiSEC(BiSE):

    def __init__(self, *args, alpha_init=0, invert_thresholded_alpha=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv.bias = None
        self.complementation_layer = ComplementationLayer(
            self.complementation_threshold_mode, alpha_init=alpha_init, invert_thresholded_alpha=invert_thresholded_alpha
        )

        self._bias = nn.Parameter(torch.tensor([0.5]).float(), requires_grad=False)


    def forward(self, x: Tensor) -> Tensor:
        output = self.complementation_layer(x)
        output = super().forward(output)
        if self.thresholded_alpha < 1/2:
            return 1 - output
        # output = self.complementation_layer(output)
        return output

    @property
    def bias(self):
        return -(
            min(self.thresholded_alpha, 1 - self.thresholded_alpha) *
            (self._normalized_weight.sum() - 1) + self._bias
        )

    @property
    def alpha(self):
        return self.complementation_layer.alpha

    @property
    def thresholded_alpha(self):
        return self.complementation_layer.thresholded_alpha

    @property
    def complementation_threshold_mode(self):
        return self.threshold_mode["complementation"]

    @staticmethod
    def _init_threshold_mode(threshold_mode):
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["weight", "activation", "complementation"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode
