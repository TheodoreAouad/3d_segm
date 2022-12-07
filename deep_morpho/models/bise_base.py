from enum import Enum
from typing import Tuple, Union, Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .threshold_layer import dispatcher, ThresholdEnum
from .binary_nn import BinaryNN
from .bias_layer import BiasSoftplus, BiasRaw, BiasBiseSoftplusProjected, BiasBiseSoftplusReparametrized
from .weights_layer import WeightsThresholdedBise, WeightsNormalizedBiSE, WeightsEllipse, WeightsEllipseRoot
from deep_morpho.initializer import InitBiseHeuristicWeights, BiseInitializer, InitSybiseConstantVarianceWeights, InitBiseConstantVarianceWeights
from deep_morpho.binarization import (
    ClosestSelemEnum, BiseClosestMinDistBounds, distance_agg_min, distance_agg_max_second_derivative,
    ClosestSelemDistanceEnum, BiseClosestSelemWithDistanceAgg, distance_fn_to_bounds, BiseClosestMinDistOnCst
)
from general.utils import set_borders_to


# class ClosestSelemEnum(Enum):
#     MIN_DIST = 1
#     MAX_SECOND_DERIVATIVE = 2


# class ClosestSelemDistanceEnum(Enum):
#     DISTANCE_TO_BOUNDS = 1
#     DISTANCE_BETWEEN_BOUNDS = 2
#     DISTANCE_TO_AND_BETWEEN_BOUNDS = 3


class BiseBiasOptimEnum(Enum):
    RAW = 0     # no transformation to bias
    POSITIVE = 1    # only softplus applied
    POSITIVE_INTERVAL_PROJECTED = 2     # softplus applied and projected gradient on the relevent values [min(W), W.sum()] (no torch grad)
    POSITIVE_INTERVAL_REPARAMETRIZED = 3     # softplus applied and reparametrized on the relevent values [min(W), W.sum()] (with torch grad)


class BiseWeightsOptimEnum(Enum):
    THRESHOLDED = 0
    NORMALIZED = 1
    ELLIPSE = 2
    ELLIPSE_ROOT = 3


conv_kwargs = {"in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias",
    "padding_mode", "device", "dtype"}


class BiSEBase(BinaryNN):

    operation_code = {"erosion": 0, "dilation": 1}

    def __init__(
        self,
        kernel_size: Tuple,
        threshold_mode: Union[Dict[str, str], str] = {"weight": "softplus", "activation": "tanh"},
        activation_P: float = 10,
        constant_activation_P: bool = False,
        weights_optim_mode: BiseWeightsOptimEnum = BiseWeightsOptimEnum.THRESHOLDED,
        weights_optim_args: Dict = {},
        shared_weights: torch.tensor = None,  # deprecated
        # initializer_method: InitBiseEnum = InitBiseEnum.CUSTOM_HEURISTIC,
        # initializer_args: Dict = {"init_bias_value": 1},
        initializer: BiseInitializer = InitBiseHeuristicWeights(init_bias_value=1, input_mean=0.5),
        # init_bias_value: float = 1,
        # input_mean: float = 0.5,
        # init_weight_mode: InitBiseEnum = InitBiseEnum.CUSTOM_HEURISTIC,
        in_channels: int = 1,
        out_channels: int = 1,
        do_mask_output: bool = False,
        closest_selem_method: ClosestSelemEnum = ClosestSelemEnum.MIN_DIST_DIST_TO_CST,
        closest_selem_args: Dict = {},
        # closest_selem_distance_fn: ClosestSelemDistanceEnum = ClosestSelemDistanceEnum.DISTANCE_TO_AND_BETWEEN_BOUNDS,
        bias_optim_mode: BiseBiasOptimEnum = BiseBiasOptimEnum.POSITIVE_INTERVAL_REPARAMETRIZED,
        bias_optim_args: Dict = {},
        padding=None,
        padding_mode: str = "replicate",
        *args,
        **kwargs
    ):
        super().__init__()
        self.threshold_mode = self._init_threshold_mode(threshold_mode)
        self.activation_P_init = activation_P
        self.kernel_size = self._init_kernel_size(kernel_size)
        self.do_mask_output = do_mask_output
        self.closest_selem_method = closest_selem_method
        # self.closest_selem_distance_fn = closest_selem_distance_fn
        self.closest_selem_args = closest_selem_args
        self.bias_optim_mode = bias_optim_mode
        self.bias_optim_args = bias_optim_args
        self.weights_optim_mode = weights_optim_mode
        self.weights_optim_args = weights_optim_args

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.padding = self.kernel_size[0] // 2 if padding is None else padding
        self.padding_mode = padding_mode

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            # padding="same",
            padding=self.padding,
            padding_mode=self.padding_mode,
            bias=False,
            **{k: kwargs[k] for k in conv_kwargs.intersection(kwargs.keys())}
        )

        self.shared_weights = shared_weights

        self.activation_threshold_layer = dispatcher[self.activation_threshold_mode](P_=activation_P, constant_P=constant_activation_P, n_channels=out_channels, axis_channels=1)

        self.bias_handler = self.create_bias_handler(**self.bias_optim_args)
        self.weights_handler = self.create_weights_handler(**self.weights_optim_args)
        self.closest_selem_handler = self.create_closest_selem_handler(**self.closest_selem_args)

        self.initializer = initializer
        self.initializer.initialize(self)

        self._closest_selem = np.zeros((out_channels, in_channels, *self.kernel_size)).astype(bool)
        self._closest_operation = np.zeros((out_channels))
        self._closest_selem_dist = np.zeros((out_channels))

        self._learned_selem = np.zeros((out_channels, in_channels, *self.kernel_size)).astype(bool)
        self._learned_operation = np.zeros((out_channels))
        self._is_activated = np.zeros((out_channels)).astype(bool)

        self.update_binary_selems()


    def create_closest_selem_handler(self, **kwargs):
        if self.closest_selem_method == ClosestSelemEnum.MIN_DIST_DIST_TO_BOUNDS:
            return BiseClosestMinDistBounds(bise_module=self, **kwargs)

        elif self.closest_selem_method == ClosestSelemEnum.MIN_DIST:
            kwargs['distance_agg_fn'] = distance_agg_min
            kwargs['distance_fn'] = kwargs.get('distance_fn', distance_fn_to_bounds)
            return BiseClosestSelemWithDistanceAgg(bise_module=self, **kwargs)

        elif self.closest_selem_method == ClosestSelemEnum.MAX_SECOND_DERIVATIVE:
            kwargs['distance_agg_fn'] = distance_agg_max_second_derivative
            return BiseClosestSelemWithDistanceAgg(bise_module=self, **kwargs)

        elif self.closest_selem_method == ClosestSelemEnum.MIN_DIST_DIST_TO_CST:
            return BiseClosestMinDistOnCst(bise_module=self, **kwargs)


    def create_bias_handler(self, **kwargs):
        if self.bias_optim_mode == BiseBiasOptimEnum.POSITIVE:
            kwargs['offset'] = kwargs.get('offset', 0.5)
            return BiasSoftplus(bise_module=self, **kwargs)

        elif self.bias_optim_mode == BiseBiasOptimEnum.RAW:
            return BiasRaw(bise_module=self, **kwargs)

        elif self.bias_optim_mode == BiseBiasOptimEnum.POSITIVE_INTERVAL_PROJECTED:
            kwargs['offset'] = kwargs.get('offset', 0)
            return BiasBiseSoftplusProjected(bise_module=self, **kwargs)

        elif self.bias_optim_mode == BiseBiasOptimEnum.POSITIVE_INTERVAL_REPARAMETRIZED:
            kwargs['offset'] = kwargs.get('offset', 0)
            return BiasBiseSoftplusReparametrized(bise_module=self, **kwargs)

        raise NotImplementedError(f'self.bias_optim_mode must be in {BiseBiasOptimEnum._member_names_}')

    def create_weights_handler(self, **kwargs):
        if self.weights_optim_mode == BiseWeightsOptimEnum.THRESHOLDED:
            return WeightsThresholdedBise(bise_module=self, threshold_mode=self.weight_threshold_mode, **kwargs)

        elif self.weights_optim_mode == BiseWeightsOptimEnum.NORMALIZED:
            return WeightsNormalizedBiSE(bise_module=self, threshold_mode=self.weight_threshold_mode, **kwargs)

        elif self.weights_optim_mode == BiseWeightsOptimEnum.ELLIPSE:
            return WeightsEllipse(bise_module=self, **kwargs)

        elif self.weights_optim_mode == BiseWeightsOptimEnum.ELLIPSE_ROOT:
            return WeightsEllipseRoot(bise_module=self, **kwargs)

        raise NotImplementedError(f'self.bias_optim_mode must be in {BiseWeightsOptimEnum._member_names_}')

    def update_closest_selems(self):
        for chan in range(self.out_channels):
            self.find_closest_selem_and_operation_chan(chan)

    def update_binary_selems(self):
        self.update_closest_selems()
        self.update_learned_selems()

    def update_learned_selems(self):
        for chan in range(self.out_channels):
            self.find_selem_and_operation_chan(chan)

    def binary(self, mode: bool = True):
        if mode:
            self.update_binary_selems()
        return super().binary(mode)

    # TODO: move to BiSE class
    @staticmethod
    def bise_from_selem(selem: np.ndarray, operation: str, threshold_mode: str = "softplus", **kwargs):
        assert set(np.unique(selem)).issubset([0, 1])
        net = BiSEBase(kernel_size=selem.shape, threshold_mode=threshold_mode, out_channels=1, **kwargs)

        if threshold_mode == "identity":
            net.set_param_from_weights(torch.FloatTensor(selem)[None, None, ...])
        else:
            net.set_param_from_weights((torch.tensor(selem) + 0.01)[None, None, ...])
        bias_value = -.5 if operation == "dilation" else -float(selem.sum()) + .5
        net.set_bias(torch.FloatTensor([bias_value]))

        net.update_learned_selems()

        return net

    @staticmethod
    def _init_kernel_size(kernel_size: Union[Tuple, int]):
        if isinstance(kernel_size, int):
            return (kernel_size, kernel_size)
        return kernel_size

    # def init_weights_and_bias(self):
    #     self.init_weights()
    #     self.init_bias()

    # def init_weights(self):
    #     if self.init_weight_mode == InitBiseEnum.NORMAL:
    #         self.set_param_from_weights(self._init_normal_identity(self.kernel_size, self.out_channels))
    #     elif self.init_weight_mode == InitBiseEnum.IDENTITY:
    #         self._init_as_identity()
    #     elif self.init_weight_mode == InitBiseEnum.KAIMING_UNIFORM:
    #         self.set_param_from_weights(self.weight + 1)
    #     elif self.init_weight_mode == InitBiseEnum.CUSTOM_HEURISTIC:
    #         nb_params = torch.tensor(self._normalized_weights.shape[1:]).prod()
    #         mean = self.init_bias_value / (self.input_mean * nb_params)
    #         std = .5
    #         lb = mean * (1 - std)
    #         ub = mean * (1 + std)
    #         self.set_param_from_weights(
    #             torch.rand_like(self.weights) * (lb - ub) + ub
    #         )
    #     elif self.init_weight_mode == InitBiseEnum.CUSTOM_CONSTANT:
    #         p = 1
    #         nb_params = torch.tensor(self._normalized_weights.shape[1:]).prod()

    #         if self.init_bias_value == "auto":
    #             # To keep trakc with previous experiment
    #             # if self.input_mean > 0.7:
    #             #     lb1 = 1/p * torch.sqrt(6*nb_params / (12 + 1/self.input_mean**2))
    #             # else:
    #             #     lb1 = 1/(2*p) * torch.sqrt(3/2 * nb_params)
    #             lb1 = 1/p * torch.sqrt(6*nb_params / (12 + 1/self.input_mean**2))
    #             lb2 = 1 / p * torch.sqrt(nb_params / 2)
    #             self.init_bias_value = (lb1 + lb2) / 2

    #         mean = self.init_bias_value / (self.input_mean * nb_params)
    #         sigma = (2 * nb_params - 4 * self.init_bias_value**2 * p ** 2) / (p ** 2 * nb_params ** 2)
    #         diff = torch.sqrt(3 * sigma)
    #         lb = mean - diff
    #         ub = mean + diff

    #         new_weights = torch.rand_like(self.weights) * (lb - ub) + ub
    #         self.set_param_from_weights(
    #             # new_weights / new_weights.sum()  # DEBUG
    #             new_weights
    #         )
    #     else:
    #         warnings.warn(f"init weight mode {self.init_weight_mode} not recognized.")
    #     pass

    # def init_bias(self):
    #     self.set_bias(
    #         torch.zeros_like(self.bias) - self.init_bias_value
    #     )

    # @staticmethod
    # def _init_normal_identity(kernel_size, chan_output, std=0.3, mean=1):
    #     weights = torch.randn((chan_output,) + kernel_size)[:, None, ...] * std - mean
    #     weights[..., kernel_size[0] // 2, kernel_size[1] // 2] += 2*mean
    #     return weights

    # def _init_as_identity(self):
    #     self.conv.weight.data.fill_(-1)
    #     shape = self.conv.weight.shape
    #     self.conv.weight.data[..., shape[-2]//2, shape[-1]//2] = 1

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
        Replaces the BiSEBase with the closest learned operation.
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
        weights = torch.zeros_like(self._normalized_weight)
        bias = torch.zeros_like(self.bias)
        operations = torch.zeros_like(self.bias)

        weights[self._is_activated] = torch.FloatTensor(self._learned_selem[self._is_activated]).to(weights.device)
        weights[~self._is_activated] = torch.FloatTensor(self._closest_selem[~self._is_activated]).to(bias.device)

        dil_key = self.operation_code['dilation']
        ero_key = self.operation_code['erosion']
        operations[self._is_activated] = torch.FloatTensor(self._learned_operation[self._is_activated]).to(operations.device)
        operations[~self._is_activated] = torch.FloatTensor(self._closest_operation[~self._is_activated]).to(operations.device)
        bias[operations == dil_key] = -0.5
        bias[operations == ero_key] = -weights[operations == ero_key].sum((1, 2, 3)) + 0.5

        return weights, bias

    def mask_output(self, output):
        masker = set_borders_to(torch.ones(output.shape[-2:], requires_grad=False), border=np.array(self.kernel_size) // 2)[None, None, ...]
        return output * masker.to(output.device)

    # @staticmethod
    # def distance_to_bounds(bound_fn, normalized_weights: torch.Tensor, bias: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1) -> float:
    #     assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
    #     lb, ub = bound_fn(normalized_weights=normalized_weights, S=S, v1=v1, v2=v2)
    #     dist_lb = lb + bias  # if dist_lb < 0 : lower bound respected
    #     dist_ub = -bias - ub  # if dist_ub < 0 : upper bound respected
    #     return max(dist_lb, dist_ub, 0)

    # @staticmethod
    # def distance_between_bounds(bound_fn, normalized_weights: torch.Tensor, bias: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1) -> float:
    #     assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
    #     lb, ub = bound_fn(normalized_weights=normalized_weights, S=S, v1=v1, v2=v2)
    #     return lb - ub

    # def distance_dispatcher(self, bound_fn, method: ClosestSelemDistanceEnum, normalized_weights: torch.Tensor, bias: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1) -> float:
    #     if method == ClosestSelemDistanceEnum.DISTANCE_TO_BOUNDS:
    #         distance_fn = self.distance_to_bounds
    #     elif method == ClosestSelemDistanceEnum.DISTANCE_BETWEEN_BOUNDS:
    #         distance_fn = self.distance_between_bounds
    #     elif method == ClosestSelemDistanceEnum.DISTANCE_TO_AND_BETWEEN_BOUNDS:
    #         distance_fn = lambda *args, **kwargs: self.distance_to_bounds(*args, **kwargs) + self.distance_between_bounds(*args, **kwargs)
    #     return distance_fn(bound_fn, normalized_weights, bias, S, v1, v2)

    # def distance_to_dilation(self, method: ClosestSelemDistanceEnum, normalized_weights: torch.Tensor, bias: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1) -> float:
    #     return self.distance_dispatcher(self.bias_bounds_dilation, method, normalized_weights, bias, S, v1, v2)

    # def distance_to_erosion(self, method: ClosestSelemDistanceEnum, normalized_weights: torch.Tensor, bias: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1) -> float:
    #     return self.distance_dispatcher(self.bias_bounds_erosion, method, normalized_weights, bias, S, v1, v2)

    def is_erosion_by(self, normalized_weights: torch.Tensor, bias: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1) -> bool:
        assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
        lb, ub = self.bias_bounds_erosion(normalized_weights=normalized_weights, S=S, v1=v1, v2=v2)
        return lb < -bias < ub

    def is_dilation_by(self, normalized_weights: torch.Tensor, bias: torch.Tensor, S: np.ndarray, v1: float = 0, v2: float = 1) -> bool:
        assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
        lb, ub = self.bias_bounds_dilation(normalized_weights=normalized_weights, S=S, v1=v1, v2=v2)
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

    def find_selem_for_operation_chan(self, operation: str, chout: int = 0, v1: float = 0, v2: float = 1):
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
        weights = self._normalized_weight[chout]
        bias = self.bias[chout]
        is_op_fn = {'dilation': self.is_dilation_by, 'erosion': self.is_erosion_by}[operation]
        born = {'dilation': -bias / v2, "erosion": (weights.sum() + bias) / (1 - v1)}[operation]

        selem = (weights > born).cpu().detach().numpy()
        if not selem.any():
            return None

        if is_op_fn(weights, bias, selem, v1, v2):
            return selem
        return None

    # def compute_selem_dist_for_operation_chan(
    #     self, weights, bias, operation: str, closest_selem_distance_fn: ClosestSelemDistanceEnum, chout: int = 0, v1: float = 0, v2: float = 1
    # ):
    #     weights = weights[chout]
    #     weight_values = weights.unique().detach().cpu().numpy()
    #     bias = bias[chout]
    #     distance_fn = {'dilation': self.distance_to_dilation, 'erosion': self.distance_to_erosion}[operation]

    #     dists = np.zeros_like(weight_values)
    #     selems = []
    #     for value_idx, value in enumerate(weight_values):
    #         selem = (weights >= value).cpu().detach().numpy()
    #         dists[value_idx] = distance_fn(method=closest_selem_distance_fn, normalized_weights=weights, bias=bias, S=selem, v1=v1, v2=v2)
    #         selems.append(selem)

    #     return selems, dists

    # def find_closest_selem_for_operation_chan(self, weights, bias, operation: str, closest_selem_method, closest_selem_distance_fn, chout: int = 0, v1: float = 0, v2: float = 1):
    #     """
    #     We find the selem for either a dilation or an erosion. In theory, there is at most one selem that works.

    #     Args:
    #         operation (str): 'dilation' or 'erosion', the operation we want to check for
    #         v1 (float): the lower value of the almost binary
    #         v2 (float): the upper value of the almost binary (input not in ]v1, v2[)

    #     Returns:
    #         np.ndarray: the closest selem
    #         float: the distance to the constraint space
    #     """
    #     if closest_selem_method == ClosestSelemEnum.MIN_DIST:
    #         return self.closest_selem_by_min_dists(weights=weights, bias=bias, chout=chout, operation=operation, closest_selem_distance_fn=closest_selem_distance_fn, v1=v1, v2=v2)

    #     if closest_selem_method == ClosestSelemEnum.MAX_SECOND_DERIVATIVE:
    #         return self.closest_selem_by_second_derivative(weights=weights, bias=bias, chout=chout, operation=operation, closest_selem_distance_fn=closest_selem_distance_fn, v1=v1, v2=v2)

    # def closest_selem_by_min_dists(self, weights, bias, operation, chout, closest_selem_distance_fn, v1, v2):
    #     selems, dists = self.compute_selem_dist_for_operation_chan(
    #         weights=weights, bias=bias, chout=chout, operation=operation, closest_selem_distance_fn=closest_selem_distance_fn, v1=v1, v2=v2
    #     )

    #     idx_min = dists.argmin()
    #     return selems[idx_min], dists[idx_min]

    # def closest_selem_by_second_derivative(self, weights, bias, chout, operation, closest_selem_distance_fn, v1, v2):
    #     selems, dists = self.compute_selem_dist_for_operation_chan(
    #         weights=weights, bias=bias, chout=chout, operation=operation, closest_selem_distance_fn=closest_selem_distance_fn, v1=v1, v2=v2
    #     )
    #     idx_min = (dists[2:] + dists[:-2] - 2 * dists[1:-1]).argmax() + 1
    #     return selems[idx_min], dists[idx_min]

    # def find_closest_selem_and_operation_chan_static(
    #     self, weights, bias, chout=0, closest_selem_method=ClosestSelemEnum.MIN_DIST,
    #     closest_selem_distance_fn=ClosestSelemDistanceEnum.DISTANCE_TO_BOUNDS, v1=0, v2=1
    # ):
    #     final_dist = np.infty
    #     for operation in ['dilation', 'erosion']:
    #         new_selem, new_dist = self.find_closest_selem_for_operation_chan(
    #             weights=weights, bias=bias, chout=chout, operation=operation,
    #             closest_selem_method=closest_selem_method, closest_selem_distance_fn=closest_selem_distance_fn, v1=v1, v2=v2
    #         )
    #         if new_dist < final_dist:
    #             final_dist = new_dist
    #             final_selem = new_selem
    #             final_operation = operation  # str array has 1 character

    #     return final_dist, final_selem, final_operation

    def find_closest_selem_and_operation_chan(self, chout: int = 0, v1: float = 0, v2: float = 1) -> Tuple[np.ndarray, str, float]:
        """Find the closest selem and the operation given the almost binary features.

        Args:
            v1 (float): lower bound of almost binary input deadzone. Defaults to 0.
            v2 (float): upper bound of almost binary input deadzone. Defaults to 1.

        Returns:
            (np.ndarray, str, float): if the selem is found, returns the selem and the operation
        """
        # final_dist, final_selem, final_operation = self.find_closest_selem_and_operation_chan_static(
        #     weights=self._normalized_weight, bias=self.bias, chout=chout, closest_selem_method=self.closest_selem_method,
        #     closest_selem_distance_fn=self.closest_selem_distance_fn, v1=v1, v2=v2
        # )

        final_dist, final_selem, final_operation = self.closest_selem_handler(chout=chout, v1=v1, v2=v2)

        self._closest_selem[chout] = final_selem.astype(bool)
        self._closest_operation[chout] = self.operation_code[final_operation]
        self._closest_selem_dist[chout] = final_dist
        return final_selem, final_operation, final_dist

    def find_selem_and_operation_chan(self, chout: int = 0, v1: float = 0, v2: float = 1):
        """Find the selem and the operation given the almost binary features.

        Args:
            v1 (float): lower bound of almost binary input deadzone. Defaults to 0.
            v2 (float): upper bound of almost binary input deadzone. Defaults to 1.

        Returns:
            (np.ndarray, operation): if the selem is found, returns the selem and the operation
            (None, None): if nothing is found, returns None
        """
        for operation in ['dilation', 'erosion']:
            selem = self.find_selem_for_operation_chan(chout=chout, operation=operation, v1=v1, v2=v2)
            if selem is not None:
                self._learned_selem[chout] = selem.astype(bool)
                self._learned_operation[chout] = self.operation_code[operation]  # str array has 1 character
                self._is_activated[chout] = True
                return selem, operation

        self._is_activated[chout] = False
        self._learned_selem[chout] = None
        self._learned_operation[chout] = None

        return None, None

    def get_outputs_bounds(self, v1: float = 0, v2: float = 1):
        """If the BiSEBase is learned, returns the bounds of the deadzone of the almost binary output.

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

    @staticmethod
    def _init_threshold_mode(threshold_mode):
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["weight", "activation"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode

    @property
    def weight_threshold_layer(self):
        return self.weights_handler.threshold_layer

    @property
    def weight_threshold_mode(self):
        return self.threshold_mode["weight"]

    @property
    def activation_threshold_mode(self):
        return self.threshold_mode["activation"]

    @property
    def _normalized_weight(self):
        return self.weights_handler()

    @property
    def _normalized_weights(self):
        return self._normalized_weight

    @property
    def weight(self):
        return self.weights_handler.param

    def set_weights_param(self, new_param: torch.Tensor) -> torch.Tensor:
        return self.weights_handler.set_param(new_param)

    def set_param_from_weights(self, new_weights: torch.Tensor) -> torch.Tensor:
        return self.weights_handler.set_param_from_weights(new_weights)

    def set_bias(self, new_bias: torch.Tensor) -> torch.Tensor:
        assert self.bias.shape == new_bias.shape
        self.bias_handler.set_bias(new_bias)
        return new_bias

    def set_activation_P(self, new_P: torch.Tensor) -> torch.Tensor:
        assert self.activation_P.shape == new_P.shape
        self.activation_threshold_layer.P_.data = new_P
        return new_P

    @property
    def activation_P(self):
        return self.activation_threshold_layer.P_

    @property
    def weight_P(self):
        return self.weight_threshold_layer.P_

    @property
    def weights(self):
        return self.weight

    @property
    def bias(self):
        return self.bias_handler()

    @property
    def closest_selem(self):
        return self._closest_selem

    @property
    def closest_operation(self):
        return self._closest_operation

    @property
    def closest_selem_dist(self):
        return self._closest_selem_dist

    @property
    def learned_selem(self):
        return self._learned_selem

    @property
    def learned_operation(self):
        return self._learned_operation

    @property
    def is_activated(self):
        return self._is_activated


class SyBiSEBase(BiSEBase):
    POSSIBLE_THRESHOLDS = (dispatcher[ThresholdEnum.tanh_symetric], dispatcher[ThresholdEnum.sigmoid_symetric])

    def __init__(
        self,
        *args,
        threshold_mode: Union[Dict[str, str], str] = {"weight": "softplus", "activation": "tanh_symetric"},
        bias_optim_mode: BiseBiasOptimEnum = BiseBiasOptimEnum.RAW,
        # init_bias_value: float = 0,
        # input_mean: float = 0,
        initializer: BiseInitializer = InitSybiseConstantVarianceWeights(input_mean=0, mean_weight="auto"),
        # initializer_method: InitBiseEnum = InitBiseEnum.CUSTOM_CONSTANT,
        # initializer_args: Dict = {},
        mean_weight_value: float = "auto",
        **kwargs
    ):
        self.mean_weight_value = mean_weight_value
        super().__init__(*args,
            threshold_mode=threshold_mode,
            bias_optim_mode=bias_optim_mode,
            # initializer_method=initializer_method,
            # initializer_args=initializer_args,
            initializer=initializer,
        **kwargs)
        assert isinstance(self.activation_threshold_layer, self.POSSIBLE_THRESHOLDS), "Choose a symetric threshold for activation."

    def bias_bounds_erosion(self, normalized_weights: torch.Tensor, S: np.ndarray, v1=None, v2: float = 1) -> Tuple[float, float]:
        lb_dil, ub_dil = self.bias_bounds_dilation(normalized_weights, S, v2=v2)
        return -ub_dil, -lb_dil

    @staticmethod
    def bias_bounds_dilation(normalized_weights: torch.Tensor, S: np.ndarray, v1=None, v2: float = 1) -> Tuple[float, float]:
        S = S.astype(bool)
        epsilon = v2
        W = normalized_weights.cpu().detach().numpy()
        return (
            -epsilon * W[(W > 0) & S].sum() - W[(W < 0) & S].sum() + np.abs(W[~S]).sum(),
            -np.abs(W).sum() + (1 + epsilon) * max(W[S].min(), 0)
        )
