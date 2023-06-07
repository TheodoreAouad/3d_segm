from typing import Tuple, TYPE_CHECKING, List
from enum import Enum
from functools import partial

from tqdm import tqdm
import numpy as np
import cvxpy as cp

if TYPE_CHECKING:
    from deep_morpho.bise import BiSE
    import torch


from general.nn.experiments.experiment_methods import ExperimentMethods
from .projection_constant_set import ProjectionConstantSet


class ClosestSelemEnum(Enum):
    MIN_DIST = 1
    MAX_SECOND_DERIVATIVE = 2
    MIN_DIST_DIST_TO_BOUNDS = 3
    MIN_DIST_DIST_TO_CST = 4


class ClosestSelemDistanceEnum(Enum):
    DISTANCE_TO_BOUNDS = 1
    DISTANCE_BETWEEN_BOUNDS = 2
    DISTANCE_TO_AND_BETWEEN_BOUNDS = 3


def distance_agg_min(self, selems: np.ndarray, dists: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx_min = dists.argmin()
    return selems[idx_min], dists[idx_min]


def distance_agg_max_second_derivative(self, selems: np.ndarray, dists: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx_min = (dists[2:] + dists[:-2] - 2 * dists[1:-1]).argmax() + 1
    return selems[idx_min], dists[idx_min]


def distance_to_bounds_base(bound_fn, weights: "torch.Tensor", bias: "torch.Tensor", S: np.ndarray, v1: float = 0, v2: float = 1) -> float:
    assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
    lb, ub = bound_fn(weights=weights, S=S, v1=v1, v2=v2)
    dist_lb = lb + bias  # if dist_lb < 0 : lower bound respected
    dist_ub = -bias - ub  # if dist_ub < 0 : upper bound respected
    return max(dist_lb, dist_ub, 0)


def distance_fn_to_bounds(self, weights: "torch.Tensor", bias: "torch.Tensor", operation: str, S: np.ndarray, v1: float, v2: float) -> float:
    if operation == "dilation":
        bound_fn = self.bise_module.bias_bounds_dilation
    elif operation == "erosion":
        bound_fn = self.bise_module.bias_bounds_erosion
    else:
        raise SyntaxError("operation must be either 'dilation' or 'erosion'.")

    return distance_to_bounds_base(bound_fn, weights=weights, bias=bias, S=S, v1=v1, v2=v2)


def distance_between_bounds_base(bound_fn, weights: "torch.Tensor", bias: "torch.Tensor", S: np.ndarray, v1: float = 0, v2: float = 1) -> float:
    assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
    lb, ub = bound_fn(weights=weights, S=S, v1=v1, v2=v2)
    return lb - ub


def distance_fn_between_bounds(self, weights: "torch.Tensor", bias: "torch.Tensor", operation: str, S: np.ndarray, v1: float, v2: float) -> float:
    if operation == "dilation":
        bound_fn = self.bise_module.bias_bounds_dilation
    elif operation == "erosion":
        bound_fn = self.bise_module.bias_bounds_erosion
    else:
        raise SyntaxError("operation must be either 'dilation' or 'erosion'.")

    return distance_between_bounds_base(bound_fn, weights=weights, bias=bias, S=S, v1=v1, v2=v2)


class BiseClosestSelemHandler(ExperimentMethods):

    def __init__(self, bise_module: "BiSE" = None):
        self.bise_module = bise_module

    def __call__(self, chans: List[int] = None, v1: float = 0, v2: float = 1, verbose: bool = True,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the closest selem and operation for all given chans. If no chans are given, for all chans.

        Args:
            chans (list): list of given channels. If non given, all channels computed.
            v1 (float, optional): first argument for almost binary. Defaults to 0.
            v2 (float, optional): second argument for almost binary. Defaults to 1.
            verbose (bool, optional): shows progress bar.

        Returns:
            array (n_chan, *kernel_size) bool, array(n_chan) str, array(n_chan) float: the selem, the operation and the distance to the closest selem
        """
        pass


class BiseClosestSelemWithDistanceAgg(BiseClosestSelemHandler):
    """ Children must define:
    distance_fn(self, weights: "torch.Tensor", bias: "torch.Tensor", operation: str, S: np.ndarray, v1: float, v2: float) -> float
        given an operation, weights, bias, selems and almost binary bounds, outputs a similarity measure float
    distance_agg_fn(self, selems: np.ndarray, dists: np.ndarray) -> Tuple[np.ndarray, float]
        given a list of selems and dists in the form of arrays, outputs the best selem given the distances, and its corresponding distance
    """

    def __init__(self, distance_fn=None, distance_agg_fn=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distance_fn = partial(distance_fn, self=self) if distance_fn is not None else None
        self._distance_agg_fn = partial(distance_agg_fn, self=self) if distance_agg_fn is not None else None

    @property
    def distance_fn(self):
        return self._distance_fn

    @property
    def distance_agg_fn(self):
        return self._distance_agg_fn

    def compute_selem_dist_for_operation_chan(
        self, weights, bias, operation: str, chout: int = 0, v1: float = 0, v2: float = 1
    ):
        weights = weights[chout]
        # weight_values = weights.unique().detach().cpu().numpy()
        weight_values = np.unique(weights)
        bias = bias[chout]

        dists = np.zeros_like(weight_values)
        selems = []
        for value_idx, value in enumerate(weight_values):
            selem = (weights >= value)
            # selem = (weights >= value).cpu().detach().numpy()
            dists[value_idx] = self.distance_fn(weights=weights, bias=bias, operation=operation, S=selem, v1=v1, v2=v2)
            selems.append(selem)

        return selems, dists


    def find_closest_selem_and_operation_chan(
        self, weights, bias, chout=0, v1=0, v2=1
    ):
        final_dist = np.infty
        weights = weights.detach().cpu().numpy()
        bias = bias.detach().cpu().numpy()
        for operation in ['dilation', 'erosion']:
            selems, dists = self.compute_selem_dist_for_operation_chan(
                weights=weights, bias=bias, chout=chout, operation=operation, v1=v1, v2=v2
            )
            new_selem, new_dist = self.distance_agg_fn(selems=selems, dists=dists)
            if new_dist < final_dist:
                final_dist = new_dist
                final_selem = new_selem
                final_operation = operation

        return final_selem, final_operation, final_dist

    def __call__(self, chans: List[int] = None, v1: float = 0, v2: float = 1, verbose: bool = True,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if chans is None:
            chans = range(self.bise_module.out_channels)
        if verbose:
            chans = tqdm(chans, leave=False, desc="Approximate binarization")

        # for chan in chans:
        #     self.find_closest_selem_and_operation_chan(chan)

        closest_selems = np.zeros((len(chans), *self.bise_module.kernel_size), dtype=bool)
        closest_operations = np.zeros(len(chans), dtype=str)
        closest_dists = np.zeros(len(chans))

        for chout_idx, chout in enumerate(chans):
            selem, op, dist = self.find_closest_selem_and_operation_chan(
                weights=self.bise_module.weights, bias=self.bise_module.bias, chout=chout, v1=v1, v2=v2
            )
            closest_selems[chout_idx] = selem.astype(bool)
            closest_operations[chout_idx] = self.bise_module.operation_code[op]
            closest_dists[chout_idx] = dist

        return closest_selems, closest_operations, closest_dists


class BiseClosestMinDistBounds(BiseClosestSelemWithDistanceAgg):

    def __init__(self, *args, **kwargs):
        super().__init__(distance_fn=distance_fn_to_bounds, distance_agg_fn=distance_agg_min, *args, **kwargs)


class BiseClosestActivationSpaceIteratedPositive(BiseClosestSelemWithDistanceAgg):

    def __init__(self, *args, **kwargs):
        super().__init__(distance_agg_fn=distance_agg_min, *args, **kwargs)
        self._distance_fn = partial(self.solve)

    def solve(self, weights: np.ndarray, bias: np.ndarray, operation: str, S: np.ndarray, v1: float, v2: float) -> float:
        if operation == "erosion":
            bias = weights.sum() - bias
        # print("banana")

        weights = weights.flatten()
        S = S.flatten()

        Wvar = cp.Variable(weights.shape)
        bvar = cp.Variable(1)

        if operation == "dilation":
            constraints = self.dilation_constraints(Wvar, bvar, S)
        elif operation == "erosion":
            constraints = self.erosion_constraints(Wvar, bvar, S)
        else:
            raise ValueError("operation must be dilation or erosion")

        objective = cp.Minimize(1/2 * cp.sum_squares(Wvar - weights) + 1/2 * cp.sum_squares(bvar + bias))
        prob = cp.Problem(objective, constraints)
        prob.solve()

        return prob.value

    def dilation_constraints(self, Wvar: cp.Variable, bvar: cp.Variable, S: np.ndarray):
        self.constraint0 = [cp.sum(Wvar[~S]) <= bvar]
        self.constraintsT = [bvar <= Wvar[S]]
        self.constraintsK = [Wvar >= 0]
        return self.constraint0 + self.constraintsT + self.constraintsK

    def erosion_constraints(self, Wvar: cp.Variable, bvar: cp.Variable, S: np.ndarray):
        self.constraint0 = [cp.sum(Wvar[S]) >= bvar]
        self.constraintsT = [cp.sum(Wvar) - Wvar[S] <= bvar]
        self.constraintsK = [Wvar >= 0]
        return self.constraint0 + self.constraintsT + self.constraintsK


class BiseClosestMinDistOnCstOld(BiseClosestSelemWithDistanceAgg):

    def __init__(self, *args, **kwargs):
        super().__init__(distance_fn=self.distance_fn_selem, distance_agg_fn=distance_agg_min, *args, **kwargs)
        self.distance_fn_bias = partial(distance_fn_between_bounds, self=self)

    @staticmethod
    def distance_fn_selem(self, weights: "torch.Tensor", bias: "torch.Tensor", operation: str, S: np.ndarray, v1: float, v2: float) -> float:
        W = weights.cpu().detach().numpy()
        return (W ** 2).sum() - 1 / S.sum() * (W[S].sum()) ** 2

    def find_closest_selem_and_operation_chan(
        self, weights, bias, chout=0, v1=0, v2=1
    ):
        final_dist_selem = np.infty
        final_dist_bias = np.infty

        selems, dists = self.compute_selem_dist_for_operation_chan(
            weights=weights, bias=bias, chout=chout, operation=None, v1=v1, v2=v2
        )
        new_selem, new_dist = self.distance_agg_fn(selems=selems, dists=dists)
        if new_dist < final_dist_selem:
            final_dist_selem = new_dist
            final_selem = new_selem

        wsum = weights[chout].sum()
        if -bias[chout] <= wsum / 2:
            final_operation = "dilation"
        else:
            final_operation = "erosion"

        final_dist_bias = (-bias[chout] - wsum).abs().cpu().detach().numpy()

        # for operation in ['dilation', 'erosion']:
        #     dist_bias = self.distance_fn_bias(weights=weights[chout], bias=bias, operation=operation, S=final_selem, v1=v1, v2=v2)
        #     if dist_bias < final_dist_bias:
        #         final_operation = operation

        final_dist = final_dist_selem + final_dist_bias

        return final_selem, final_operation, final_dist


class BiseClosestMinDistOnCst(BiseClosestSelemHandler):
    # operation_code = {"erosion": 0, "dilation": 1}  # TODO: be coherent with bisebase attribute operation_code
    # @staticmethod
    # def distance_fn_selem(weights: "torch.Tensor", S: np.ndarray,) -> float:
    #     # We reshape to sum over all dimensions except the first one.
    #     weights = weights.reshape(weights.shape[0], -1)
    #     S = S.reshape(S.shape[0], -1)
    #     return (weights ** 2).sum(1) - 1 / S.sum(1) * (np.where(S, weights, 0).sum(axis=1)) ** 2

    # @staticmethod
    # def find_best_index(w_values: np.ndarray) -> np.ndarray:
    #     r"""Gives the arg maximum of the distance function $\frac{\sum_{k \in S}{W_k}}{\sqrt{card{S}}} = \frac{\sum_{k = 1}^j{w_k}}{\sqrt{j}}$
    #     with $w_k$ the sorted values in descending order of the weights $W$.

    #     Args:
    #         w_values (np.ndarray): (n_chout, prod(kernel_size))

    #     Returns:
    #         array (n_chout): the index of the best value, for each channel.
    #     """
    #     return (np.cumsum(w_values, axis=1) / np.sqrt(np.arange(1, 1+w_values.shape[1]))).argmax(1)


    # @classmethod
    # def compute_closest_constant(cls, weights: np.ndarray, bias: np.ndarray, verbose: bool = True,) -> (np.ndarray, np.ndarray, np.ndarray):
    #     r"""Computes the projection onto constant set for each weight and bias.

    #     Args:
    #         weights (np.ndarray): (nb_weight, n_params_per_weight)
    #         bias (np.ndarray): (nb_weight)
    #         verbose (bool, optional): Shows progress bar. Defaults to True.

    #     Returns:
    #         np.ndarray: (nb_weight, n_params_per_weight) the closest constants et
    #         np.ndarray: (nb_weight) the closest operation (erosion or dilation)
    #         np.ndarray: (nb_weight) the distance to the closest activated space of constant set
    #     """
    #     W = weights
    #     w_values = np.zeros((W.shape[0], np.prod(W.shape[1:])))


    #     iterate = range(W.shape[0])
    #     if verbose:
    #         iterate = tqdm(iterate, leave=False, desc="Approximate binarization")

    #     for chout_idx, _ in enumerate(iterate):
    #         w_value_tmp = np.unique(W[chout_idx])[::-1]
    #         w_values[chout_idx, :len(w_value_tmp)] = w_value_tmp  # We assume that W don't repeat values. TODO: handle case with repeated values. Hint: add micro noise?

    #     best_idx = cls.find_best_index(w_values)
    #     S = (W >= w_values[np.arange(w_values.shape[0]), best_idx, None, None, None])

    #     best_dist_selem = cls.distance_fn_selem(weights=W, S=S,)

    #     wsum = W.reshape(W.shape[0], -1).sum(1)
    #     final_operation = np.empty(W.shape[0], dtype=str)
    #     final_operation[wsum / 2 >= -bias] = cls.operation_code["dilation"]
    #     final_operation[wsum / 2 < -bias] = cls.operation_code["erosion"]

    #     final_dist_bias = np.abs(-bias - wsum)
    #     final_dist = best_dist_selem + final_dist_bias

    #     return S, final_operation, final_dist


    def find_closest_selem_and_operation(
        self, weights, bias, chans=None, v1=0, v2=1, verbose: bool = True,
    ):
        if chans is None:
            chans = range(self.bise_module.out_channels)

        W = weights.cpu().detach().numpy()[chans]
        bias = bias.cpu().detach().numpy()[chans]

        S, final_operation, final_dist = ProjectionConstantSet.compute(W.reshape(W.shape[0], -1), bias, verbose=verbose)
        S = S.reshape(W.shape)
        return S, final_operation, final_dist

        # if verbose:
        #     chans = tqdm(chans, leave=False, desc="Approximate binarization")

        # w_values = np.zeros((len(chans), np.prod(W.shape[1:])))
        # for chout_idx, _ in enumerate(chans):
        #     w_value_tmp = np.unique(W[chout_idx])[::-1]
        #     w_values[chout_idx, :len(w_value_tmp)] = w_value_tmp  # We assume that W don't repeat values. TODO: handle case with repeated values. Hint: add micro noise?

        # best_idx = self.find_best_index(w_values)
        # S = (W >= w_values[np.arange(w_values.shape[0]), best_idx, None, None, None])

        # best_dist_selem = self.distance_fn_selem(weights=W, S=S,)

        # wsum = W.reshape(W.shape[0], -1).sum(1)
        # final_operation = np.empty(len(chans), dtype=str)
        # final_operation[wsum / 2 >= -bias] = self.bise_module.operation_code["dilation"]
        # final_operation[wsum / 2 < -bias] = self.bise_module.operation_code["erosion"]

        # final_dist_bias = np.abs(-bias - wsum)
        # final_dist = best_dist_selem + final_dist_bias

        # return S, final_operation, final_dist

    # def find_closest_selem(self, W, chans):
    #     w_values = np.zeros((len(chans), np.prod(W.shape[1:])))
    #     for chout_idx, _ in enumerate(chans):
    #         w_value_tmp = np.unique(W[chout_idx])[::-1]
    #         # if chout_idx == 920:
    #         #     return w_value_tmp  # DEBUG
    #         w_values[chout_idx, :len(w_value_tmp)] = w_value_tmp  # We assume that W don't repeat values. TODO: handle case with repeated values. Hint: add micro noise?

    #     best_idx = self.find_best_index(w_values)
    #     S = (W >= w_values[np.arange(w_values.shape[0]), best_idx, None, None, None])
    #     best_dist_selem = self.distance_fn_selem(weights=W, S=S,)
    #     return S, best_dist_selem

    # def find_operation(self, W, bias, chans):
    #     wsum = W.reshape(W.shape[0], -1).sum(1)
    #     final_operation = np.empty(len(chans), dtype=str)
    #     final_operation[wsum / 2 >= -bias] = self.bise_module.operation_code["dilation"]
    #     final_operation[wsum / 2 < -bias] = self.bise_module.operation_code["erosion"]
    #     # if -bias <= wsum / 2:
    #     #     final_operation = "dilation"
    #     # else:
    #     #     final_operation = "erosion"

    #     final_dist_bias = np.abs(-bias - wsum)
    #     return final_operation, final_dist_bias

    def __call__(self, chans: List[int], v1: float = 0, v2: float = 1, verbose: bool = True) -> Tuple[float, np.ndarray, str]:
        return self.find_closest_selem_and_operation(
            chans=chans, weights=self.bise_module.weights, bias=self.bise_module.bias, v1=v1, v2=v2, verbose=verbose,
        )
