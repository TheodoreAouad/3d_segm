from typing import Tuple, TYPE_CHECKING
from enum import Enum
from functools import partial

import numpy as np

if TYPE_CHECKING:
    from deep_morpho.bise import BiSE
    import torch


from general.nn.experiments.experiment_methods import ExperimentMethods



class ClosestSelemEnum(Enum):
    MIN_DIST = 1
    MAX_SECOND_DERIVATIVE = 2
    MIN_DIST_DIST_TO_BOUNDS = 3
    MIN_DIST_DIST_TO_CST = 4


class ClosestSelemDistanceEnum(Enum):
    DISTANCE_TO_BOUNDS = 1
    DISTANCE_BETWEEN_BOUNDS = 2
    DISTANCE_TO_AND_BETWEEN_BOUNDS = 3


class BiseClosestSelemHandler(ExperimentMethods):

    def __init__(self, bise_module: "BiSE"):
        self.bise_module = bise_module

    def __call__(self, chin: int, chout: int, v1: float, v2: float) -> Tuple[float, np.ndarray, str]:
        pass


class BiseClosestSelemWithDistanceAgg(BiseClosestSelemHandler):
    """ Children must define:
    distance_fn(self, normalized_weights: "torch.Tensor", bias: "torch.Tensor", operation: str, S: np.ndarray, v1: float, v2: float) -> float
        given an operation, weights, bias, selems and almost binary bounds, outputs a similarity measure float
    distance_agg_fn(self, selems: np.ndarray, dists: np.ndarray) -> Tuple[np.ndarray, float]
        given a list of selems and dists in the form of arrays, outputs the best selem given the distances, and its corresponding distance
    """

    def __init__(self, distance_fn, distance_agg_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance_fn = partial(distance_fn, self=self)
        self.distance_agg_fn = partial(distance_agg_fn, self=self)

    def compute_selem_dist_for_operation(
        self, weights, bias, operation: str, v1: float = 0, v2: float = 1
    ):
        weight_values = weights.unique().detach().cpu().numpy()

        dists = np.zeros_like(weight_values)
        selems = []
        for value_idx, value in enumerate(weight_values):
            selem = (weights >= value).cpu().detach().numpy()
            dists[value_idx] = self.distance_fn(normalized_weights=weights, bias=bias, operation=operation, S=selem, v1=v1, v2=v2)
            selems.append(selem)

        return selems, dists


    def find_closest_selem_and_operation_chan(
        self, weights, bias, chin=0, chout=0, v1=0, v2=1
    ):
        final_dist = np.infty
        for operation in ['dilation', 'erosion']:
            selems, dists = self.compute_selem_dist_for_operation(
                weights=weights[chout, chin], bias=bias[chout, chin], operation=operation, v1=v1, v2=v2
            )
            new_selem, new_dist = self.distance_agg_fn(selems=selems, dists=dists)
            if new_dist < final_dist:
                final_dist = new_dist
                final_selem = new_selem
                final_operation = operation

        return final_dist, final_selem, final_operation

    def __call__(self, chin: int, chout: int, v1: float, v2: float) -> Tuple[float, np.ndarray, str]:
        return self.find_closest_selem_and_operation_chan(
            weights=self.bise_module._normalized_weights, bias=self.bise_module.bias, chin=chin, chout=chout, v1=v1, v2=v2
        )


def distance_agg_min(self, selems: np.ndarray, dists: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx_min = dists.argmin()
    return selems[idx_min], dists[idx_min]


def distance_agg_max_second_derivative(self, selems: np.ndarray, dists: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx_min = (dists[2:] + dists[:-2] - 2 * dists[1:-1]).argmax() + 1
    return selems[idx_min], dists[idx_min]


def distance_to_bounds_base(bound_fn, normalized_weights: "torch.Tensor", bias: "torch.Tensor", S: np.ndarray, v1: float = 0, v2: float = 1) -> float:
    assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
    lb, ub = bound_fn(normalized_weights=normalized_weights, S=S, v1=v1, v2=v2)
    dist_lb = lb + bias  # if dist_lb < 0 : lower bound respected
    dist_ub = -bias - ub  # if dist_ub < 0 : upper bound respected
    return max(dist_lb, dist_ub, 0)


def distance_fn_to_bounds(self, normalized_weights: "torch.Tensor", bias: "torch.Tensor", operation: str, S: np.ndarray, v1: float, v2: float) -> float:
    if operation == "dilation":
        bound_fn = self.bise_module.bias_bounds_dilation
    elif operation == "erosion":
        bound_fn = self.bise_module.bias_bounds_erosion
    else:
        raise SyntaxError("operation must be either 'dilation' or 'erosion'.")

    return distance_to_bounds_base(bound_fn, normalized_weights=normalized_weights, bias=bias, S=S, v1=v1, v2=v2)


def distance_between_bounds_base(bound_fn, normalized_weights: "torch.Tensor", bias: "torch.Tensor", S: np.ndarray, v1: float = 0, v2: float = 1) -> float:
    assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
    lb, ub = bound_fn(normalized_weights=normalized_weights, S=S, v1=v1, v2=v2)
    return lb - ub


def distance_fn_between_bounds(self, normalized_weights: "torch.Tensor", bias: "torch.Tensor", operation: str, S: np.ndarray, v1: float, v2: float) -> float:
    if operation == "dilation":
        bound_fn = self.bise_module.bias_bounds_dilation
    elif operation == "erosion":
        bound_fn = self.bise_module.bias_bounds_erosion
    else:
        raise SyntaxError("operation must be either 'dilation' or 'erosion'.")

    return distance_between_bounds_base(bound_fn, normalized_weights=normalized_weights, bias=bias, S=S, v1=v1, v2=v2)


class BiseClosestMinDistBounds(BiseClosestSelemWithDistanceAgg):

    def __init__(self, *args, **kwargs):
        super().__init__(distance_fn=distance_fn_to_bounds, distance_agg_fn=distance_agg_min, *args, **kwargs)


class BiseClosestMinDistOnCst(BiseClosestSelemWithDistanceAgg):

    def __init__(self, *args, **kwargs):
        super().__init__(distance_fn=self.distance_fn_selem, distance_agg_fn=distance_agg_min, *args, **kwargs)
        self.distance_fn_bias = partial(distance_fn_between_bounds, self=self)

    @staticmethod
    def distance_fn_selem(self, normalized_weights: "torch.Tensor", bias: "torch.Tensor", operation: str, S: np.ndarray, v1: float, v2: float) -> float:
        W = normalized_weights.cpu().detach().numpy()
        return (W ** 2).sum() - 1 / S.sum() * (W[S].sum()) ** 2

    def find_closest_selem_and_operation_chan(
        self, weights, bias, chin=0, chout=0, v1=0, v2=1
    ):
        weights = weights[chout, chin]
        bias = bias[chout, chin]

        final_dist_selem = np.infty
        final_dist_bias = np.infty

        selems, dists = self.compute_selem_dist_for_operation(
            weights=weights, bias=bias, operation=None, v1=v1, v2=v2
        )
        new_selem, new_dist = self.distance_agg_fn(selems=selems, dists=dists)
        if new_dist < final_dist_selem:
            final_dist_selem = new_dist
            final_selem = new_selem

        wsum = weights.sum()
        if -bias <= wsum / 2:
            final_operation = "dilation"
        else:
            final_operation = "erosion"

        final_dist_bias = (-bias - wsum).abs().cpu().detach().numpy()
        final_dist = final_dist_selem + final_dist_bias

        return final_dist, final_selem, final_operation
