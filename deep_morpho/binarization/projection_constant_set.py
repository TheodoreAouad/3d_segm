from typing import Union
from types import ModuleType

import numpy as np
import torch
from tqdm import tqdm


class ProjectionConstantSet:
    operation_code = {"erosion": 0, "dilation": 1}  # TODO: be coherent with bisebase attribute operation_code

    def __init__(self, weights: Union[np.ndarray, torch.tensor], bias: Union[np.ndarray, torch.tensor],) -> None:
        self.weights = weights
        self.bias = bias

        self.final_dist_weight = None
        self.final_dist_bias = None
        self.final_operation = None
        self.S = None

    @staticmethod
    def distance_fn_selem(weights: Union[np.ndarray, torch.tensor], S: Union[np.ndarray, torch.tensor], module_: ModuleType = np) -> float:
        """
        Args:
            weights (Union[np.ndarray, torch.tensor]): shape (nb_weights, nb_params_per_weight)
            S (Union[np.ndarray, torch.tensor]): shape (nb_weights, nb_params_per_weight)
            module_ (ModuleType, optional): Module to work with. Defaults to np.

        Returns:
            float: _description_
        """
        return (weights ** 2).sum(1) - 1 / S.sum(1) * (module_.where(S, weights, 0).sum(axis=1)) ** 2

    @staticmethod
    def find_best_index(w_values: Union[np.ndarray, torch.tensor], module_: ModuleType = np) -> Union[np.ndarray, torch.tensor]:
        r"""Gives the arg maximum of the distance function $\frac{\sum_{k \in S}{W_k}}{\sqrt{card{S}}} = \frac{\sum_{k = 1}^j{w_k}}{\sqrt{j}}$
        with $w_k$ the sorted values in descending order of the weights $W$.

        Args:
            w_values (np.ndarray): (n_chout, prod(kernel_size))

        Returns:
            array (n_chout): the index of the best value, for each channel.
        """
        return (module_.cumsum(w_values, axis=1) / module_.sqrt(module_.arange(1, 1+w_values.shape[1]))).argmax(1)


    def compute(self, verbose: bool = True,) -> (Union[np.ndarray, torch.tensor], Union[np.ndarray, torch.tensor], Union[np.ndarray, torch.tensor]):
        r"""Computes the projection onto constant set for each weight and bias.

        Args:
            weights (Union[np.ndarray, torch.tensor]): (nb_weight, n_params_per_weight)
            bias (Union[np.ndarray, torch.tensor]): (nb_weight)
            verbose (bool, optional): Shows progress bar. Defaults to True.

        Returns:
            Union[np.ndarray, torch.tensor]: (nb_weight, n_params_per_weight) the closest constants et
            Union[np.ndarray, torch.tensor]: (nb_weight) the closest operation (erosion or dilation)
            Union[np.ndarray, torch.tensor]: (nb_weight) the distance to the closest activated space of constant set
        """
        W = self.weights
        bias = self.bias
        if isinstance(W, torch.Tensor):
            module_ = torch

        else:
            module_ = np

        w_values = module_.zeros_like(W)

        iterate = range(W.shape[0])
        if verbose:
            iterate = tqdm(iterate, leave=False, desc="Approximate binarization")

        for chout_idx, _ in enumerate(iterate):
            w_value_tmp = self.flip(module_.unique(W[chout_idx]))
            w_values[chout_idx, :len(w_value_tmp)] = w_value_tmp  # We assume that W don't repeat values. TODO: handle case with repeated values. Hint: add micro noise?

        best_idx = self.find_best_index(w_values, module_=module_)
        self.S = (W >= w_values[module_.arange(w_values.shape[0]), best_idx, None])

        self.final_dist_weight = self.distance_fn_selem(weights=W, S=self.S, module_=module_)

        wsum = W.sum(1)
        self.final_operation = np.empty(W.shape[0], dtype=str)
        dilation_idx = np.array(wsum / 2 >= -bias)
        if dilation_idx.any():
            self.final_operation[dilation_idx] = self.operation_code["dilation"]
        if (~dilation_idx).any():
            self.final_operation[~dilation_idx] = self.operation_code["erosion"]

        self.final_dist_bias = module_.abs(-bias - wsum)
        self.final_dist = self.final_dist_weight + self.final_dist_bias

        return self

    @staticmethod
    def flip(weights: Union[np.ndarray, torch.tensor]) -> Union[np.ndarray, torch.tensor]:
        if isinstance(weights, torch.Tensor):
            return weights.flip(-1)
        else:
            return weights[..., ::-1]
