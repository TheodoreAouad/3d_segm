from typing import Tuple, Union, Dict

import torch
import numpy as np

from .bise_base import BiSEBase, InitBiseEnum, ClosestSelemDistanceEnum, ClosestSelemEnum, BiseBiasOptimEnum


class BiSE(BiSEBase):
    """Given the BiSEL implementation, the BiSE always has an input chan of 1.
    """

    def __init__(
        self,
        kernel_size: Tuple,
        threshold_mode: Union[Dict[str, str], str] = {"weight": "softplus", "activation": "tanh"},
        activation_P: float = 10,
        constant_activation_P: bool = False,
        shared_weights: torch.tensor = None,
        init_bias_value: float = 1,
        input_mean: float = 0.5,
        init_weight_mode: InitBiseEnum = InitBiseEnum.CUSTOM_HEURISTIC,
        out_channels: int = 1,
        do_mask_output: bool = False,
        closest_selem_method: ClosestSelemEnum = ClosestSelemEnum.MIN_DIST,
        closest_selem_distance_fn: ClosestSelemDistanceEnum = ClosestSelemDistanceEnum.DISTANCE_TO_AND_BETWEEN_BOUNDS,
        bias_optim_mode: BiseBiasOptimEnum = BiseBiasOptimEnum.POSITIVE_INTERVAL_REPARAMETRIZED,
        padding=None,
        padding_mode: str = "replicate",
        *args,
        **kwargs
    ):
        super().__init__(
            kernel_size=kernel_size,
            threshold_mode=threshold_mode,
            activation_P=activation_P,
            constant_activation_P=constant_activation_P,
            shared_weights=shared_weights,
            init_bias_value=init_bias_value,
            input_mean=input_mean,
            init_weight_mode=init_weight_mode,
            out_channels=out_channels,
            in_channels=1,
            do_mask_output=do_mask_output,
            closest_selem_method=closest_selem_method,
            closest_selem_distance_fn=closest_selem_distance_fn,
            bias_optim_mode=bias_optim_mode,
            padding=padding,
            padding_mode=padding_mode,
            *args,
            **kwargs
        )

    @property
    def closest_selem(self):
        return self._closest_selem[:, 0, ...]

    @property
    def learned_selem(self):
        return self._learned_selem[:, 0, ...]
