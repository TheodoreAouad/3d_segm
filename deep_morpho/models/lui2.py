from typing import Union, Dict, Tuple

import torch
import numpy as np

from .bise_base import BiSEBase, InitBiseEnum, ClosestSelemDistanceEnum, ClosestSelemEnum, BiseBiasOptimEnum


class LUI(BiSEBase):

    def __init__(
        self,
        threshold_mode: Union[Dict[str, str], str] = {"weight": "softplus", "activation": "tanh"},
        activation_P: float = 10,
        constant_activation_P: bool = False,
        shared_weights: torch.tensor = None,
        init_bias_value: float = 1,
        input_mean: float = 0.5,
        init_weight_mode: InitBiseEnum = InitBiseEnum.CUSTOM_HEURISTIC,
        out_channels: int = 1,
        in_channels: int = 1,
        closest_selem_method: ClosestSelemEnum = ClosestSelemEnum.MIN_DIST,
        closest_selem_distance_fn: ClosestSelemDistanceEnum = ClosestSelemDistanceEnum.DISTANCE_TO_AND_BETWEEN_BOUNDS,
        bias_optim_mode: BiseBiasOptimEnum = BiseBiasOptimEnum.POSITIVE_INTERVAL_REPARAMETRIZED,
        force_identity: bool = False,
        *args,
        **kwargs
    ):
        self.force_identity = force_identity or (in_channels == out_channels == 1)
        if self.force_identity:
            bias_optim_mode = BiseBiasOptimEnum.RAW
        super().__init__(
            kernel_size=(1, 1),
            threshold_mode=threshold_mode,
            activation_P=activation_P,
            constant_activation_P=constant_activation_P,
            shared_weights=shared_weights,
            init_bias_value=init_bias_value,
            input_mean=input_mean,
            init_weight_mode=init_weight_mode,
            out_channels=out_channels,
            in_channels=in_channels,
            do_mask_output=False,
            closest_selem_method=closest_selem_method,
            closest_selem_distance_fn=closest_selem_distance_fn,
            bias_optim_mode=bias_optim_mode,
            padding=0,
            *args,
            **kwargs
        )

    def forward(self, x):
        if self.force_identity:
            return x
        return super().forward(x)

    @property
    def activation_P(self):
        if self.force_identity:
            return torch.ones_like(self.activation_threshold_layer.P_, requires_grad=False)
        return super().activation_P

    @property
    def bias(self):
        if self.force_identity:
            return torch.zeros_like(super().bias, requires_grad=False)
        return super().bias

    def init_weights(self):
        if self.force_identity:
            return
        return super().init_weights()

    def init_bias(self):
        if self.force_identity:
            return
        return super().init_bias()

    @property
    def coefs(self):
        return self._normalized_weight[..., 0, 0]

    def update_binary_sets(self, *args, **kwargs):
        return self.update_binary_selems(*args, **kwargs)

    def find_set_and_operation_chan(self, *args, **kwargs):
        return self.find_selem_and_operation_chan(*args, **kwargs)

    def find_closest_set_and_operation_chan(self, *args, **kwargs):
        return self.find_closest_selem_and_operation_chan(*args, **kwargs)

    # def find_selem_and_operation_chan(self, chin: int = 0, v1: float = 0, v2: float = 1):
    #     return super().find_selem_and_operation_chan(chout=0, chin=chin, v1=v1, v2=v2)

    # def find_closest_selem_and_operation_chan(self, chin: int = 0, v1: float = 0, v2: float = 1) -> Tuple[np.ndarray, str, float]:
    #     return super().find_closest_selem_and_operation_chan(chin=chin, chout=0, v1=v1, v2=v2)

    @property
    def closest_set(self):
        return self._closest_selem[0]

    @property
    def learned_selem(self):
        return self._learned_selem[0]
