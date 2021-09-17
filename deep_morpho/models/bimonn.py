from typing import List, Tuple, Union

import torch.nn as nn

from .bise import BiSE, BiSEC
from .dilation_sum_layer import MaxPlusAtom
from .cobise import COBiSE, COBiSEC


class BiMoNN(nn.Module):

    def __init__(
        self,
        kernel_size: List[Union[Tuple, int]],
        atomic_element: Union[str, List[str]] = 'bise',
        weight_P: Union[float, List[float]] = 1,
        threshold_mode: Union[Union[str, dict], List[Union[str, dict]]] = "tanh",
        activation_P: Union[float, List[float]] = 10,
        init_bias_value: Union[float, List[float]] = -2,
        init_weight_identity: Union[bool, List[bool]] = True,
        out_channels: Union[int, List[int]] = 1,
        alpha_init: Union[float, List[float]] = 0,
        init_value: Union[float, List[float]] = -10,
        share_weights: Union[bool, List[bool]] = True,
    ):
        super().__init__()
        self.kernel_size = self._init_kernel_size(kernel_size)

        # for attr in set(self.bises_args).union(self.bisecs_args).difference(['kernel_size']).union(['atomic_element']):
        for attr in set(self.all_args):
            setattr(self, attr, self._init_attr(attr, eval(attr)))

        self.layers = []
        self.bises_idx = []
        self.bisecs_idx = []
        for idx in range(len(self)):
            layer = self._make_layer(idx)
            self.layers.append(layer)
            setattr(self, f'layer{idx+1}', layer)

    @property
    def bises(self):
        return [self.layers[idx] for idx in self.bises_idx]

    @property
    def bisecs(self):
        return [self.layers[idx] for idx in self.bisecs_idx]

    def forward(self, x):
        output = self.layers[0](x)
        for bise_layer in self.layers[1:]:
            output = bise_layer(output)
        return output


    def __len__(self):
        return len(self.kernel_size)

    @staticmethod
    def _init_kernel_size(kernel_size: List[Union[Tuple, int]]):
        res = []
        for size in kernel_size:
            if isinstance(size, int):
                res.append((size, size))
            else:
                res.append(size)
        return res

    def _init_atomic_element(self, atomic_element: Union[str, List[str]]):
        if isinstance(atomic_element, list):
            return [s.lower() for s in atomic_element]

        return [atomic_element.lower() for _ in range(len(self))]

    def _init_attr(self, attr_name, attr_value):
        if attr_name == "kernel_size":
            return self._init_kernel_size(attr_value)

        if attr_name == "atomic_element":
            return self._init_atomic_element(attr_value)

        if isinstance(attr_value, list):
            return attr_value

        return [attr_value for _ in range(len(self))]

    def bises_kwargs_idx(self, idx):
        return dict(
            **{'shared_weights': None, 'shared_weight_P': None},
            **{k: getattr(self, k)[idx] for k in self.bises_args}
        )

    def bisecs_kwargs_idx(self, idx):
        return dict(
            **{'shared_weights': None, 'shared_weight_P': None},
            **{k: getattr(self, k)[idx] for k in self.bisecs_args}
        )

    def cobise_kwargs_idx(self, idx):
        return dict(
            **{k: getattr(self, k)[idx] for k in self.cobise_args},
        )

    def cobisec_kwargs_idx(self, idx):
        return dict(
            **{k: getattr(self, k)[idx] for k in self.cobisec_args},
        )

    def max_plus_kwargs_idx(self, idx):
        return dict(
            **{k: getattr(self, k)[idx] for k in self.max_plus_args},
        )

    @property
    def bises_args(self):
        return [
            'kernel_size', 'weight_P', 'threshold_mode', 'activation_P',
            'init_bias_value', 'init_weight_identity', 'out_channels'
        ]

    @property
    def bisecs_args(self):
        return set(self.bises_args).difference(['init_bias_value']).union(['alpha_init'])

    @property
    def max_plus_args(self):
        return ['kernel_size', 'alpha_init', 'init_value', 'threshold_mode']

    @property
    def cobise_args(self):
        return set(self.bises_args).union(['share_weights'])

    @property
    def cobisec_args(self):
        return set(self.bisecs_args).union(['share_weights'])

    def _make_layer(self, idx):
        if self.atomic_element[idx] == 'bise':
            layer = BiSE(**self.bises_kwargs_idx(idx))
            self.bises_idx.append(idx)

        elif self.atomic_element[idx] == 'bisec':
            layer = BiSEC(**self.bisecs_kwargs_idx(idx))
            self.bisecs_idx.append(idx)

        elif self.atomic_element[idx] == 'max_plus':
            layer = MaxPlusAtom(**self.max_plus_kwargs_idx(idx))

        elif self.atomic_element[idx] == 'cobise':
            layer = COBiSE(**self.cobise_kwargs_idx(idx))

        elif self.atomic_element[idx] == 'cobisec':
            layer = COBiSEC(**self.cobisec_kwargs_idx(idx))

        return layer

    @property
    def all_args(self):
        return [
            "kernel_size", "atomic_element", "weight_P", "threshold_mode", "activation_P", "init_bias_value",
            "init_weight_identity", "out_channels", "alpha_init", "init_value", "share_weights",
        ]
