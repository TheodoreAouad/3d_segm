from typing import List, Tuple, Union

import torch.nn as nn

from .bise import BiSE, LogicalNotBiSE


class BiMoNN(nn.Module):

    def __init__(
        self,
        kernel_size: List[Union[Tuple, int]],
        logical_not: Union[bool, List[bool]] = False,
        weight_P: Union[float, List[float]] = 1,
        weight_threshold_mode: Union[str, List[str]] = "sigmoid",
        activation_P: Union[float, List[float]] = 10,
        activation_threshold_mode: Union[str, List[str]] = "sigmoid",
        init_bias_value: Union[float, List[float]] = -2,
        init_weight_identity: Union[bool, List[bool]] = True,
        out_channels: Union[int, List[int]] = 1,
        alpha_init: Union[float, List[float]] = 0,
    ):
        super().__init__()
        self.kernel_size = self._init_kernel_size(kernel_size)

        for attr in set(self.bises_args).union(self.logical_not_bises_args).difference(['kernel_size']).union(['logical_not']):
            setattr(self, attr, self._init_attr(eval(attr)))

        self.bises = []
        for idx in range(len(self)):
            if self.logical_not[idx]:
                layer = LogicalNotBiSE(**self.logical_not_bises_kwargs_idx(idx))
            else:
                layer = BiSE(**self.bises_kwargs_idx(idx))
            setattr(self, f'layer{idx+1}', layer)
            self.bises.append(layer)

    def forward(self, x):
        output = self.bises[0](x)
        for bise_layer in self.bises[1:]:
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

    def _init_attr(self, attr):
        if isinstance(attr, list):
            return attr

        return [attr for _ in range(len(self))]

    def bises_kwargs_idx(self, idx):
        return dict(
            **{'shared_weights': None, 'shared_weight_P': None},
            **{k: getattr(self, k)[idx] for k in self.bises_args}
        )

    def logical_not_bises_kwargs_idx(self, idx):
        return dict(
            **{'shared_weights': None, 'shared_weight_P': None},
            **{k: getattr(self, k)[idx] for k in self.logical_not_bises_args},
        )

    @property
    def bises_args(self):
        return [
            'kernel_size', 'weight_P', 'weight_threshold_mode', 'activation_P',
            'activation_threshold_mode', 'init_bias_value', 'init_weight_identity', 'out_channels'
        ]

    @property
    def logical_not_bises_args(self):
        return set(self.bises_args).difference(['init_bias_value']).union(['alpha_init'])

    def _make_layer(self, bises_kwargs):
        return BiSE(**bises_kwargs)
