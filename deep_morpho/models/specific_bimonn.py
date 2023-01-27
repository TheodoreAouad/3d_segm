from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np

from .bisel import BiSEL
from .binary_nn import BinaryNN
from .dense_lui import DenseLUI
from .layers_not_binary import DenseLuiNotBinary
from ..initializer import InitBiseEnum, BimonnInitializer, BiselInitializer


# TODO: Merge with Bimonn class
class BimonnDenseBase(BinaryNN):
    def __init__(
        self,
        last_layer: DenseLUI,
        channels: List[int],
        input_size: int,
        n_classes: int,
        initializer_method: InitBiseEnum = InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
        initializer_args: Dict = {"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": .5},
        input_mean: float = .5,
        *args,
        **kwargs
    ):
        super().__init__()

        self.input_size = input_size
        self.n_classes = n_classes
        self.channels = [input_size] + channels + [n_classes]

        self.initializer_method = initializer_method
        self.initializer_args = initializer_args
        self.input_mean = input_mean

        self.first_init_args = {k: v for k, v in initializer_args.items() if k != "input_mean"}
        self.first_init_args["input_mean"] = input_mean

        self.initializer_fn = BimonnInitializer.get_init_class(initializer_method, atomic_element="bisel")

        self.flatten = nn.Flatten()
        self.layers = []

        if len(self.channels) > 2:  # If len==2, then there is only one layer.
            self.dense1 = DenseLUI(
                in_channels=self.channels[0],
                out_channels=self.channels[1],
                initializer=self.initializer_fn(**self.first_init_args),
                **kwargs
            )
            self.layers.append(self.dense1)


        for idx, (chin, chout) in enumerate(zip(self.channels[1:-2], self.channels[2:-1]), start=2):
            setattr(self, f"dense{idx}", DenseLUI(
                in_channels=chin,
                out_channels=chout,
                initializer=self.initializer_fn(**initializer_args),
                **kwargs
            ))
            self.layers.append(getattr(self, f"dense{idx}"))

        self.classification_layer = last_layer(
            in_channels=self.channels[-2],
            out_channels=self.channels[-1],
            initializer=self.initializer_fn(**initializer_args),
            **kwargs
        )
        self.layers.append(self.classification_layer)


    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
        return x


class BimonnDense(BimonnDenseBase):
    def __init__(self, *args, **kwargs):
        super().__init__(last_layer=DenseLUI, *args, **kwargs)


class BimonnDenseNotBinary(BimonnDenseBase):
    def __init__(self, *args, **kwargs):
        super().__init__(last_layer=DenseLuiNotBinary, *args, **kwargs)


class BimonnBiselDenseBase(BinaryNN):
    def __init__(
        self,
        last_layer: DenseLUI,
        kernel_size: Tuple[int],
        channels: List[int],
        input_size: Tuple[int],
        n_classes: int,
        initializer_bise_method: InitBiseEnum = InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
        initializer_bise_args: Dict = {"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": .5},
        initializer_lui_method: InitBiseEnum = InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS_RANDOM_BIAS,
        initializer_lui_args: Dict = {"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": .5},
        input_mean: float = .5,
        *args,
        **kwargs
    ):
        super().__init__()

        self.input_size = np.array(input_size)
        self.n_classes = n_classes
        self.channels = [input_size[0]] + channels + [n_classes]

        self.initializer_bise_method = initializer_bise_method
        self.initializer_bise_args = initializer_bise_args
        self.initializer_lui_method = initializer_lui_method
        self.initializer_lui_args = initializer_lui_args
        self.input_mean = input_mean

        self.first_init_args = {k: v for k, v in self.initializer_bise_args.items() if k != "input_mean"}
        self.first_init_args["input_mean"] = input_mean

        self.initializer_bise_fn = BimonnInitializer.get_init_class(self.initializer_bise_method, atomic_element="bisel")
        self.initializer_lui_fn = BimonnInitializer.get_init_class(self.initializer_lui_method, atomic_element="bisel")

        self.layers = []

        self.bisel1 = BiSEL(
            in_channels=self.channels[0],
            out_channels=self.channels[1],
            kernel_size=kernel_size,
            initializer=BiselInitializer(
                bise_initializer=self.initializer_bise_fn(**self.first_init_args),
                lui_initializer=self.initializer_lui_fn(**self.initializer_lui_args),
            ),
            *args, **kwargs
        )
        self.layers.append(self.bisel1)

        self.maxpool1 = nn.MaxPool2d(2)
        self.layers.append(self.maxpool1)
        self.input_size[1:] = np.array(self.input_size[1:]) // 2

        for idx, (chin, chout) in enumerate(zip(self.channels[1:-3], self.channels[2:-2]), start=2):
            setattr(self, f"bisel{idx}", BiSEL(
                in_channels=chin,
                out_channels=chout,
                kernel_size=kernel_size,
                initializer=BiselInitializer(
                    bise_initializer=self.initializer_bise_fn(**self.initializer_bise_args),
                    lui_initializer=self.initializer_lui_fn(**self.initializer_lui_args),
                ),
                *args, **kwargs
            ))
            self.layers.append(getattr(self, f"bisel{idx}"))

            setattr(self, f"maxpool{idx}", nn.MaxPool2d(2))
            self.layers.append(getattr(self, f"maxpool{idx}"))
            self.input_size[1:] = np.array(self.input_size[1:]) // 2

        self.flatten = nn.Flatten()
        self.layers.append(self.flatten)

        self.dense1 = DenseLUI(
            in_channels=self.channels[-3] * np.prod(self.input_size[1:]),
            out_channels=self.channels[-2],
            initializer=self.initializer_bise_fn(**initializer_bise_args),
            **kwargs
        )
        self.layers.append(self.dense1)

        self.classification_layer = last_layer(
            in_channels=self.channels[-2],
            out_channels=self.channels[-1],
            initializer=self.initializer_bise_fn(**initializer_bise_args),
            **kwargs
        )

        self.layers.append(self.classification_layer)


    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class BimonnBiselDense(BimonnBiselDenseBase):
    def __init__(self, *args, **kwargs):
        super().__init__(last_layer=DenseLUI, *args, **kwargs)


class BimonnBiselDenseNotBinary(BimonnBiselDenseBase):
    def __init__(self, *args, **kwargs):
        super().__init__(last_layer=DenseLuiNotBinary, *args, **kwargs)
