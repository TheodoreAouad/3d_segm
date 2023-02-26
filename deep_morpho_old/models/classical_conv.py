from typing import Tuple, List, Callable
import torch
import torch.nn as nn
import numpy as np

from .binary_nn import BinaryNN


class ConvNetLastLinear(BinaryNN):

    def __init__(
        self,
        kernel_size: int,
        channels: List[int],
        n_classes: int,
        input_size: Tuple[int, int, int],
        do_maxpool: bool = False,
        activation_constructor: Callable = nn.ReLU,
        **kwargs
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = [input_size[0]] + channels + [n_classes]
        self.n_classes = n_classes
        self.input_size = np.array(input_size)
        self.input_dense = np.array(input_size)
        self.do_maxpool = do_maxpool
        self.activation_constructor = activation_constructor

        conv_layers = []
        for (chin, chout) in zip(self.channels[:-3], self.channels[1:-2]):
            conv_layers.append(nn.Conv2d(chin, chout, kernel_size, padding="same",))
            conv_layers.append(activation_constructor())
            if do_maxpool:
                conv_layers.append(nn.MaxPool2d(2))
                self.input_dense[1:] = self.input_dense[1:] // 2

        self.conv_layers = nn.Sequential(*conv_layers)

        # self.conv_layers = nn.Sequential(*
        #     [nn.Conv2d(self.channels[i], self.channels[i + 1], kernel_size, padding="same", **kwargs) for i in range(len(self.channels) - 2)]
        # )
        self.flatten = nn.Flatten()

        linears = []

        linears.append(nn.Linear(in_features=self.channels[-3] * np.prod(self.input_dense[1:]), out_features=self.channels[-2]))
        linears.append(activation_constructor())
        linears.append(nn.Linear(in_features=self.channels[-2], out_features=self.channels[-1]))

        self.linear_layers = nn.Sequential(*linears)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)

        return x
