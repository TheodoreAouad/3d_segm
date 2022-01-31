from typing import List, Tuple, Union, Dict

import torch.nn as nn
import numpy as np

from .bise import BiSE, BiSEC
from .dilation_sum_layer import MaxPlusAtom
from .cobise import COBiSE, COBiSEC
from .bisel import BiSEL


class BiMoNN(nn.Module):

    def __init__(
        self,
        kernel_size: List[Union[Tuple, int]],
        channels: List[int],
        atomic_element: Union[str, List[str]] = 'bise',
        weight_P: Union[float, List[float]] = 1,
        threshold_mode: Union[Union[str, dict], List[Union[str, dict]]] = "tanh",
        activation_P: Union[float, List[float]] = 10,
        constant_activation_P: Union[bool, List[bool]] = False,
        constant_weight_P: Union[bool, List[bool]] = False,
        init_bias_value: Union[float, List[float]] = -2,
        init_weight_identity: Union[bool, List[bool]] = True,
        alpha_init: Union[float, List[float]] = 0,
        init_value: Union[float, List[float]] = -2,
        share_weights: Union[bool, List[bool]] = True,
        constant_P_lui: Union[bool, List[bool]] = False,
    ):
        """
        Initialize the model.

        Args:
            self: write your description
            kernel_size: write your description
            Tuple: write your description
            channels: write your description
            atomic_element: write your description
            weight_P: write your description
            threshold_mode: write your description
            dict: write your description
            dict: write your description
            activation_P: write your description
            constant_activation_P: write your description
            constant_weight_P: write your description
            init_bias_value: write your description
            init_weight_identity: write your description
            alpha_init: write your description
            init_value: write your description
            share_weights: write your description
            constant_P_lui: write your description
        """
        super().__init__()
        self.kernel_size = self._init_kernel_size(kernel_size)

        # for attr in set(self.bises_args).union(self.bisecs_args).difference(['kernel_size']).union(['atomic_element']):
        for attr in set(self.all_args):
            setattr(self, attr, self._init_attr(attr, eval(attr)))

        self.layers = []
        self.bises_idx = []
        self.bisecs_idx = []
        self.bisels_idx = []
        for idx in range(len(self)):
            layer = self._make_layer(idx)
            self.layers.append(layer)
            setattr(self, f'layer{idx+1}', layer)

    @property
    def bises(self):
        """
        List of bises in the network.

        Args:
            self: write your description
        """
        return [self.layers[idx] for idx in self.bises_idx]

    @property
    def bisecs(self):
        """
        List of bisecting layers.

        Args:
            self: write your description
        """
        return [self.layers[idx] for idx in self.bisecs_idx]

    @property
    def bisels(self):
        """
        List of bisels in the network.

        Args:
            self: write your description
        """
        return [self.layers[idx] for idx in self.bisels_idx]

    def forward(self, x):
        """
        Apply the forward layers to the input.

        Args:
            self: write your description
            x: write your description
        """
        output = self.layers[0](x)
        for layer in self.layers[1:]:
            output = layer(output)
        return output

    def get_bise_selems(self) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
        """Go through all BiSE indexes and shows the learned selem and operation. If None are learned, puts value
        None.

        Returns:
            (dict, dict): the dictionary of learned selems and operations with indexes as keys.
        """
        selems = {}
        operations = {}
        v1, v2 = 0, 1
        for idx in self.bises_idx:
            if v1 is not None:
                selems[idx], operations[idx] = self.layers[idx].find_selem_and_operation(v1, v2)
                v1, v2 = self.layers[idx].get_outputs_bounds(v1, v2)
            else:
                selems[idx], operations[idx] = None, None
        return selems, operations

    def __len__(self):
        """
        Returns the length of the kernel.

        Args:
            self: write your description
        """
        return len(self.kernel_size)

    @staticmethod
    def _init_kernel_size(kernel_size: List[Union[Tuple, int]]):
        """
        Initialize the kernel size.

        Args:
            kernel_size: write your description
            List: write your description
            Union: write your description
            Tuple: write your description
            int: write your description
        """
        res = []
        for size in kernel_size:
            if isinstance(size, int):
                res.append((size, size))
            else:
                res.append(size)
        return res

    def _init_channels(self, channels: List[int]):
        """
        Initialize the internal list of channels.

        Args:
            self: write your description
            channels: write your description
        """
        self.out_channels = channels[1:]
        self.in_channels = channels[:-1]
        self.channels = channels
        return self.channels

    def _init_atomic_element(self, atomic_element: Union[str, List[str]]):
        """
        Initializes the atomic element.

        Args:
            self: write your description
            atomic_element: write your description
        """
        if isinstance(atomic_element, list):
            return [s.lower() for s in atomic_element]

        return [atomic_element.lower() for _ in range(len(self))]

    def _init_attr(self, attr_name, attr_value):
        """
        Initializes an attribute of the object.

        Args:
            self: write your description
            attr_name: write your description
            attr_value: write your description
        """
        if attr_name == "kernel_size":
            # return self._init_kernel_size(attr_value)
            return self.kernel_size

        if attr_name == "atomic_element":
            return self._init_atomic_element(attr_value)

        if attr_name == "channels":
            return self._init_channels(attr_value)

        if isinstance(attr_value, list):
            return attr_value

        return [attr_value for _ in range(len(self))]

    def bises_kwargs_idx(self, idx):
        """
        Returns a dict with the bises_kwargs for the specified index.

        Args:
            self: write your description
            idx: write your description
        """
        return dict(
            **{'shared_weights': None, 'shared_weight_P': None},
            **{k: getattr(self, k)[idx] for k in self.bises_args}
        )

    def bisels_kwargs_idx(self, idx):
        """
        Return kwargs for a bisels method at the given index.

        Args:
            self: write your description
            idx: write your description
        """
        return dict(
            **{'shared_weights': None, 'shared_weight_P': None},
            **{k: getattr(self, k)[idx] for k in self.bisels_args}
        )

    def bisecs_kwargs_idx(self, idx):
        """
        Return kwargs for bisecs method at index idx.

        Args:
            self: write your description
            idx: write your description
        """
        return dict(
            **{'shared_weights': None, 'shared_weight_P': None},
            **{k: getattr(self, k)[idx] for k in self.bisecs_args}
        )

    def cobise_kwargs_idx(self, idx):
        """
        Return a dict of kwargs for the specified index.

        Args:
            self: write your description
            idx: write your description
        """
        return dict(
            **{k: getattr(self, k)[idx] for k in self.cobise_args},
        )

    def cobisec_kwargs_idx(self, idx):
        """
        Return kwargs for a specific index

        Args:
            self: write your description
            idx: write your description
        """
        return dict(
            **{k: getattr(self, k)[idx] for k in self.cobisec_args},
        )

    def max_plus_kwargs_idx(self, idx):
        """
        Return max_plus_kwargs for the given index.

        Args:
            self: write your description
            idx: write your description
        """
        return dict(
            **{k: getattr(self, k)[idx] for k in self.max_plus_args},
        )

    @property
    def bises_args(self):
        """
        Returns a list of arguments for the Bises command.

        Args:
            self: write your description
        """
        return [
            'kernel_size', 'weight_P', 'threshold_mode', 'activation_P',
            'init_bias_value', 'init_weight_identity', 'out_channels', "constant_activation_P",
            "constant_weight_P"
        ]

    @property
    def bisels_args(self):
        """
        List of arguments for the bises command.

        Args:
            self: write your description
        """
        return self.bises_args + ['in_channels', 'constant_P_lui']

    @property
    def bisecs_args(self):
        """
        All bises command line arguments.

        Args:
            self: write your description
        """
        return set(self.bises_args).difference(['init_bias_value']).union(['alpha_init'])

    @property
    def max_plus_args(self):
        """
        Return maximum number of plus arguments.

        Args:
            self: write your description
        """
        return ['kernel_size', 'alpha_init', 'init_value', 'threshold_mode']

    @property
    def cobise_args(self):
        """
        Return the cobise arguments for the command.

        Args:
            self: write your description
        """
        return set(self.bises_args).union(['share_weights'])

    @property
    def cobisec_args(self):
        """
        The arguments to pass to the cobisec_command.

        Args:
            self: write your description
        """
        return set(self.bisecs_args).union(['share_weights'])

    def _make_layer(self, idx):
        """
        Create a layer of the appropriate type based on the atomic element of the element at the given index

        Args:
            self: write your description
            idx: write your description
        """
        if self.atomic_element[idx] == 'bise':
            layer = BiSE(**self.bises_kwargs_idx(idx))
            self.bises_idx.append(idx)

        elif self.atomic_element[idx] == 'bisel':
            layer = BiSEL(**self.bisels_kwargs_idx(idx))
            self.bisels_idx.append(idx)

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
        """
        Returns a list of all the arguments that are passed to the function.

        Args:
            self: write your description
        """
        return [
            "kernel_size", "atomic_element", "weight_P", "threshold_mode", "activation_P", "constant_activation_P",
            "init_bias_value", "init_weight_identity", "alpha_init", "init_value", "share_weights",
            "constant_weight_P", "constant_P_lui", "channels",
        ]
