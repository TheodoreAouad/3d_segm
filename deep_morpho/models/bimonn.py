from abc import ABC
from typing import List, Tuple, Union, Dict, Optional, Any
import inspect

import numpy as np
import torch.nn as nn

from .bisel import BiSEL, SyBiSEL
from .layers_not_binary import BiSELNotBinary 
from .binary_nn import BinaryNN
from ..initializer import BimonnInitializer, InitBimonnEnum, BimonnInitInputMean, InitBiseEnum


class BiMoNN(BinaryNN):

    def __init__(
        self,
        kernel_size: List[Union[Tuple, int]],
        channels: List[int],
        atomic_element: Union[str, List[str]] = 'sybisel',
        initializer_method: InitBimonnEnum = InitBimonnEnum.INPUT_MEAN,
        initializer_args: Dict = None,
        **kwargs,
    ):
        super().__init__()
        self.length = max([len(value) if isinstance(value, list) else 0 for value in [kernel_size, atomic_element] + list(kwargs.values())])
        self.length = max(len(channels) - 1, self.length)
        self.kernel_size = self._init_kernel_size(kernel_size)
        self.atomic_element = self._init_atomic_element(atomic_element)

        self.initializer_method = initializer_method
        self.initializer_args = initializer_args if initializer_args is not None else self._default_init_args()
        self.initalizer = self.create_initializer(**self.initializer_args)

        kwargs['channels'] = channels

        for attr, value in kwargs.items():
            setattr(self, attr, self._init_attr(attr, value))

        self.bisel_initializers = self.initalizer.generate_bisel_initializers(self)

        self.layers = []
        self.bises_idx = []
        self.bisecs_idx = []
        self.bisels_idx = []
        for idx in range(len(self)):
            layer = self._make_layer(idx)
            self.layers.append(layer)
            setattr(self, f'layer{idx+1}', layer)

    def _default_init_args(self):
        if self.atomic_element[0] == "bisel":
            return {
                "input_mean": 0.5,
                "bise_init_method": InitBiseEnum.CUSTOM_HEURISTIC,
                "bise_init_args": {"init_bias_value": 1},
            }

        elif self.atomic_element[0] == "sybisel":
            return {
                "input_mean": 0,
                "bise_init_method": InitBiseEnum.CUSTOM_CONSTANT,
                "bise_init_args": {"mean_weight": "auto"},
            }

    def create_initializer(self, **kwargs):
        if self.initializer_method == InitBimonnEnum.IDENTICAL:
            return BimonnInitializer(**kwargs)

        elif self.initializer_method == InitBimonnEnum.INPUT_MEAN:
            return BimonnInitInputMean(**kwargs)

        raise NotImplementedError("Initializer not recognized.")

    @property
    def bises(self):
        return [self.layers[idx] for idx in self.bises_idx]

    @property
    def bisecs(self):
        return [self.layers[idx] for idx in self.bisecs_idx]

    @property
    def bisels(self):
        return [self.layers[idx] for idx in self.bisels_idx]

    def forward(self, x):
        output = self.layers[0](x)
        for layer in self.layers[1:]:
            output = layer(output)
        return output

    def forward_save(self, x):
        """ Saves all intermediary outputs.
        Args:
            x (torch.Tensor): input images of size (batch_size, channel, width, height)

        Returns:
            list: list of dict. list[layer][output_lui_channel], list[layer][output_bisel_inchan, output_bisel_outchan]
        """
        output = {"input": x}
        cur = self.layers[0].forward_save(x)
        output[0] = cur
        for layer_idx, layer in enumerate(self.layers[1:], start=1):
            cur = layer.forward_save(cur['output'])
            output[layer_idx] = cur
        output["output"] = cur["output"]
        return output

    # deprecated
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
        return self.length
        # return len(self.atomic_element)
        # return len(self.layers)

    def _init_kernel_size(self, kernel_size: List[Union[Tuple, int]]):
        if isinstance(kernel_size, list):
            res = []
            for size in kernel_size:
                if isinstance(size, int):
                    res.append((size, size))
                else:
                    res.append(size)
            return res

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        return [kernel_size for _ in range(len(self))]


    def _init_channels(self, channels: List[int]):
        self.out_channels = channels[1:]
        self.in_channels = channels[:-1]
        self.channels = channels
        return self.channels

    def _init_input_mean(self, input_mean: Union[float, List[float]]):
        if isinstance(input_mean, list):
            return input_mean
        return [input_mean] + [0 if self.atomic_element[idx - 1] == "sybisel" else .5 for idx in range(1, len(self))]

    def _init_atomic_element(self, atomic_element: Union[str, List[str]]):
        if isinstance(atomic_element, list):
            return [s.lower() for s in atomic_element]

        return [atomic_element.lower() for _ in range(len(self))]

    def _init_attr(self, attr_name, attr_value):
        if attr_name == "kernel_size":
            # return self._init_kernel_size(attr_value)
            return self.kernel_size

        if attr_name == "atomic_element":
            return self._init_atomic_element(attr_value)

        if attr_name == "channels":
            return self._init_channels(attr_value)

        if attr_name == "input_mean":
            return self._init_input_mean(attr_value)

        if isinstance(attr_value, list):
            return attr_value

        return [attr_value for _ in range(len(self))]
        # return [attr_value for _ in range(len(self.kernel_size))]

    def is_not_default(self, key: str) -> bool:
        return key in self.__dict__.keys()

    def bisels_kwargs_idx(self, idx):
        return dict(
            **{'shared_weights': None, },
            **{k: getattr(self, k)[idx] for k in self.bisels_args if self.is_not_default(k)}
        )

    @property
    def bises_args(self):
        # return [
        #     'kernel_size', 'weight_P', 'threshold_mode', 'activation_P',
        #     'out_channels', "constant_activation_P",
        #     "constant_weight_P",
        #     "closest_selem_method", "closest_selem_distance_fn",
        #     "bias_optim_mode", "bias_optim_args", "weights_optim_mode", "weights_optim_args",
        # ]
        return BiSEL._bises_args()

    @property
    def bisels_args(self):
        return set(self.bises_args).union(
            [
                'in_channels', 'constant_P_lui', "lui_kwargs",
                # "init_bias_value_bise", "init_bias_value_lui"
                # "bise_initializer_method", "bise_initializer_args",
                # "lui_initializer_method", "lui_initializer_args",
            ]
        ).difference(["init_bias_value"])

    def _make_layer(self, idx):
        if self.atomic_element[idx] == 'bisel':
            layer = BiSEL(initializer=self.bisel_initializers[idx], **self.bisels_kwargs_idx(idx))
            self.bisels_idx.append(idx)

        elif self.atomic_element[idx] == 'sybisel':
            layer = SyBiSEL(initializer=self.bisel_initializers[idx], **self.bisels_kwargs_idx(idx))
            self.bisels_idx.append(idx)

        return layer


class BiMoNNClassifier(BiMoNN, ABC):
    @classmethod
    def select_(cls, name: str) -> Optional["BiMoNNClassifier"]:
        """
        Recursive class method iterating over all subclasses to return the
        desired model class.
        """
        if cls.__name__.lower() == name:
            return cls

        for subclass in cls.__subclasses__():
            selected = subclass.select_(name)
            if selected is not None:
                return selected

        return None

    @classmethod
    def select(cls, name: str, **kwargs: Any) -> "BiMoNNClassifier":
        """
        Class method iterating over all subclasses to instantiate the desired
        model.
        """

        selected = cls.select_(name)
        if selected is None:
            raise ValueError("The selected model was not found.")

        return selected(**kwargs)

    @classmethod
    def listing(cls) -> List[str]:
        """List all the available models."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls.__name__.lower()}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing())

        return list(subclasses)


class BiMoNNClassifierLastLinear(BiMoNNClassifier):

    def __init__(
        self,
        kernel_size: List[Union[Tuple, int]],
        n_classes: int,
        input_size: Tuple[int],
        final_bisel_kwargs: Dict = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, kernel_size=kernel_size, **kwargs)
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        self.bisel_kwargs = self.bisels_kwargs_idx(0) if final_bisel_kwargs is None else final_bisel_kwargs
        self.bisel_kwargs["in_channels"] = self.out_channels[-1]
        self.bisel_kwargs["out_channels"] = n_classes
        self.bisel_kwargs["kernel_size"] = input_size
        self.bisel_kwargs["padding"] = 0
        # self.bisel_kwargs["init_bias_value_bise"] = 0.5
        # self.bisel_kwargs["lui_kwargs"]["force_identity"] = True
        self.classification_layer = BiSEL(**self.bisel_kwargs)

        self.in_channels.append(self.out_channels[-1])
        self.out_channels.append(n_classes)
        self.kernel_size.append(input_size)

        self.layers.append(self.classification_layer)
        self.bisels_idx.append(len(self.layers) - 1)

    def forward(self, x):
        output = super().forward(x)
        batch_size = output.shape[0]
        output = output.squeeze()
        if batch_size == 1:
            output = output.unsqueeze(0)
        return output

    def forward_save(self, x):
        output = super().forward_save(x)
        batch_size = output["output"].shape[0]
        output["output"] = output["output"].squeeze()
        if batch_size == 1:
            output["output"] = output["output"].unsqueeze(0)
        return output


class BiMoNNClassifierMaxPool(BiMoNNClassifier):
    def __init__(
        self,
        kernel_size: List[Union[Tuple, int]],
        n_classes: int,
        input_size: Tuple[int],
        final_bisel_kwargs: Dict = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, kernel_size=kernel_size, **kwargs)
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        self.repr_size = input_size
        self.maxpool_layers = []
        for _ in range(len(self)):
            self.repr_size = (self.repr_size[0] // 2, self.repr_size[1] // 2)
            self.maxpool_layers.append(nn.MaxPool2d((2, 2)))

        self.bisel_kwargs = self.bisels_kwargs_idx(0) if final_bisel_kwargs is None else final_bisel_kwargs
        self.bisel_kwargs["in_channels"] = self.out_channels[-1]
        self.bisel_kwargs["out_channels"] = n_classes
        self.bisel_kwargs["kernel_size"] = self.repr_size
        self.bisel_kwargs["padding"] = 0

        self.classification_layer = BiSEL(**self.bisel_kwargs)

        self.in_channels.append(self.out_channels[-1])
        self.out_channels.append(n_classes)
        self.kernel_size.append(input_size)

        self.layers.append(self.classification_layer)
        self.bisels_idx.append(len(self.layers) - 1)

    def forward(self, x):
        output = self.maxpool_layers[0](self.layers[0](x))
        for layer_bise, layer_maxpool in zip(self.layers[1:-1], self.maxpool_layers[1:]):
            output = layer_maxpool(layer_bise(output))
        output = self.layers[-1](output)

        batch_size = output.shape[0]
        output = output.squeeze()
        if batch_size == 1:
            output = output.unsqueeze(0)

        return output

    def forward_save(self, x):
        output = {"input": x}
        cur = self.layers[0].forward_save(x)
        output[0] = cur
        cur['output'] = nn.MaxPool(cur['output'])

        for layer_idx, layer in enumerate(self.layers[1:], start=1):
            cur = layer.forward_save(cur['output'])
            cur['output'] = nn.MaxPool(cur['output'])
            output[layer_idx] = cur

        output[len(self) - 1] = self.layers[-1].forward_save(cur['output'])
        output["output"] = cur["output"]

        batch_size = output["output"].shape[0]
        output["output"] = output["output"].squeeze()
        if batch_size == 1:
            output["output"] = output["output"].unsqueeze(0)
        return output


class BiMoNNClassifierMaxPoolNotBinary(BiMoNNClassifier):
    def __init__(
        self,
        kernel_size: List[Union[Tuple, int]],
        n_classes: int,
        input_size: Tuple[int],
        final_bisel_kwargs: Dict = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, kernel_size=kernel_size, **kwargs)
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        self.repr_size = input_size
        self.maxpool_layers = []
        for _ in range(len(self)):
            self.repr_size = (self.repr_size[0] // 2, self.repr_size[1] // 2)
            self.maxpool_layers.append(nn.MaxPool2d((2, 2)))

        self.bisel_kwargs = self.bisels_kwargs_idx(0) if final_bisel_kwargs is None else final_bisel_kwargs
        self.bisel_kwargs["in_channels"] = self.out_channels[-1]
        self.bisel_kwargs["out_channels"] = n_classes
        self.bisel_kwargs["kernel_size"] = self.repr_size
        self.bisel_kwargs["padding"] = 0

        self.classification_layer = BiSELNotBinary(**self.bisel_kwargs)

        self.in_channels.append(self.out_channels[-1])
        self.out_channels.append(n_classes)
        self.kernel_size.append(input_size)

        self.layers.append(self.classification_layer)
        self.bisels_idx.append(len(self.layers) - 1)

    def forward(self, x):
        output = self.maxpool_layers[0](self.layers[0](x))
        for layer_bise, layer_maxpool in zip(self.layers[1:-1], self.maxpool_layers[1:]):
            output = layer_maxpool(layer_bise(output))
        output = self.layers[-1](output)

        batch_size = output.shape[0]
        output = output.squeeze()
        if batch_size == 1:
            output = output.unsqueeze(0)

        return output

    def forward_save(self, x):
        output = {"input": x}
        cur = self.layers[0].forward_save(x)
        output[0] = cur
        cur['output'] = nn.MaxPool(cur['output'])

        for layer_idx, layer in enumerate(self.layers[1:], start=1):
            cur = layer.forward_save(cur['output'])
            cur['output'] = nn.MaxPool(cur['output'])
            output[layer_idx] = cur

        output[len(self) - 1] = self.layers[-1].forward_save(cur['output'])
        output["output"] = cur["output"]

        batch_size = output["output"].shape[0]
        output["output"] = output["output"].squeeze()
        if batch_size == 1:
            output["output"] = output["output"].unsqueeze(0)
        return output
