from typing import List, Tuple, Union, Dict

import numpy as np

from .bise import BiSE
from .dilation_sum_layer import MaxPlusAtom
# from .cobise import COBiSE, COBiSEC
from .bisel import BiSEL, SyBiSEL
from .binary_nn import BinaryNN


class BiMoNN(BinaryNN):

    def __init__(
        self,
        kernel_size: List[Union[Tuple, int]],
        channels: List[int],
        atomic_element: Union[str, List[str]] = 'bisel',
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = self._init_kernel_size(kernel_size)

        # for attr in set(self.all_args):
        #     setattr(self, attr, self._init_attr(attr, eval(attr)))

        kwargs['kernel_size'] = kernel_size
        kwargs['channels'] = channels
        kwargs['atomic_element'] = atomic_element
        for attr, value in kwargs.items():
            setattr(self, attr, self._init_attr(attr, value))


        self.layers = []
        self.bises_idx = []
        self.bisecs_idx = []
        self.bisels_idx = []
        for idx in range(len(self.kernel_size)):
            layer = self._make_layer(idx)
            self.layers.append(layer)
            setattr(self, f'layer{idx+1}', layer)

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
        return len(self.layers)

    @staticmethod
    def _init_kernel_size(kernel_size: List[Union[Tuple, int]]):
        res = []
        for size in kernel_size:
            if isinstance(size, int):
                res.append((size, size))
            else:
                res.append(size)
        return res

    def _init_channels(self, channels: List[int]):
        self.out_channels = channels[1:]
        self.in_channels = channels[:-1]
        self.channels = channels
        return self.channels

    def _init_input_mean(self, input_mean: Union[float, List[float]]):
        if isinstance(input_mean, list):
            return input_mean
        return [input_mean] + [0.5 for _ in range(1, len(self.kernel_size))]

    def _init_atomic_element(self, atomic_element: Union[str, List[str]]):
        if isinstance(atomic_element, list):
            return [s.lower() for s in atomic_element]

        return [atomic_element.lower() for _ in range(len(self.kernel_size))]

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

        return [attr_value for _ in range(len(self.kernel_size))]

    def is_not_default(self, key: str) -> bool:
        return key in self.__dict__.keys()

    def bises_kwargs_idx(self, idx):
        return dict(
            **{'shared_weights': None, },
            **{k: getattr(self, k)[idx] for k in self.bises_args if self.is_not_default(k)}
        )

    def bisels_kwargs_idx(self, idx):
        return dict(
            **{'shared_weights': None, },
            **{k: getattr(self, k)[idx] for k in self.bisels_args if self.is_not_default(k)}
        )

    def bisecs_kwargs_idx(self, idx):
        return dict(
            **{'shared_weights': None, },
            **{k: getattr(self, k)[idx] for k in self.bisecs_args if self.is_not_default(k)}
        )

    def cobise_kwargs_idx(self, idx):
        return dict(
            **{k: getattr(self, k)[idx] for k in self.cobise_args if self.is_not_default(k)},
        )

    def cobisec_kwargs_idx(self, idx):
        return dict(
            **{k: getattr(self, k)[idx] for k in self.cobisec_args if self.is_not_default(k)},
        )

    def max_plus_kwargs_idx(self, idx):
        return dict(
            **{k: getattr(self, k)[idx] for k in self.max_plus_args if self.is_not_default(k)},
        )

    @property
    def bises_args(self):
        return [
            'kernel_size', 'weight_P', 'threshold_mode', 'activation_P',
            'init_bias_value', 'init_weight_mode', 'out_channels', "constant_activation_P",
            "constant_weight_P", "input_mean", "closest_selem_method", "closest_selem_distance_fn",
            "bias_optim_mode",
        ]

    @property
    def bisels_args(self):
        return set(self.bises_args).union(
            ['in_channels', 'constant_P_lui', "lui_kwargs", "init_bias_value_bise", "init_bias_value_lui"]
        ).difference(["init_bias_value"])

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

        elif self.atomic_element[idx] == 'bisel':
            layer = BiSEL(**self.bisels_kwargs_idx(idx))
            self.bisels_idx.append(idx)

        # elif self.atomic_element[idx] == 'bisel2':
        #     layer = BiSEL2(**self.bisels_kwargs_idx(idx))
        #     self.bisels_idx.append(idx)

        elif self.atomic_element[idx] == 'sybisel':
            layer = SyBiSEL(**self.bisels_kwargs_idx(idx))
            self.bisels_idx.append(idx)

        # elif self.atomic_element[idx] == 'bisec':
        #     layer (**self.bisecs_kwargs_idx(idx))
        #     self.bisecs_idx.append(idx)

        elif self.atomic_element[idx] == 'max_plus':
            layer = MaxPlusAtom(**self.max_plus_kwargs_idx(idx))

        # elif self.atomic_element[idx] == 'cobise':
        #     layer = COBiSE(**self.cobise_kwargs_idx(idx))

        # elif self.atomic_element[idx] == 'cobisec':
        #     layer = COBiSEC(**self.cobisec_kwargs_idx(idx))

        return layer

    # @property
    # def all_args(self):
    #     return [
    #         "kernel_size", "atomic_element", "weight_P", "threshold_mode", "activation_P", "constant_activation_P",
    #         "init_bias_value_bise", "init_bias_value_lui", "init_weight_mode", "alpha_init", "init_value", "share_weights",
    #         "constant_weight_P", "constant_P_lui", "channels", "lui_kwargs", "input_mean", "closest_selem_method",
    #         "closest_selem_distance_fn", "bias_optim_mode"
    #     ]



class BiMoNNClassifier(BiMoNN):

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
