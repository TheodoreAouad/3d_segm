from typing import Union, Tuple, Dict

import torch

from .bise_old import InitBiseEnum, BiseBiasOptimEnum
# from .bise_old import BiSE, InitBiseEnum, SyBiSE, BiseBiasOptimEnum
# from .bise_old2 import BiSE as BiSE_OLD2

from .bise import BiSE, SyBiSE
from .lui import LUI, SyLUI
from .binary_nn import BinaryNN


class BiSELBase(BinaryNN):

    def __init__(
        self,
        bise_module: BiSE,
        lui_module: LUI,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        threshold_mode: Union[Dict[str, str], str] = {"weight": "softplus", "activation": "tanh"},
        constant_P_lui: bool = False,
        init_bias_value_bise: float = 0.5,
        init_bias_value_lui: float = 0.5,
        input_mean: float = 0.5,
        lui_input_mean: float = 0.5,
        init_weight_mode: InitBiseEnum = InitBiseEnum.CUSTOM_HEURISTIC,
        lui_kwargs: Dict = {},
        **bise_kwargs
    ):
        super().__init__()
        self.bise_module = bise_module
        self.lui_module = lui_module
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.threshold_mode = self._init_threshold_mode(threshold_mode)
        self.constant_P_lui = constant_P_lui
        self.bise_kwargs = bise_kwargs
        self.lui_kwargs = lui_kwargs
        self.init_bias_value_bise = init_bias_value_bise
        self.init_bias_value_lui = init_bias_value_lui
        self.input_mean = input_mean
        self.lui_input_mean = lui_input_mean
        self.init_weight_mode = init_weight_mode

        self.bises = self._init_bises()
        self.luis = self._init_luis()

    @staticmethod
    def _init_threshold_mode(threshold_mode):
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["weight", "activation"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode

    def _init_bises(self):
        bises = []
        for idx in range(self.in_channels):
            layer = self.bise_module(
                out_channels=self.out_channels, kernel_size=self.kernel_size, init_weight_mode=self.init_weight_mode,
                threshold_mode=self.threshold_mode, init_bias_value=self.init_bias_value_bise, input_mean=self.input_mean,
                **self.bise_kwargs
            )
            setattr(self, f'bise_{idx}', layer)
            bises.append(layer)
        return bises

    def _init_luis(self):
        luis = []
        for idx in range(self.out_channels):
            layer = self.lui_module(
                in_channels=self.in_channels,
                threshold_mode=self.threshold_mode,
                out_channels=1,
                constant_activation_P=self.constant_P_lui,
                init_bias_value=self.init_bias_value_lui,
                input_mean=self.lui_input_mean,
                init_weight_mode=self.init_weight_mode,
                **self.lui_kwargs
            )
            setattr(self, f'lui_{idx}', layer)
            luis.append(layer)
        return luis


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bise_res2 = torch.cat([
            layer(x[:, chan_input:chan_input+1, ...])[:, None, ...] for chan_input, layer in enumerate(self.bises)
        ], axis=1)  # bise_res shape: (batch_size, in_channels, out_channels, width, length)
        # bise_res2 = torch.cat([
        #     1-bise_res,
        #     bise_res,
        # ], axis=1)  # the chan_output does not change, so we have 2x chan_input

        lui_res = torch.cat([
            layer(bise_res2[:, :, chan_output, ...]) for chan_output, layer in enumerate(self.luis)
        ], axis=1)

        return lui_res

    def forward_save(self, x):
        output = {}

        bise_res2 = torch.cat([
            layer(x[:, chan_input:chan_input+1, ...])[:, None, ...] for chan_input, layer in enumerate(self.bises)
        ], axis=1)
        lui_res = torch.cat([
            layer(bise_res2[:, :, chan_output, ...]) for chan_output, layer in enumerate(self.luis)
        ], axis=1)

        for chan_output in range(len(self.luis)):
            output[chan_output] = lui_res[:, chan_output, ...]
            for chan_input in range(len(self.bises)):
                output[chan_input, chan_output] = bise_res2[:, chan_input, chan_output, ...]

        output['output'] = lui_res

        return output

    @property
    def weight(self) -> torch.Tensor:
        """ Returns the convolution weights, of shape (out_channels, in_channels, W, L).
        """
        return torch.cat([layer.weight for layer in self.bises], axis=1)

    @property
    def weights(self) -> torch.Tensor:
        return self.weight

    @property
    def activation_P_bise(self) -> torch.Tensor:
        """ Returns the activations P of the bise layers, of shape (out_channels, in_channels).
        """
        return torch.stack([layer.activation_P for layer in self.bises], axis=-1)

    @property
    def weight_P_bise(self) -> torch.Tensor:
        """ Returns the weights P of the bise layers, of shape (out_channels, in_channels).
        """
        return torch.stack([layer.weight_P for layer in self.bises], axis=-1)


    @property
    def activation_P_lui(self) -> torch.Tensor:
        """ Returns the activations P of the lui layer, of shape (out_channels).
        """
        return torch.cat([layer.activation_P for layer in self.luis])


    @property
    def bias_bise(self) -> torch.Tensor:
        """ Returns the bias of the bise layers, of shape (out_channels, in_channels).
        """
        return torch.stack([layer.bias for layer in self.bises], axis=-1)

    @property
    def bias_bises(self) -> torch.Tensor:
        return self.bias_bise

    @property
    def bias_lui(self) -> torch.Tensor:
        """Returns the bias of the lui layer, of shape (out_channels).
        """
        return torch.cat([layer.bias for layer in self.luis])

    @property
    def bias_luis(self) -> torch.Tensor:
        return self.bias_lui

    @property
    def normalized_weight(self) -> torch.Tensor:
        """ Returns the convolution weights, of shape (out_channels, in_channels, W, L).
        """
        return torch.cat([layer._normalized_weight for layer in self.bises], axis=1)

    @property
    def normalized_weights(self) -> torch.Tensor:
        return self.normalized_weight

    @property
    def _normalized_weight(self) -> torch.Tensor:
        return self.normalized_weight

    @property
    def _normalized_weights(self) -> torch.Tensor:
        return self.normalized_weights

    @property
    def coefs(self) -> torch.Tensor:
        """ Returns the coefficients of the linear operation of LUI, of shape (out_channels, in_channels).
        """
        return torch.cat([layer.coefs for layer in self.luis], axis=0)



class BiSEL(BiSELBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        threshold_mode: Union[Dict[str, str], str] = {"weight": "softplus", "activation": "tanh_symetric"},
        constant_P_lui: bool = False,
        init_bias_value_bise: float = 0.5,
        init_bias_value_lui: float = 0.5,
        input_mean: float = 0.5,
        init_weight_mode: InitBiseEnum = InitBiseEnum.CUSTOM_HEURISTIC,
        lui_kwargs: Dict = {},
        **bise_kwargs
    ):
        super().__init__(
            bise_module=BiSE,
            lui_module=LUI,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            threshold_mode=threshold_mode,
            constant_P_lui=constant_P_lui,
            init_bias_value_bise=init_bias_value_bise,
            init_bias_value_lui=init_bias_value_lui,
            input_mean=input_mean,
            init_weight_mode=init_weight_mode,
            lui_kwargs=lui_kwargs,
            **bise_kwargs
        )


class SyBiSEL(BiSELBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        threshold_mode: Union[Dict[str, str], str] = {"weight": "softplus", "activation": "tanh_symetric"},
        constant_P_lui: bool = False,
        init_bias_value_bise: float = 0,
        init_bias_value_lui: float = 0,
        input_mean: float = 0,
        init_weight_mode: InitBiseEnum = InitBiseEnum.CUSTOM_CONSTANT,
        bias_optim_mode: BiseBiasOptimEnum = BiseBiasOptimEnum.RAW,
        lui_kwargs: Dict = {},
        **bise_kwargs
    ):
        super().__init__(
            bise_module=SyBiSE,
            lui_module=SyLUI,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias_optim_mode=bias_optim_mode,
            threshold_mode=threshold_mode,
            constant_P_lui=constant_P_lui,
            init_bias_value_bise=init_bias_value_bise,
            init_bias_value_lui=init_bias_value_lui,
            input_mean=input_mean,
            init_weight_mode=init_weight_mode,
            lui_kwargs=lui_kwargs,
            **bise_kwargs
        )
