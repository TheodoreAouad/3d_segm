from typing import Union, Tuple, Dict

from .lui import LUI
from .bise import BiSE
from ..initializer.bisel_initializer import BiselInitializer, BiselInitIdentical
from ..initializer.bise_initializer import InitBiseHeuristicWeights
from .bisel import BiSELBase


class LuiNotBinary(LUI):
    def forward(self, x):
        return self._forward(x)

    # def _specific_numel_binary(self):
    def numel_binary(self):
        return 0


class BiSELNotBinary(BiSELBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        threshold_mode: Union[Dict[str, str], str] = {"weight": "softplus", "activation": "tanh_symetric"},
        constant_P_lui: bool = False,
        # init_bias_value_bise: float = 0.5,
        # init_bias_value_lui: float = 0.5,
        # input_mean: float = 0.5,
        # init_weight_mode: InitBiseEnum = InitBiseEnum.CUSTOM_HEURISTIC,
        initializer: BiselInitializer = BiselInitIdentical(InitBiseHeuristicWeights(init_bias_value=1, input_mean=0.5)),
        # initializer_method: InitBiseEnum = InitBiseEnum.CUSTOM_HEURISTIC,
        # initializer_args: Dict = {"init_bias_value": 1, "input_mean": 0.5},
        lui_kwargs: Dict = {},
        **bise_kwargs
    ):
        super().__init__(
            bise_module=BiSE,
            lui_module=LuiNotBinary,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            threshold_mode=threshold_mode,
            constant_P_lui=constant_P_lui,
            initializer=initializer,
            # init_bias_value_bise=init_bias_value_bise,
            # init_bias_value_lui=init_bias_value_lui,
            # input_mean=input_mean,
            # init_weight_mode=init_weight_mode,
            lui_kwargs=lui_kwargs,
            **bise_kwargs
        )
