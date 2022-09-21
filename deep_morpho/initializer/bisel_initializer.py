from typing import Dict, List, Tuple
from copy import deepcopy

from .bise_initializer import BiseInitializer


class BiselInitializer:
    """The Bisel initializer works in two steps. First, it generates one initializer for bise and one for lui. Then, it proposes
    another step of post initialization, to rework the initialization (ex: in case we need to have dependent init between lui and bise).
    """
    def __init__(
        self,
        bise_initializer_method: BiseInitializer = BiseInitializer,
        bise_initializer_args: Dict = {},
        lui_initializer_method: BiseInitializer = BiseInitializer,
        lui_initializer_args: Dict = {},
        *args, **kwargs
    ):
        self.bise_initializer_method = bise_initializer_method
        self.bise_initializer_args = bise_initializer_args
        self.lui_initializer_method = lui_initializer_method
        self.lui_initializer_args = lui_initializer_args

    def get_bise_initializers(self, module) -> Tuple[BiseInitializer, Dict]:
        return self.bise_initializer_method, deepcopy(self.bise_initializer_args)

    def get_lui_initializers(self, module) -> Tuple[BiseInitializer, Dict]:
        return self.lui_initializer_method, deepcopy(self.lui_initializer_args)

    def post_initialize(self, module):
        pass


class BiselInitIdenticalMethod:
    def __init__(self, initializer_method: BiseInitializer, initializer_args_bise: Dict, initializer_args_lui: Dict, *args, **kwargs):
        super().__init__(
            bise_initializer_method=initializer_method,
            bise_initializer_args=initializer_args_bise,
            lui_initializer_method=initializer_method,
            lui_initializer_args=initializer_args_lui,
        )

    @property
    def initializer_method(self):
        return self.bise_initializer_method
