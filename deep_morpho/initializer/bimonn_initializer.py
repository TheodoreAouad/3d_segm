from copy import deepcopy
from typing import Dict, List, Tuple

from .bisel_initializer import BiselInitializer, BiselInitIdenticalMethod


class BimonnInitializer:
    """The Bimonn initializer works in two steps. First, it generates intializers for each bisel. Then, it proposes
    another step of post initialization, to rework the initialization (ex: in case we need to adapt init to other layers).
    """
    def generate_bisel_initializers(self, module) -> List[Tuple[BiselInitializer, Dict]]:
        return [(BiselInitializer, {}) for _ in range(len(module))]

    def post_initialize(self, module):
        pass


class BimonnInitIdenticalBisel(BimonnInitializer):

    def __init__(self, bisel_initializer_method: BiselInitializer, bisel_initializer_args: Dict = {}):
        self.bisel_initializer_method = bisel_initializer_method
        self.bisel_initializer_args = bisel_initializer_args

    def generate_bisel_initializers(self, module) -> List[Tuple[BiselInitializer, Dict]]:
        return [(self.bisel_initializer_method, self.bisel_initializer_args) for _ in range(len(module))]


class BimonnInitIdenticalBise(BimonnInitIdenticalBisel):
    def __init__(self, bise_initializer_args: Dict = {}):
        super().__init__(bisel_initializer_method=BiselInitIdentical, bisel_initializer_args=bise_initializer_args)


class BimonnInitIdenticalFirstInput(BimonnInitializer):
    def __init__(self, input_mean: float, bisel_initializer_method: BiselInitializer, bisel_initializer_args: Dict = {}):
        self.input_mean = input_mean
        self.bisel_initializer_method = bisel_initializer_method
        self.bisel_initializer_args = bisel_initializer_args

    def generate_bisel_initializers(self, module) -> List[Tuple[BiselInitializer, Dict]]:
        res = [deepcopy(self.bisel_initializer_args).update({"input_mean": self.input_mean})]

        return res + [(self.bisel_initializer_method, deepcopy(self.bisel_initializer_args)) for _ in range(1, len(module))]
