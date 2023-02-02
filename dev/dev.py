from typing import List, Tuple

import inspect

from deep_morpho.datasets.datamodule_base import DataModule
from torch.utils.data.dataset import Dataset


class A(DataModule, Dataset):
    def __init__(self, a=5, b=2):
        pass

    # @classmethod
    # def default_args(cls) -> List[Tuple[str, dict]]:
    #     """Return the default arguments of the model, in the format of argparse.ArgumentParser
    #     Ex:
    #     >>> args = {}"""
    #     return [
    #         (
    #             name,
    #             {"default": p.default}
    #         )
    #         for name, p in inspect.signature(cls.__init__).parameters.items()
    #     ]
