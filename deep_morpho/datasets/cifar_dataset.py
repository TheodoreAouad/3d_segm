from typing import Tuple, Dict, Callable

import torch
from torchvision.datasets import CIFAR10, CIFAR100
from random import choice

from .gray_to_channels_dataset import GrayToChannelDatasetBase, LevelsetValuesEqualIndex, LevelsetValuesHandler
from .select_indexes_dataset import SelectIndexesDataset


with open('deep_morpho/datasets/root_cifar10_dir.txt', 'r') as f:
    ROOT_CIFAR10_DIR = f.read()

with open('deep_morpho/datasets/root_cifar100_dir.txt', 'r') as f:
    ROOT_CIFAR100_DIR = f.read()


class CIFAR10Dataset(GrayToChannelDatasetBase, CIFAR10):
    def __init__(
        self,
        root: str = ROOT_CIFAR10_DIR,
        preprocessing: Callable = None,
        train: bool = True,
        *args, **kwargs
    ):
        CIFAR10.__init__(self, root=root, transform=lambda x: torch.tensor(x), train=train, )
        self.preprocessing = preprocessing

        GrayToChannelDatasetBase.__init__(
            self,
            img=torch.tensor(choice(self.data).transpose(2, 0, 1)),
            *args, **kwargs
        )


class CIFAR100Dataset(GrayToChannelDatasetBase, CIFAR100):
    def __init__(
        self,
        root: str = ROOT_CIFAR100_DIR,
        preprocessing: Callable = None,
        train: bool = True,
        *args, **kwargs
    ):
        CIFAR100.__init__(self, root=root, transform=lambda x: torch.tensor(x), train=train, )
        self.preprocessing = preprocessing

        GrayToChannelDatasetBase.__init__(
            self,
            img=torch.tensor(choice(self.data).transpose(2, 0, 1)),
            *args, **kwargs
        )
