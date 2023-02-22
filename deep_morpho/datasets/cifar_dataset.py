from typing import Callable

import torch
from torchvision.datasets import CIFAR10, CIFAR100
from random import choice
import torchvision.transforms as transforms

from .gray_to_channels_dataset import GrayToChannelDatasetBase
from .select_indexes_dataset import SelectIndexesDataset


with open('deep_morpho/datasets/root_cifar10_dir.txt', 'r') as f:
    ROOT_CIFAR10_DIR = f.read()

with open('deep_morpho/datasets/root_cifar100_dir.txt', 'r') as f:
    ROOT_CIFAR100_DIR = f.read()



transform_default = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class CIFAR10Classical(SelectIndexesDataset, CIFAR10):
    def __init__(
        self,
        root: str = ROOT_CIFAR10_DIR,
        preprocessing: Callable = None,
        train: bool = True,
        transform: Callable = transform_default,
        *args, **kwargs
    ):
        CIFAR10.__init__(self, root=root, transform=transform, train=train,)
        self.preprocessing = preprocessing

        SelectIndexesDataset.__init__(self, *args, **kwargs)


class CIFAR100Classical(SelectIndexesDataset, CIFAR100):
    def __init__(
        self,
        root: str = ROOT_CIFAR100_DIR,
        preprocessing: Callable = None,
        train: bool = True,
        transform: Callable = transform_default,
        *args, **kwargs
    ):
        CIFAR100.__init__(self, root=root, transform=transform, train=train,)
        self.preprocessing = preprocessing

        SelectIndexesDataset.__init__(self, *args, **kwargs)


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
