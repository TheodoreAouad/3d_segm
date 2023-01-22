from typing import Tuple

import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
from random import choice

from deep_morpho.tensor_with_attributes import TensorGray
from .gray_to_channels_dataset import GrayToChannelDataset, LevelsetValuesEqualIndex


with open('deep_morpho/datasets/root_cifar10_dir.txt', 'r') as f:
    ROOT_CIFAR10_DIR = f.read()

with open('deep_morpho/datasets/root_cifar100_dir.txt', 'r') as f:
    ROOT_CIFAR100_DIR = f.read()


class CIFAR10Dataset(CIFAR10, GrayToChannelDataset):
    def __init__(self, root=ROOT_CIFAR10_DIR, levelset_handler_mode=LevelsetValuesEqualIndex, levelset_handler_args={"n_values": 10}, transform=ToTensor(), *args, **kwargs):
        CIFAR10.__init__(self, root=root, transform=transform, *args, **kwargs)
        self.levelset_handler_mode = levelset_handler_mode
        self.levelset_handler_args = levelset_handler_args

        self.levelset_handler_args["img"] = CIFAR10.__getitem__(self, choice(range(len(self))))[0]
        self.levelset_handler = levelset_handler_mode(**levelset_handler_args)

        GrayToChannelDataset.__init__(self, self.levelset_handler)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, target = super().__getitem__(index)
        original_img = img + 0

        img = TensorGray(self.from_gray_to_channels(img))
        img.original = original_img


        return img, target




class CIFAR100Dataset(CIFAR100):
    def __init__(self, root=ROOT_CIFAR100_DIR, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
