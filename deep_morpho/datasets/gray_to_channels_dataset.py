from abc import ABC, abstractmethod

import torch
import numpy as np

from deep_morpho.gray_scale import level_sets_from_gray, gray_from_level_sets, undersample


class LevelsetValuesHandler(ABC):

    def __init__(self, img: torch.Tensor, *args, **kwargs):
        """
        Args:
            img (torch.Tensor): shape (n_channels, *img_shape)
        """
        self.levelset_values = self.init_levelset_values(img, *args, **kwargs)

    @abstractmethod
    def init_levelset_values(self, img: torch.Tensor, *args, **kwargs):
        return


class LevelsetValuesDefault(LevelsetValuesHandler):
    def init_levelset_values(self, img: torch.Tensor, *args, **kwargs):
        return torch.tensor([img.unique() for _ in range(img.shape[0])])


class LevelsetValuesEqualIndex(LevelsetValuesHandler):
    def __init__(self, n_values: int, *args, **kwargs):
        super().__init__(*args, n_values=n_values, **kwargs)

    def init_levelset_values(self, img: torch.Tensor, n_values: int, *args, **kwargs):
        levelsets = torch.zeros(img.shape[0], n_values)
        for chan in range(img.shape[0]):
            values, count = img.unique(return_counts=True)
            values = torch.tensor(sorted(sum([[v for _ in range(c)] for v, c in zip(values, count)], start=[])))  # repeat values that occur multiple times
            levelsets[chan] = values[undersample(0, len(values) - 1, n_values)]
        return levelsets




class GrayToChannelDataset:
    def __init__(self, values_handler: LevelsetValuesHandler):
        self.values_handler = values_handler

    @property
    def levelset_values(self):
        return self.values_handler.levelset_values

    def from_gray_to_channels(self, img: torch.Tensor) -> torch.Tensor:
        return self._from_gray_to_channels(img, self.levelset_values)

    def from_channels_to_gray(self, img_channels: torch.Tensor) -> torch.Tensor:
        return self._from_channels_to_gray(img_channels, self.levelset_values)

    @staticmethod
    def _from_gray_to_channels(img: torch.Tensor, levelset_values: torch.Tensor) -> torch.Tensor:
        """ Given a gray scale image with multiple channels, outputs a binary image with level sets as channels.

        Args:
            img (torch.Tensor): shape (channels, width, length), gray scale image
            levelset_values (np.ndarray): shape (channels, n_values), level set values for each channel

        Returns:
            torch.Tensor: shape (channels * n_values, width, length), binary image with channels as level sets
        """
        all_binary_imgs = []

        for chan in range(img.shape[0]):
            bin_img, _ = level_sets_from_gray(img[chan], levelset_values[chan])
            all_binary_imgs.append(bin_img)

        return torch.cat(all_binary_imgs, axis=0)

    @staticmethod
    def _from_channels_to_gray(img_channels: torch.Tensor, levelset_values: torch.Tensor) -> torch.Tensor:
        """ Given a binary image with level sets as channels, gives the gray scale image with multiple channels.

        Args:
            img_channels (torch.Tensor):  shape (channels * n_values, width, length), binary image with
                                            channels as level sets
            levelset_values (np.ndarray): shape (channels, n_values), level set values for each channel

        Returns:
            torch.Tensor: shape (channels, width, length), gray scale image
        """

        n_channels, n_values = levelset_values.shape
        gray_img = torch.zeros((n_channels,) + img_channels.shape[1:])

        for chan in range(n_channels):
            gray_img[chan] = gray_from_level_sets(
                img_channels[chan * n_values:(chan + 1) * n_values], levelset_values[chan]
            )

        return gray_img
