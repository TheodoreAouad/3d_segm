import torch
import numpy as np

from deep_morpho.datasets.gray_to_channels_dataset import GrayToChannelDatasetBase


class TestGrayToChannelDatasetBase:
    @staticmethod
    def test_from_gray_to_channels():
        x = torch.ones(3, 20, 20)
        values = np.zeros((3, 10))
        values[0] = np.arange(10)
        values[1] = np.arange(10)
        values[2] = np.arange(10)

        xbin = GrayToChannelDatasetBase.from_gray_to_channels(x, values)
        assert xbin.shape == (30, 20, 20)

    @staticmethod
    def test_identity():
        x = torch.ones(3, 20, 20)
        values = np.zeros((3, 10))
        values[0] = np.arange(10)
        values[1] = np.arange(10)
        values[2] = np.arange(10)

        xbin = GrayToChannelDatasetBase.from_gray_to_channels(x, values)
        xrec = GrayToChannelDatasetBase.from_channels_to_gray(xbin, values)
        assert (x == xrec).all()
