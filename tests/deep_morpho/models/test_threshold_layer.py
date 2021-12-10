from numpy.lib.function_base import disp
import torch

from deep_morpho.models.threshold_layer import dispatcher


class TestThresholdLayer:

    @staticmethod
    def test_threshold_image():
        x = torch.randn(1, 50, 50)
        x[(x > -.2) & (x < .2)] = 1
        layer = dispatcher['tanh'](P_=50, n_channels=1)
        assert ((x > 0).int() - layer(x)).abs().sum() < 1e-6

    @staticmethod
    def test_threshold_image_channels():
        x = torch.randn(6, 50, 50)
        x[(x > -.2) & (x < .2)] = 1
        layer = dispatcher['tanh'](P_=50, n_channels=6)
        assert ((x > 0).int() - layer(x)).abs().sum() < 1e-6
