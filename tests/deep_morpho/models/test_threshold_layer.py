import torch

from deep_morpho.models.threshold_layer import dispatcher


class TestThresholdLayer:

    @staticmethod
    def test_threshold_image():
        x = torch.randn(1, 1, 50, 50)
        x[(x > -.2) & (x < .2)] = 1
        layer = dispatcher['tanh'](P_=50, n_channels=1)
        assert ((x > 0).int() - layer(x)).abs().sum() < 1e-6

    @staticmethod
    def test_threshold_image_channels():
        x = torch.randn(1, 6, 50, 50)
        x[(x > -.2) & (x < .2)] = 1
        layer = dispatcher['tanh'](P_=50, n_channels=6)
        assert ((x > 0).int() - layer(x)).abs().sum() < 1e-6

    @staticmethod
    def test_inverse():
        x = torch.rand(1, 6, 50, 50)
        # x[(x > -.2) & (x < .2)] = 1
        for thresh_name in ['tanh', 'sigmoid', 'arctan', 'softplus']:
            layer = dispatcher[thresh_name](P_=1, n_channels=6)
            diff = (x - layer.forward_inverse(layer(x))).abs().max()
            assert diff < 1e-6
