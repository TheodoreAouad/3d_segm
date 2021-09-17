import torch

from deep_morpho.models.complementation_layer import ComplementationLayer


class TestComplementationLayer:

    @staticmethod
    def test_identity():
        layer = ComplementationLayer(alpha_init=10000)
        x = torch.randint(2, (30, 30, 30))

        assert (layer(x) - x).abs().sum() / x.sum() < 1e-3

    @staticmethod
    def test_complementation():
        layer = ComplementationLayer(alpha_init=-10000)
        x = torch.randint(2, (30, 30, 30))

        assert (layer(x) - (1 - x)).abs().sum() / (1-x).sum() < 1e-3
