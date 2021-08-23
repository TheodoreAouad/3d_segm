import torch

from deep_morpho.models.logical_not_layer import LogicalNotLayer


class TestLogicalNotLayer:

    @staticmethod
    def test_identity():
        layer = LogicalNotLayer(alpha_init=10000)
        x = torch.randint(2, (30, 30, 30))

        assert (layer(x) - x).abs().sum() / x.sum() < 1e-3

    @staticmethod
    def test_logical_not():
        layer = LogicalNotLayer(alpha_init=-10000)
        x = torch.randint(2, (30, 30, 30))

        assert (layer(x) - (1 - x)).abs().sum() / (1-x).sum() < 1e-3
