import torch

from deep_morpho.models import Softplus


class TestSoftplus:

    @staticmethod
    def test_inverse():
        layer = Softplus()
        x = torch.rand(10,)
        assert (x - layer.forward_inverse(layer(x))).abs().sum() < 0.0001
