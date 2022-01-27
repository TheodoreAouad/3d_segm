import torch

from deep_morpho.models import BiSE
from general.structuring_elements import disk


class TestBiSE:

    @staticmethod
    def test_bise_init():
        layer = BiSE((7, 7), threshold_mode='sigmoid')
        layer


    @staticmethod
    def test_bise_forward():
        layer = BiSE((7, 7), threshold_mode='sigmoid')
        inpt = torch.rand((1, 1, 50, 50))
        layer(inpt)


    @staticmethod
    def test_bise_erosion_check():
        layer = BiSE((7, 7), threshold_mode='sigmoid')
        weight = disk(3)
        layer.conv.weight.data[0] = torch.FloatTensor(100*(weight - 0.5))
        layer.conv.bias.data = -torch.FloatTensor([weight.sum() - 1/2])
        assert layer.is_erosion_by(layer._normalized_weight, layer.bias, weight, v1=0.003, v2=0.997)


    @staticmethod
    def test_bise_dilation_check():
        layer = BiSE((7, 7), threshold_mode='sigmoid')
        weight = disk(3)
        layer.conv.weight.data[0] = torch.FloatTensor(10*(weight - 0.5))
        layer.conv.bias.data = -torch.FloatTensor([1/2])
        assert layer.is_dilation_by(layer._normalized_weight, layer.bias, weight, v1=0.003, v2=0.997)
