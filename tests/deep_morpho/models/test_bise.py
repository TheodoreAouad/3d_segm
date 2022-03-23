import torch
import numpy as np
import pytest

from deep_morpho.models import BiSE
from general.structuring_elements import disk
from general.array_morphology import array_erosion, array_dilation


def softplus_inverse(x):
    return torch.log(torch.exp(x) - 1)


# # @pytest.mark.parametrize("threshold_mode", ["identity", "tanh"])
# def test_bise_forward_binary_dilation(threshold_mode):
#     weight = disk(3)
#     layer = BiSE.bise_from_selem(weight, 'dilation', threshold_mode=threshold_mode)

#     inpt = torch.randint(0, 2, (50, 50)).float()
#     output = layer.forward_binary(inpt[None, None, ...]).detach().cpu().numpy()
#     target = array_dilation(inpt, weight)

#     assert np.abs(output - target).sum() == 0

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
    def test_bise_forward_multi_channels():
        layer = BiSE((7, 7), out_channels=5)
        inpt = torch.rand(10, 1, 50, 50)
        layer(inpt)

    @staticmethod
    def test_bise_set_bias():
        layer = BiSE((7, 7))
        bias = torch.rand((1,)) - 2
        layer.set_bias(bias)
        assert layer.bias == bias

    @staticmethod
    def test_bise_erosion_check():
        layer = BiSE((7, 7), threshold_mode='sigmoid')
        weight = disk(3)
        layer.conv.weight.data[0] = torch.FloatTensor(100*(weight - 0.5))
        layer.set_bias(-torch.FloatTensor([weight.sum() - 1/2]))
        assert layer.is_erosion_by(layer._normalized_weight[0, 0], layer.bias, weight, v1=0.003, v2=0.997)

    @staticmethod
    def test_bise_dilation_check():
        layer = BiSE((7, 7), threshold_mode='sigmoid')
        weight = disk(3)
        layer.conv.weight.data[0] = torch.FloatTensor(10*(weight - 0.5))
        layer.set_bias(-torch.FloatTensor([1/2]))
        assert layer.is_dilation_by(layer._normalized_weight[0, 0], layer.bias, weight, v1=0.003, v2=0.997)

    @staticmethod
    def test_bise_erosion_init():
        weight = disk(3)
        layer = BiSE.bise_from_selem(weight, 'erosion')
        assert layer.is_erosion_by(layer._normalized_weight[0, 0], layer.bias, weight)

    @staticmethod
    def test_bise_dilation_init():
        weight = disk(3)
        layer = BiSE.bise_from_selem(weight, 'dilation')
        assert layer.is_dilation_by(layer._normalized_weight[0, 0], layer.bias, weight)

    @staticmethod
    def test_bise_conv_erosion_init():
        weight = disk(3)
        layer = BiSE.bise_from_selem(weight, 'erosion', threshold_mode="identity")
        assert layer.is_erosion_by(layer._normalized_weight[0, 0], layer.bias, weight)

    @staticmethod
    def test_bise_conv_dilation_init():
        weight = disk(3)
        layer = BiSE.bise_from_selem(weight, 'dilation', threshold_mode="identity")
        assert layer.is_dilation_by(layer._normalized_weight[0, 0], layer.bias, weight)


    @staticmethod
    def test_bise_find_selem_and_operation_chan():
        weight = disk(3)
        for original_op in ['erosion', 'dilation']:
            layer = BiSE.bise_from_selem(weight, original_op)
            selem, operation = layer.find_selem_and_operation_chan(0)
            assert (selem == weight).all()
            assert operation == original_op

    @staticmethod
    @pytest.mark.parametrize("threshold_mode", ["identity", "tanh"])
    def test_bise_forward_binary_dilation(threshold_mode):
        weight = disk(3)
        layer = BiSE.bise_from_selem(weight, 'dilation', threshold_mode=threshold_mode)

        inpt = torch.randint(0, 2, (50, 50)).float()
        output = layer.forward_binary(inpt[None, None, ...]).detach().cpu().numpy()
        target = array_dilation(inpt, weight)

        assert np.abs(output - target).sum() == 0

    @staticmethod
    @pytest.mark.parametrize("threshold_mode", ["identity", "tanh"])
    def test_bise_forward_binary_erosion(threshold_mode):
        weight = disk(3)
        layer = BiSE.bise_from_selem(weight, 'erosion', threshold_mode=threshold_mode)

        inpt = torch.randint(0, 2, (50, 50)).float()
        output = layer.forward_binary(inpt[None, None, ...]).detach().cpu().numpy()
        target = array_erosion(inpt, weight)

        assert np.abs(output - target).sum() == 0

    @staticmethod
    @pytest.mark.parametrize("threshold_mode", ["identity", "tanh"])
    def test_bise_binary_mode_dilation(threshold_mode):
        weight = disk(3)
        layer = BiSE.bise_from_selem(weight, 'dilation', threshold_mode=threshold_mode)

        inpt = torch.randint(0, 2, (50, 50)).float()

        layer.binary()
        output = layer(inpt[None, None, ...]).detach().cpu().numpy()
        target = array_dilation(inpt, weight)

        assert np.abs(output - target).sum() == 0

    @staticmethod
    @pytest.mark.parametrize("threshold_mode", ["identity", "tanh"])
    def test_bise_binary_mode_erosion(threshold_mode):
        weight = disk(3)
        layer = BiSE.bise_from_selem(weight, 'erosion', threshold_mode=threshold_mode)

        inpt = torch.randint(0, 2, (50, 50)).float()
        layer.binary()
        output = layer(inpt[None, None, ...]).detach().cpu().numpy()
        target = array_erosion(inpt, weight)

        assert np.abs(output - target).sum() == 0

    @staticmethod
    @pytest.mark.parametrize("threshold_mode", ["identity", "tanh"])
    def test_bise_binary_mode_complementation(threshold_mode):
        weight = disk(2)
        layer = BiSE.bise_from_selem(weight, 'dilation', threshold_mode=threshold_mode)
        layer.activation_threshold_layer.P_.requires_grad = False
        layer.activation_threshold_layer.P_[0] *= -1

        inpt = torch.randint(0, 2, (50, 50)).float()
        layer.binary()
        output = layer(inpt[None, None, ...]).detach().cpu().numpy()
        target = 1 - array_dilation(inpt, weight)

        assert np.abs(output - target).sum() == 0


    @staticmethod
    def test_bise_binary_forward_multi_channels():
        layer = BiSE((7, 7), out_channels=5)
        inpt = torch.rand(10, 1, 50, 50)
        output = layer.forward_binary(inpt).detach().cpu().numpy()
        assert np.isin(output, [0, 1]).all()


    @staticmethod
    def test_bise_binary_mode():
        layer = BiSE((7, 7), out_channels=5)
        layer.binary()
        layer.binary(False)
        assert layer.binary_mode == False
