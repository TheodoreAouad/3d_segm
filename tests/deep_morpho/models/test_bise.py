import torch
import torch.nn as nn
import numpy as np
import pytest

from deep_morpho.models import BiSE, InitBiseEnum, BiseBiasOptimEnum
from deep_morpho.datasets import InputOutputGeneratorDataset, get_random_diskorect_channels
from deep_morpho.models.bise_base import BiseWeightsOptimEnum
from general.structuring_elements import disk
from general.array_morphology import array_erosion, array_dilation
from deep_morpho.morp_operations import ParallelMorpOperations


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
        layer = BiSE((7, 7), bias_optim_mode=BiseBiasOptimEnum.POSITIVE)
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
    @pytest.mark.parametrize("threshold_mode", ["identity", "softplus"])
    def test_bise_forward_binary_dilation(threshold_mode):
        weight = disk(3)
        layer = BiSE.bise_from_selem(weight, 'dilation', threshold_mode=threshold_mode)

        inpt = torch.randint(0, 2, (50, 50)).float()
        output = layer.forward_binary(inpt[None, None, ...]).detach().cpu().numpy()
        target = array_dilation(inpt, weight)

        assert np.abs(output - target).sum() == 0

    @staticmethod
    @pytest.mark.parametrize("threshold_mode", ["identity", "softplus"])
    def test_bise_forward_binary_erosion(threshold_mode):
        weight = disk(3)
        layer = BiSE.bise_from_selem(weight, 'erosion', threshold_mode=threshold_mode)

        inpt = torch.randint(0, 2, (50, 50)).float()
        output = layer.forward_binary(inpt[None, None, ...]).detach().cpu().numpy()
        target = array_erosion(inpt, weight)

        assert np.abs(output - target).sum() == 0

    @staticmethod
    @pytest.mark.parametrize("threshold_mode", ["identity", "softplus"])
    def test_bise_binary_mode_dilation(threshold_mode):
        weight = disk(3)
        layer = BiSE.bise_from_selem(weight, 'dilation', threshold_mode=threshold_mode)

        inpt = torch.randint(0, 2, (50, 50)).float()

        layer.binary()
        output = layer(inpt[None, None, ...]).detach().cpu().numpy()
        target = array_dilation(inpt, weight)

        assert np.abs(output - target).sum() == 0

    @staticmethod
    @pytest.mark.parametrize("threshold_mode", ["identity", "softplus"])
    def test_bise_binary_mode_erosion(threshold_mode):
        weight = disk(3)
        layer = BiSE.bise_from_selem(weight, 'erosion', threshold_mode=threshold_mode)

        inpt = torch.randint(0, 2, (50, 50)).float()
        layer.binary()
        output = layer(inpt[None, None, ...]).detach().cpu().numpy()
        target = array_erosion(inpt, weight)

        assert np.abs(output - target).sum() == 0

    @staticmethod
    @pytest.mark.parametrize("threshold_mode", ["identity", "softplus"])
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


    @staticmethod
    @pytest.mark.parametrize("input_mean", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_init_mode_custom_constant(input_mean):
        layer = BiSE(
            (7, 7),
            init_weight_mode=InitBiseEnum.CUSTOM_CONSTANT,
            init_bias_value="auto",
            input_mean=input_mean,
        )
        layer


    @staticmethod
    def test_duality_training():
        layer = BiSE(
            kernel_size=(7, 7),
            threshold_mode={'weight': 'softplus', 'activation': 'tanh'},
            activation_P=1,
            init_bias_value=1/2,
            input_mean=1/2,
            init_weight_mode=InitBiseEnum.CUSTOM_HEURISTIC,
            out_channels=1,
            bias_optim_mode=BiseBiasOptimEnum.POSITIVE,
            bias_optim_args={"offset": 0},
            weights_optim_mode=BiseWeightsOptimEnum.NORMALIZED,
        )

        dataset = InputOutputGeneratorDataset(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02, "border": (0, 0)},
            morp_operation=ParallelMorpOperations.dilation(('disk', 2)),
            len_dataset=1000,
            seed=100,
            do_symetric_output=False,
        )

        x, y = dataset[0]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        loss = nn.BCELoss()

        out1 = layer(x)
        value_loss1 = loss(out1, y)
        value_loss1.backward()


        grads1 = {}
        for name, param in layer.named_parameters():
            if param.grad is not None:
                grads1[name] = param.grad + 0
                param.grad.zero_()

        out2 = layer(1 - x)
        value_loss2 = loss(out2, 1 - y)
        value_loss2.backward()


        grads2 = {}
        for name, param in layer.named_parameters():
            if param.grad is not None:
                grads2[name] = param.grad + 0


        assert (value_loss1 - value_loss2).abs().sum() < 1e-5
        for name, param in layer.named_parameters():
            if param.grad is not None:
                if "bias" in name:
                    assert (grads1[name] + grads2[name]).abs().sum() < 1e-5
                else:
                    assert (grads1[name] - grads2[name]).abs().sum() < 1e-5
