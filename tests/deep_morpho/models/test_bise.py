import torch
import torch.nn as nn
import numpy as np
import pytest
from skimage.morphology import dilation, erosion, opening, closing, disk

from deep_morpho.initializer.bise_initializer import InitBiseConstantVarianceWeights, InitBiseHeuristicWeights, InitDualBiseConstantVarianceWeights
from deep_morpho.models import BiSE, InitBiseEnum, BiseBiasOptimEnum
from deep_morpho.datasets import InputOutputGeneratorDataset, get_random_diskorect_channels
from deep_morpho.models.bise_base import BiseWeightsOptimEnum
from deep_morpho.initializer import InitBiseEllipseWeightsRoot
from general.structuring_elements import disk
from general.array_morphology import array_erosion, array_dilation
from deep_morpho.morp_operations import ParallelMorpOperations
from general.array_morphology import array_erosion, array_dilation, array_union_chans, array_intersection_chans


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
    def test_bise_forward_odd():
        layer = BiSE((7, 7), threshold_mode='sigmoid')
        inpt = torch.rand((1, 1, 51, 51))
        otp = layer(inpt)
        assert otp.shape == (1, 1, 51, 51)


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
        layer.set_weights_param(torch.FloatTensor(100*(weight - 0.5))[None, None, ...])
        layer.set_bias(-torch.FloatTensor([weight.sum() - 1/2]))
        assert layer.is_erosion_by(layer._normalized_weight[0, 0], layer.bias, weight, v1=0.003, v2=0.997)

    @staticmethod
    def test_bise_dilation_check():
        layer = BiSE((7, 7), threshold_mode='sigmoid')
        weight = disk(3)
        layer.set_weights_param(torch.FloatTensor(10*(weight - 0.5))[None, None, ...])
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
    def test_bise_numel_binary():
        layer = BiSE((7, 7), out_channels=1)
        n_binary = layer.numel_binary()
        n_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        assert n_binary == n_params

    @staticmethod
    @pytest.mark.parametrize("input_mean", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_init_mode_custom_constant(input_mean):
        layer = BiSE(
            (7, 7),
            # initializer_method=InitBiseEnum.CUSTOM_CONSTANT,
            # initializer_args={"input_mean": input_mean, "init_bias_value": "auto"},
            initializer=InitBiseConstantVarianceWeights(input_mean=input_mean, init_bias_value="auto")
            # init_bias_value="auto",
            # input_mean=input_mean,
        )
        layer

    @staticmethod
    def test_ellipse_device():
        model = BiSE(
            kernel_size=(7, 7),
            initializer=InitBiseEllipseWeightsRoot(init_bias_value=1),
            weights_optim_mode=BiseWeightsOptimEnum.ELLIPSE_ROOT,
            bias_optim_mode=BiseBiasOptimEnum.POSITIVE,
            out_channels=3,
        )

        model.cuda()
        assert model.weights_handler.param.is_cuda
        assert model.weights_handler.sigma_inv.is_cuda
        assert model._normalized_weight.is_cuda


class TestBiseProperties:

    # deprecated
    @staticmethod
    def test_grad_bias():
        model = BiSE(
            kernel_size=(7, 7),
            initializer=InitBiseConstantVarianceWeights(input_mean=.5),
            weights_optim_mode=BiseWeightsOptimEnum.THRESHOLDED,
            bias_optim_mode=BiseBiasOptimEnum.POSITIVE,
        )

        loss = nn.BCELoss()

        dataset = InputOutputGeneratorDataset(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02, "border": (0, 0)},
            morp_operation=ParallelMorpOperations.dilation(('disk', 2)),
            len_dataset=1000,
            seed=100,
            do_symetric_output=False,
        )

        x, _ = dataset[0]
        y = array_erosion(x[0], disk(2), return_numpy_array=False).float()

        loss1 = loss(model(x[None, ...]), y[None, None, ...])
        loss1.backward()

        grad1 = model.bias_handler.grad + 0

        model.zero_grad()
        y = array_dilation(1 - x[0], disk(2), return_numpy_array=False).float()
        loss2 = loss(model(1 - x[None, ...]), y[None, None, ...])
        loss2.backward()

        grad2 = model.bias_handler.grad + 0

        assert (grad1 + grad2).abs().sum() < 1e-6

    @staticmethod
    def test_bise_duality():
        model = BiSE(
            kernel_size=(7, 7),
            initializer=InitBiseConstantVarianceWeights(input_mean=.5),
            weights_optim_mode=BiseWeightsOptimEnum.THRESHOLDED,
            bias_optim_mode=BiseBiasOptimEnum.POSITIVE,
        )

        model.set_bias(torch.tensor([-1]))

        dataset = InputOutputGeneratorDataset(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02, "border": (0, 0)},
            morp_operation=ParallelMorpOperations.dilation(('disk', 2)),
            len_dataset=1000,
            seed=100,
            do_symetric_output=False,
        )

        x, _ = dataset[0]

        otp1 = model(1 - x)

        model.set_bias(-(model._normalized_weight.sum() + model.bias))

        otp2 = model(x)

        assert (otp1 - (1 - otp2)).abs().mean() < 1e-6


    @staticmethod
    def test_duality_training():
        def test_surrogate(bise_optim_mode, dual_ops):
            layer = BiSE(
                kernel_size=(7, 7),
                threshold_mode={'weight': 'softplus', 'activation': 'tanh'},
                activation_P=1,
                initializer=InitDualBiseConstantVarianceWeights(input_mean=.5,),
                out_channels=1,
                bias_optim_mode=bise_optim_mode,
                bias_optim_args={"offset": 0},
                weights_optim_mode=BiseWeightsOptimEnum.NORMALIZED,
            )

            # dataset = InputOutputGeneratorDataset(
            #     random_gen_fn=get_random_diskorect_channels,
            #     random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02, "border": (0, 0)},
            #     morp_operation=ParallelMorpOperations.dilation(('disk', 2)),
            #     len_dataset=1000,
            #     seed=100,
            #     do_symetric_output=False,
            # )

            selem = disk(2)
            x = get_random_diskorect_channels(
                **{'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02, "border": (0, 0)}
            )[..., 0]

            op_dil, op_ero = dual_ops

            y_dil = op_dil(x, selem)
            y_ero = op_ero(1 - x, selem)

            x_dil = torch.tensor(x)[None, None, ...].float()
            y_dil = torch.tensor(y_dil)[None, None, ...].float()

            x_ero = torch.tensor(1 - x)[None, None, ...].float()
            y_ero = torch.tensor(y_ero)[None, None, ...].float()

            loss = nn.BCELoss()

            out_dil = layer(x_dil)
            loss_dil = loss(out_dil, y_dil)
            loss_dil.backward()

            grads_dil = {}
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    grads_dil[name] = param.grad + 0
                    param.grad.zero_()


            out_ero = layer(x_ero)
            loss_ero = loss(out_ero, y_ero)
            loss_ero.backward()

            grads_ero = {}
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    grads_ero[name] = param.grad + 0


            assert (loss_dil - loss_ero).abs().sum() < 1e-5
            nb_params = 0
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    if "bias" in name:
                        assert ((grads_dil[name] + grads_ero[name]) / grads_dil[name]).abs().mean() < 1e-3
                    else:
                        assert ((grads_dil[name] - grads_ero[name]) / grads_dil[name]).abs().mean() < 1e-3
                    nb_params += 1
            assert nb_params > 0

        for bise_optim_mode in [
            BiseBiasOptimEnum.RAW, BiseBiasOptimEnum.POSITIVE, BiseBiasOptimEnum.POSITIVE_INTERVAL_PROJECTED, BiseBiasOptimEnum.POSITIVE_INTERVAL_REPARAMETRIZED
        ]:
            for dual_ops in [(dilation, erosion), (opening, closing)]:
                test_surrogate(bise_optim_mode, dual_ops)
