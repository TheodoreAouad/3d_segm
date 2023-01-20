import torch

from deep_morpho.models.bias_layer import BiasBise
from deep_morpho.models import BiSE, BiseBiasOptimEnum


class TestBiasBise:

    @staticmethod
    def test_bias_bise_init():
        model = BiSE(kernel_size=3, bias_optim_mode=BiseBiasOptimEnum.RAW)
        model

    @staticmethod
    def test_bias_bise_get_parameters():
        model = BiSE(kernel_size=3, bias_optim_mode=BiseBiasOptimEnum.RAW)
        for param in model.parameters():
            pass



class TestBiasBiasBiseSoftplusProjected:

    @staticmethod
    def test_init():
        model = BiSE(kernel_size=3, bias_optim_mode=BiseBiasOptimEnum.POSITIVE_INTERVAL_PROJECTED)
        model.bias


    @staticmethod
    def test_forward():
        model = BiSE(kernel_size=3, bias_optim_mode=BiseBiasOptimEnum.POSITIVE_INTERVAL_PROJECTED)
        x = torch.ones(3, 1, 50, 50)
        model(x)


    @staticmethod
    def test_projection():
        model = BiSE(kernel_size=3, bias_optim_mode=BiseBiasOptimEnum.POSITIVE_INTERVAL_PROJECTED)
        bmin, bmax = model.bias_handler.get_min_max_intrinsic_bias_values()

        model.bias_handler.set_bias(torch.zeros_like(model.bias) - bmin / 2)
        assert model.bias == -bmin

        model.bias_handler.set_bias(torch.zeros_like(model.bias) - 2 * bmax)
        assert model.bias == -bmax

        bvalue = (bmax + bmin) / 2
        model.bias_handler.set_bias(torch.zeros_like(model.bias) - bvalue)
        assert model.bias == -bvalue

    @staticmethod
    def test_grad_non_zero():
        model = BiSE(kernel_size=3, bias_optim_mode=BiseBiasOptimEnum.POSITIVE_INTERVAL_PROJECTED)
        with torch.no_grad():
            bmin, bmax = model.bias_handler.get_min_max_intrinsic_bias_values()

        model.bias_handler.set_bias(torch.zeros_like(model.bias) - bmin / 2)
        model.bias.sum().backward()
        assert model.bias_handler.grad != 0

        model.bias_handler.param.grad.zero_()

        model.bias_handler.set_bias(torch.zeros_like(model.bias) - 2 * bmax)
        model.bias.sum().backward()
        assert model.bias_handler.grad != 0

        model.bias_handler.param.grad.zero_()

        bvalue = (bmax + bmin) / 2
        model.bias_handler.set_bias(torch.zeros_like(model.bias) - bvalue)
        model.bias.sum().backward()
        assert model.bias_handler.grad != 0
