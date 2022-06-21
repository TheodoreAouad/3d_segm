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




# TestBiasBise.test_bias_bise_init()