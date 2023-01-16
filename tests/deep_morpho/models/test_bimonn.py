import torch

from deep_morpho.models import BiMoNN, BiMoNNClassifierMaxPool
from deep_morpho.initializer import InitBiseEnum


class TestBimonn:

    @staticmethod
    def test_bisel_init():
        model = BiMoNN(kernel_size=(7, 7), channels=[3, 5, 3], atomic_element="bisel")
        model

    @staticmethod
    def test_sybisel_init():
        model = BiMoNN(kernel_size=(7, 7), channels=[3, 5, 3], atomic_element="sybisel")
        model

    @staticmethod
    def test_bisel_multi_kernel():
        model = BiMoNN(kernel_size=[(7, 7), (3, 3)], channels=[3, 5, 3], atomic_element="bisel")
        assert model.kernel_size == [(7, 7), (3, 3)]

    @staticmethod
    def test_bisel_one_kernel():
        model = BiMoNN(kernel_size=(7, 3), channels=[3, 5, 3], atomic_element="bisel")
        assert model.kernel_size == [(7, 3), (7, 3)]

    @staticmethod
    def test_bisel_input_mean():
        model = BiMoNN(kernel_size=(7, 7), channels=[3, 5, 3, 3], atomic_element="bisel",
            initializer_args={"input_mean": 0.3, "bise_init_args": {"init_bias_value": 1}, "bise_init_method": InitBiseEnum.CUSTOM_HEURISTIC})
        assert [k.bise_initializer.input_mean for k in model.bisel_initializers] == [0.3, 0.5, 0.5]
        assert [k.lui_initializer.input_mean for k in model.bisel_initializers] == [0.5, 0.5, 0.5]

    @staticmethod
    def test_sybisel_input_mean():
        model = BiMoNN(kernel_size=(7, 7), channels=[3, 5, 3, 3], atomic_element="sybisel",
            initializer_args={"input_mean": 0.3, "bise_init_args": {"mean_weight": "auto"}, "bise_init_method": InitBiseEnum.CUSTOM_CONSTANT})
        assert [k.bise_initializer.input_mean for k in model.bisel_initializers] == [0.3, 0, 0]
        assert [k.lui_initializer.input_mean for k in model.bisel_initializers] == [0, 0, 0]

    # @staticmethod
    # def test_mix_bisel_input_mean():
    #     model = BiMoNN(kernel_size=(7, 7), channels=[3, 5, 3, 3], atomic_element=["bisel", "sybisel", "bisel"], input_mean=0.3)
    #     assert model.input_mean == [0.3, 0.5, 0]


class TestBimonnClassifierMaxPool():

    @staticmethod
    def test_init():
        model = BiMoNNClassifierMaxPool(
            kernel_size=(7, 7),
            channels=[3, 5, 3],
            atomic_element='bisel',
            input_size=(50, 50),
            n_classes=10,
        )
        model


    @staticmethod
    def test_forward():
        x = torch.ones((3, 1, 50, 50))
        n_classes = 10

        model = BiMoNNClassifierMaxPool(
            kernel_size=(7, 7),
            channels=[x.shape[1], 5, 3, 3],
            atomic_element='bisel',
            input_size=x.shape[-2:],
            n_classes=n_classes,
        )

        otp = model(x)
        assert otp.shape == (x.shape[0], n_classes)

    @staticmethod
    def test_forward_shallow():
        x = torch.ones((3, 1, 50, 50))
        n_classes = 10

        model = BiMoNNClassifierMaxPool(
            kernel_size=(7, 7),
            channels=[x.shape[1], 5],
            atomic_element='bisel',
            input_size=x.shape[-2:],
            n_classes=n_classes,
        )

        otp = model(x)
        assert otp.shape == (x.shape[0], n_classes)

    @staticmethod
    def test_forward_odd():
        x = torch.ones((3, 1, 51, 51))
        n_classes = 10

        model = BiMoNNClassifierMaxPool(
            kernel_size=(7, 7),
            channels=[x.shape[1], 5, 3, 3],
            atomic_element='bisel',
            input_size=x.shape[-2:],
            n_classes=n_classes,
        )

        otp = model(x)
        assert otp.shape == (x.shape[0], n_classes)
