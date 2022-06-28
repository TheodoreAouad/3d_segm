import torch

from deep_morpho.models import BiSEL2


class TestBiSEL:

    @staticmethod
    def test_init_bisel():
        layer = BiSEL2(
            in_channels=3,
            out_channels=2,
            kernel_size=(3, 3)
        )
        layer


    @staticmethod
    def test_inference_bisel():
        layer = BiSEL2(in_channels=3, out_channels=2, kernel_size=(3, 3))

        inpt = torch.rand(5, 3, 50, 50)
        outpt = layer(inpt)
        assert outpt.shape == (5, 2, 50, 50)

    @staticmethod
    def test_inference_bisel_identity_lui():
        layer = BiSEL2(in_channels=1, out_channels=1, kernel_size=(3, 3))

        inpt = torch.rand(1, 1, 50, 50)
        outpt = layer(inpt)
        assert outpt.shape == (1, 1, 50, 50)

    @staticmethod
    def test_inference_bisel_identity_lui2():
        layer = BiSEL2(in_channels=1, out_channels=3, kernel_size=(3, 3), lui_kwargs={"force_identity": True})

        inpt = torch.rand(5, 1, 50, 50)
        outpt = layer(inpt)
        assert outpt.shape == (5, 3, 50, 50)
