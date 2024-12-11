import torch

from deep_morpho.models import BiSEL


class TestBiSEL:

    @staticmethod
    def test_init_bisel():
        """
        Test initialization of BiSEL layer.

        Args:
        """
        layer = BiSEL(
            in_channels=3,
            out_channels=2,
            kernel_size=(3, 3)
        )
        layer


    @staticmethod
    def test_inference_bisel():
        """
        Test the BiSEL inference layer

        Args:
        """
        layer = BiSEL(in_channels=3, out_channels=2, kernel_size=(3, 3))

        inpt = torch.rand(5, 3, 50, 50)
        outpt = layer(inpt)
        assert outpt.shape == (5, 2, 50, 50)
