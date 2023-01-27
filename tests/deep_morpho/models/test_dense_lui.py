
import numpy as np
import torch

from deep_morpho.models import DenseLUI


class TestDenseLUI:

    @staticmethod
    def test_lui_init():
        layer = DenseLUI(threshold_mode='tanh', in_channels=10, out_channels=5)
        layer

    # @staticmethod
    # def test_lui_union_check():
    #     C = np.random.randint(0, 2, 10)
    #     layer = DenseLUI.from_set(C, 'union')
    #     Cpred, operation = layer.find_set_and_operation_chan(0)
    #     assert operation == "union"
    #     assert np.abs(C - Cpred).sum() == 0


    # @staticmethod
    # def test_lui_intersection_check():
    #     C = np.random.randint(0, 2, 10)
    #     layer = DenseLUI.from_set(C, 'intersection')
    #     Cpred, operation = layer.find_set_and_operation_chan(0)
    #     assert operation == "intersection"
    #     assert np.abs(C - Cpred).sum() == 0

    @staticmethod
    def test_dense_lui_binary_mode():
        layer = DenseLUI(threshold_mode='tanh', in_channels=10, out_channels=5)
        layer.binary()
        assert layer.binary_mode == True
        layer.binary(False)
        assert layer.binary_mode == False


    @staticmethod
    def test_dense_lui_forward():
        layer = DenseLUI(threshold_mode='tanh', in_channels=10, out_channels=5)
        x = torch.randint(0, 2, (5, 10)).float()

        layer.forward(x)
        # assert np.isin(output, [0, 1]).all()


    @staticmethod
    def test_dense_lui_forward_binary():
        layer = DenseLUI(threshold_mode='tanh', in_channels=10, out_channels=5)
        x = torch.randint(0, 2, (5, 10)).float()

        output = layer.forward_binary(x).detach().cpu().numpy()
        assert np.isin(output, [0, 1]).all()

    @staticmethod
    def test_dense_lui_forward_binary_mode():
        layer = DenseLUI(threshold_mode='tanh', in_channels=10, out_channels=5)
        x = torch.randint(0, 2, (5, 10)).float()

        layer.binary(True)
        output = layer.forward_binary(x).detach().cpu().numpy()

        assert np.isin(output, [0, 1]).all()

    # @staticmethod
    # def test_lui_binary_mode_identity():
    #     C = np.ones((1,))
    #     layer = DenseLUI.from_set(C, 'union')
    #     layer.binary()

    #     x = torch.randint(0, 2, (5, 1, 50, 50)).float()
    #     pred = layer(x)
    #     assert (x - pred).abs().sum() == 0
