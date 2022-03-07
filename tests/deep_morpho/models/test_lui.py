
import numpy as np
import torch

from deep_morpho.models import LUI


class TestLUI:

    @staticmethod
    def test_lui_init():
        layer = LUI(threshold_mode='tanh', chan_inputs=10, chan_outputs=5)
        layer

    @staticmethod
    def test_lui_union_check():
        C = np.random.randint(0, 2, 10)
        layer = LUI.from_set(C, 'union')
        Cpred, operation = layer.find_set_and_operation_chan(0)
        assert operation == "union"
        assert np.abs(C - Cpred).sum() == 0


    @staticmethod
    def test_lui_intersection_check():
        C = np.random.randint(0, 2, 10)
        layer = LUI.from_set(C, 'intersection')
        Cpred, operation = layer.find_set_and_operation_chan(0)
        assert operation == "intersection"
        assert np.abs(C - Cpred).sum() == 0

    @staticmethod
    def test_lui_binary_mode():
        layer = LUI(threshold_mode='tanh', chan_inputs=10, chan_outputs=5)
        layer.binary()
        assert layer.binary_mode == True
        layer.binary(False)
        assert layer.binary_mode == False

    @staticmethod
    def test_lui_forward_binary():
        layer = LUI(threshold_mode='tanh', chan_inputs=10, chan_outputs=5)
        x = torch.randint(0, 2, (5, 10, 50, 50)).float()

        output = layer.forward_binary(x).detach().cpu().numpy()
        assert np.isin(output, [0, 1]).all()


    @staticmethod
    def test_lui_binary_mode_identity():
        C = np.ones((1,))
        layer = LUI.from_set(C, 'union')
        layer.binary()

        x = torch.randint(0, 2, (5, 1, 50, 50)).float()
        pred = layer(x)
        assert (x - pred).abs().sum() == 0
