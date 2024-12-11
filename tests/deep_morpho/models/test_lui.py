import numpy as np

from deep_morpho.models import LUI


class TestLUI:

    @staticmethod
    def test_init():
        """
        Test initialization of LUI layer.

        Args:
        """
        layer = LUI(threshold_mode='tanh', chan_inputs=10, chan_outputs=5)
        layer

    @staticmethod
    def test_union_check():
        """
        Check that the LUIC layer is correctly annotated with the union layer.

        Args:
        """
        C = np.random.randint(0, 2, 10)
        layer = LUI.from_set(C, 'union')
        Cpred, operation = layer.find_set_and_operation_chan(0)
        assert operation == "union"
        assert np.abs(C - Cpred).sum() == 0


    @staticmethod
    def test_intersection_check():
        """
        Test that the LUI layer intersects the LUI layer with the given set of channels.

        Args:
        """
        C = np.random.randint(0, 2, 10)
        layer = LUI.from_set(C, 'intersection')
        Cpred, operation = layer.find_set_and_operation_chan(0)
        assert operation == "intersection"
        assert np.abs(C - Cpred).sum() == 0
