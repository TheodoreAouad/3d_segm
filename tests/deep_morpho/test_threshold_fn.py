import torch


from deep_morpho.threshold_fn import (
    arctan_threshold,
    arctan_threshold_inverse,
    tanh_threshold,
    tanh_threshold_inverse,
    sigmoid_threshold,
    sigmoid_threshold_inverse
)


class TestArctanInverse:

    @staticmethod
    def test_inverse():
        x = torch.rand(1)
        xout = arctan_threshold(arctan_threshold_inverse(x))
        assert x == xout


class TestTanhInverse:

    @staticmethod
    def test_inverse():
        x = torch.rand(1)
        xout = tanh_threshold(tanh_threshold_inverse(x))
        assert x == xout


class TestSigmoidInverse:

    @staticmethod
    def test_inverse():
        x = torch.rand(1)
        xout = sigmoid_threshold(sigmoid_threshold_inverse(x))
        assert x == xout
