import numpy as np

from general.utils import one_hot_array


class TestOneHotArray():

    @staticmethod
    def test_two_values():
        ar = np.random.randint(0, 3, (50, 50))

        one_hot = one_hot_array(ar)

        assert (one_hot[..., 0] == (ar == 1)).all()
        assert (one_hot[..., 1] == (ar == 2)).all()
