import numpy as np

from general.utils import one_hot_array, recursive_dict_copy


class TestOneHotArray():

    @staticmethod
    def test_two_values():
        ar = np.random.randint(0, 3, (50, 50))

        one_hot = one_hot_array(ar)

        assert (one_hot[..., 0] == (ar == 1)).all()
        assert (one_hot[..., 1] == (ar == 2)).all()


class TestRecursiveDictCopy:
    @staticmethod
    def test_copy():
        d = {'a': 1, 'b': {'c': 2}}
        d2 = recursive_dict_copy(d)
        assert d == d2
        assert d is not d2
        assert d['b'] is not d2['b']
    
    @staticmethod
    def test_changes():
        d = {'a': 1, 'b': {'c': 2}}
        d2 = recursive_dict_copy(d)
        d2['b']['c'] = 3
        assert d['b']['c'] == 2
        assert d2['b']['c'] == 3
