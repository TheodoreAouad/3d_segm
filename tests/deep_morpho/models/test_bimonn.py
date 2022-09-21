from deep_morpho.models import BiMoNN


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
        model = BiMoNN(kernel_size=(7, 7), channels=[3, 5, 3, 3], atomic_element="bisel", bise_initializer_args={"input_mean": 0.3, "init_bias_value": 1})
        assert [k["input_mean"] for k in model.bise_initializer_args] == [0.3, 0.5, 0.5]

    @staticmethod
    def test_sybisel_input_mean():
        model = BiMoNN(kernel_size=(7, 7), channels=[3, 5, 3, 3], atomic_element="sybisel", input_mean=0.3)
        assert model.input_mean == [0.3, 0, 0]

    # @staticmethod
    # def test_mix_bisel_input_mean():
    #     model = BiMoNN(kernel_size=(7, 7), channels=[3, 5, 3, 3], atomic_element=["bisel", "sybisel", "bisel"], input_mean=0.3)
    #     assert model.input_mean == [0.3, 0.5, 0]
