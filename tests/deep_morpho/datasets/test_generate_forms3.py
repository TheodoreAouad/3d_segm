from deep_morpho.datasets.generate_forms3 import get_random_diskorect_channels

from deep_morpho.utils import set_seed


class TestGetRandomDiskorectChannels:

    @staticmethod
    def test_reproducibility():
        random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02}
        seed = set_seed()
        img1 = get_random_diskorect_channels(**random_gen_args)

        set_seed(seed)
        img2 = get_random_diskorect_channels(**random_gen_args)

        set_seed(seed)
        img3 = get_random_diskorect_channels(**random_gen_args)

        assert (img1 == img2).all()
        assert (img3 == img2).all()
