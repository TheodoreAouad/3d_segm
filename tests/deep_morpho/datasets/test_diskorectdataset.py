from black import generate_comments
from deep_morpho.datasets.generate_forms3 import get_random_diskorect_channels
from deep_morpho.datasets.diskorect_dataset import DiskorectDataset
from deep_morpho.morp_operations import ParallelMorpOperations
from deep_morpho.utils import set_seed


class TestInputOutputGeneratorDataset:

    @staticmethod
    def test_reproducibility1():
        seed = set_seed()
        morp_operation = ParallelMorpOperations.erosion(('disk', 3))

        dataset1 = DiskorectDataset(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02},
            morp_operation=morp_operation,
            len_dataset=10
        )

        batchs1 = []
        for idx in range(3):
            img, target = dataset1[idx]
            batchs1.append((img, target))

        set_seed(seed)
        morp_operation = ParallelMorpOperations.erosion(('disk', 3))
        dataset2 = DiskorectDataset(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02},
            morp_operation=morp_operation,
            len_dataset=10
        )

        batchs2 = []
        for idx in range(3):
            img, target = dataset2[idx]
            batchs2.append((img, target))

        set_seed(seed)
        morp_operation = ParallelMorpOperations.erosion(('disk', 3))
        dataset3 = DiskorectDataset(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02},
            morp_operation=morp_operation,
            len_dataset=10
        )

        batchs3 = []
        for idx in range(3):
            img, target = dataset3[idx]
            batchs3.append((img, target))

        for (img1, tar1), (img2, tar2), (img3, tar3) in zip(batchs1, batchs2, batchs3):
            assert (img1 == img2).all()
            assert (tar1 == tar2).all()
            assert (img1 == img3).all()
            assert (tar1 == tar3).all()

    @staticmethod
    def test_reproducibility2():
        morp_operation = ParallelMorpOperations.erosion(('disk', 3))
        seed = set_seed()

        dataset1 = DiskorectDataset(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02},
            morp_operation=morp_operation,
            len_dataset=10
        )

        batchs1 = []
        for idx in range(3):
            img, target = dataset1[idx]
            batchs1.append((img, target))

        set_seed(seed)
        dataset2 = DiskorectDataset(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02},
            morp_operation=morp_operation,
            len_dataset=10
        )

        batchs2 = []
        for idx in range(3):
            img, target = dataset2[idx]
            batchs2.append((img, target))

        set_seed(seed)
        dataset3 = DiskorectDataset(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02},
            morp_operation=morp_operation,
            len_dataset=10
        )



        batchs3 = []
        for idx in range(3):
            img, target = dataset3[idx]
            batchs3.append((img, target))

        for (img1, tar1), (img2, tar2), (img3, tar3) in zip(batchs1, batchs2, batchs3):
            assert (img1 == img2).all()
            assert (tar1 == tar2).all()
            assert (img1 == img3).all()
            assert (tar1 == tar3).all()

    @staticmethod
    def test_reproducibility_dataloader():
        morp_operation = ParallelMorpOperations.erosion(('disk', 3))
        seed = set_seed()

        dataloader1 = DiskorectDataset.get_loader(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02},
            morp_operation=morp_operation,
            n_inputs=9,
            batch_size=3,
            num_workers=3,
        )

        batchs1 = []
        for img, target in dataloader1:
            batchs1.append((img, target))

        set_seed(seed)
        dataloader2 = DiskorectDataset.get_loader(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02},
            morp_operation=morp_operation,
            n_inputs=9,
            batch_size=3,
            num_workers=3,
        )

        batchs2 = []
        for img, target in dataloader2:
            batchs2.append((img, target))


        set_seed(seed)
        dataloader3 = DiskorectDataset.get_loader(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02},
            morp_operation=morp_operation,
            n_inputs=9,
            batch_size=3,
            num_workers=3,
        )

        batchs3 = []
        for img, target in dataloader3:
            batchs3.append((img, target))

        assert len(batchs1) != 0
        for (img1, tar1), (img2, tar2), (img3, tar3) in zip(batchs1, batchs2, batchs3):
            assert (img1 == img2).all()
            assert (tar1 == tar2).all()
            assert (img1 == img3).all()
            assert (tar1 == tar3).all()

    @staticmethod
    def test_generate_one_batch():
        morp_operation = ParallelMorpOperations.erosion(('disk', 3))
        dataloader = DiskorectDataset.get_loader(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02},
            morp_operation=morp_operation,
            n_inputs=256,
            batch_size=256,
            max_generation_nb=256,
            num_workers=0,
        )

        assert len(dataloader) == 1
        inp1, tar1 = next(iter(dataloader))
        inp2, tar2 = next(iter(dataloader))

        assert (inp1 - inp2).abs().sum() == 0
        assert (tar1 - tar2).abs().sum() == 0

    @staticmethod
    def test_generate_multi_batch():
        morp_operation = ParallelMorpOperations.erosion(('disk', 3))
        dataloader = DiskorectDataset.get_loader(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02},
            morp_operation=morp_operation,
            n_inputs=256 * 10,
            batch_size=256,
            max_generation_nb=256,
            num_workers=0,
        )

        inp1, tar1 = next(iter(dataloader))
        assert (inp1[0] - inp1[1]).abs().sum() != 0  # proba of happening is really low
        for inp2, tar2 in dataloader:
            assert (inp1 - inp2).abs().sum() == 0
            assert (tar1 - tar2).abs().sum() == 0

    @staticmethod
    def test_two_batch_randomness():
        morp_operation = ParallelMorpOperations.erosion(('disk', 3))
        dataloader = DiskorectDataset.get_loader(
            random_gen_fn=get_random_diskorect_channels,
            random_gen_args={'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02},
            morp_operation=morp_operation,
            n_inputs=256 * 10,
            batch_size=256,
            max_generation_nb=0,
            num_workers=0,
        )

        inp1, tar1 = next(iter(dataloader))
        inp2, tar2 = next(iter(dataloader))

        assert (inp1 - inp2).abs().sum() != 0
