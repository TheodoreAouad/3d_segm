from deep_morpho.datasets.mnist_dataset import MnistMorphoDataset, MnistClassifDataset
from deep_morpho.morp_operations import ParallelMorpOperations


class TestMnistMorphoDataset:

    @staticmethod
    def test_init_dataset():
        dataset = MnistMorphoDataset(n_inputs=10000, morp_operation=ParallelMorpOperations.erosion(('disk', 3)))
        dataset[0]

    @staticmethod
    def test_dataloader():
        dataloader = MnistMorphoDataset.get_loader(
            batch_size=20, first_idx=0, n_inputs=1000, train=True,
            morp_operation=ParallelMorpOperations.erosion(('disk', 3)),
        )
        for batch in dataloader:
            pass

    @staticmethod
    def test_train_val_test_loader():
        trainloader, valloader, testloader = MnistMorphoDataset.get_train_val_test_loader(
            n_inputs_train=1000, n_inputs_val=1000, n_inputs_test=1000,
            batch_size=20, morp_operation=ParallelMorpOperations.erosion(('disk', 3)),
        )
        for batch in trainloader:
            pass

        for batch in valloader:
            pass

        for batch in testloader:
            pass

    @staticmethod
    def test_dataset_multi_chans():
        morp_operation = ParallelMorpOperations(operations=[
            [
                [('dilation', ('hstick', 7), False), 'union'],
                [('dilation', ('vstick', 7), False), 'union'],
            ],
        ])
        dataset = MnistMorphoDataset(n_inputs=10000, morp_operation=morp_operation)
        dataset[0]


class TestMnistClassifDataset:

    @staticmethod
    def test_init_dataset():
        dataset = MnistClassifDataset(n_inputs=10000)
        dataset[0]

    @staticmethod
    def test_dataloader():
        dataloader = MnistClassifDataset.get_loader(
            batch_size=20, first_idx=0, n_inputs=1000, train=True,
        )
        for batch in dataloader:
            pass
