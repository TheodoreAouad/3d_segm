from os.path import join

from torchvision.datasets import FashionMNIST
from torch.utils.data.dataloader import DataLoader

from deep_morpho.morp_operations import ParallelMorpOperations
from deep_morpho.datasets.collate_fn_gray import collate_fn_gray_scale
from .mnist_base_dataset import MnistGrayScaleBaseDataset

with open('deep_morpho/datasets/root_fashionmnist_dir.txt', 'r') as f:
    ROOT_FASHIONMNIST_DIR = f.read()


class FashionMnistGrayScaleDataset(MnistGrayScaleBaseDataset, FashionMNIST):

    def __init__(
        self,
        morp_operation: ParallelMorpOperations,
        n_inputs: int = "all",
        n_gray_scale_values: str = "all",
        size=(50, 50),
        first_idx: int = 0,
        preprocessing=None,
        root: str = ROOT_FASHIONMNIST_DIR,
        train: bool = True,
        do_symetric_output: bool = False,
        **kwargs,
    ) -> None:
        FashionMNIST.__init__(self, root, train, **kwargs)
        MnistGrayScaleBaseDataset.__init__(
            self,
            morp_operation=morp_operation,
            n_inputs=n_inputs,
            n_gray_scale_values=n_gray_scale_values,
            size=size,
            first_idx=first_idx,
            preprocessing=preprocessing,
            do_symetric_output=do_symetric_output,
        )

    @staticmethod
    def get_loader(
        batch_size, n_inputs, morp_operation, train, first_idx=0, size=(50, 50),
        do_symetric_output=False, preprocessing=None, n_gray_scale_values="all",
    **kwargs):
        if n_inputs == 0:
            return DataLoader([])
        return DataLoader(
            FashionMnistGrayScaleDataset(
                morp_operation=morp_operation, n_inputs=n_inputs, first_idx=first_idx,
                train=train, preprocessing=preprocessing, size=size,
                do_symetric_output=do_symetric_output, n_gray_scale_values=n_gray_scale_values,
            ), batch_size=batch_size, collate_fn=collate_fn_gray_scale, **kwargs)

    @staticmethod
    def get_train_val_test_loader(n_inputs_train, n_inputs_val, n_inputs_test, *args, **kwargs):
        trainloader = FashionMnistGrayScaleDataset.get_loader(first_idx=0, n_inputs=n_inputs_train, train=True, *args, **kwargs)
        valloader = FashionMnistGrayScaleDataset.get_loader(first_idx=0, n_inputs=n_inputs_val, train=False, *args, **kwargs)
        testloader = FashionMnistGrayScaleDataset.get_loader(first_idx=n_inputs_val, n_inputs=n_inputs_test, train=False, *args, **kwargs)
        return trainloader, valloader, testloader


    @property
    def raw_folder(self) -> str:
        return join(self.root, "raw")
