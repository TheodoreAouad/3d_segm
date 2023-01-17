from os.path import join
from typing import Tuple, Any, Optional, Callable
import warnings

from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
import torch

from deep_morpho.morp_operations import ParallelMorpOperations
from deep_morpho.datasets.collate_fn_gray import collate_fn_gray_scale
from .mnist_base_dataset import MnistBaseDataset, MnistGrayScaleBaseDataset

with open('deep_morpho/datasets/root_mnist_dir.txt', 'r') as f:
    ROOT_MNIST_DIR = f.read()


class MnistMorphoDataset(MnistBaseDataset, MNIST):

    def __init__(
        self,
        morp_operation: ParallelMorpOperations,
        n_inputs: int = "all",
        threshold: float = 30,
        size=(50, 50),
        first_idx: int = 0,
        preprocessing=None,
        root: str = ROOT_MNIST_DIR,
        train: bool = True,
        invert_input_proba: bool = 0,
        do_symetric_output: bool = False,
        **kwargs,
    ) -> None:
        MNIST.__init__(self, root, train, **kwargs)
        MnistBaseDataset.__init__(
            self,
            morp_operation=morp_operation,
            n_inputs=n_inputs,
            threshold=threshold,
            size=size,
            first_idx=first_idx,
            preprocessing=preprocessing,
            invert_input_proba=invert_input_proba,
            do_symetric_output=do_symetric_output,
        )

    @property
    def processed_folder(self) -> str:
        return join(self.root, 'processed')


    @staticmethod
    def get_loader(batch_size, n_inputs, morp_operation, train, first_idx=0, threshold=.5, size=(50, 50), invert_input_proba=0, do_symetric_output=False, preprocessing=None, **kwargs):
        if n_inputs == 0:
            return DataLoader([])
        return DataLoader(
            MnistMorphoDataset(
                morp_operation=morp_operation, n_inputs=n_inputs, first_idx=first_idx,
                train=train, threshold=threshold, preprocessing=preprocessing,
                size=size, invert_input_proba=invert_input_proba,
                do_symetric_output=do_symetric_output,
            ), batch_size=batch_size, **kwargs)

    @staticmethod
    def get_train_val_test_loader(n_inputs_train, n_inputs_val, n_inputs_test, *args, **kwargs):
        trainloader = MnistMorphoDataset.get_loader(first_idx=0, n_inputs=n_inputs_train, train=True, shuffle=True, *args, **kwargs)
        valloader = MnistMorphoDataset.get_loader(first_idx=0, n_inputs=n_inputs_val, train=False, shuffle=False, *args, **kwargs)
        testloader = MnistMorphoDataset.get_loader(first_idx=n_inputs_val, n_inputs=n_inputs_test, train=False, shuffle=False, *args, **kwargs)
        return trainloader, valloader, testloader


class MnistGrayScaleDataset(MnistGrayScaleBaseDataset, MNIST):

    def __init__(
        self,
        morp_operation: ParallelMorpOperations,
        n_inputs: int = "all",
        n_gray_scale_values: str = "all",
        size=(50, 50),
        first_idx: int = 0,
        preprocessing=None,
        root: str = ROOT_MNIST_DIR,
        train: bool = True,
        do_symetric_output: bool = False,
        **kwargs,
    ) -> None:
        MNIST.__init__(self, root, train, **kwargs)
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
            MnistGrayScaleDataset(
                morp_operation=morp_operation, n_inputs=n_inputs, first_idx=first_idx,
                train=train, preprocessing=preprocessing, size=size,
                do_symetric_output=do_symetric_output, n_gray_scale_values=n_gray_scale_values,
            ), batch_size=batch_size, collate_fn=collate_fn_gray_scale, **kwargs)

    @staticmethod
    def get_train_val_test_loader(n_inputs_train, n_inputs_val, n_inputs_test, *args, **kwargs):
        trainloader = MnistGrayScaleDataset.get_loader(first_idx=0, n_inputs=n_inputs_train, train=True, shuffle=True, *args, **kwargs)
        valloader = MnistGrayScaleDataset.get_loader(first_idx=0, n_inputs=n_inputs_val, train=False, shuffle=False, *args, **kwargs)
        testloader = MnistGrayScaleDataset.get_loader(first_idx=n_inputs_val, n_inputs=n_inputs_test, train=False, shuffle=False, *args, **kwargs)
        return trainloader, valloader, testloader


    @property
    def processed_folder(self) -> str:
        return join(self.root, 'processed')


class MnistClassifDataset(MNIST):

    def __init__(
        self,
        root: str = ROOT_MNIST_DIR,
        n_inputs: int = "all",
        threshold: float = 30,
        first_idx: int = 0,
        train: bool = True,
        invert_input_proba: float = 0,
        preprocessing=None,
        do_symetric_output: bool = False,
        size=None,
        **kwargs
    ) -> None:
        super().__init__(root=root, train=train, **kwargs)
        self.n_inputs = n_inputs
        self.first_idx = first_idx
        self.threshold = threshold
        self.invert_input_proba = invert_input_proba
        self.n_classes = 10
        self.preprocessing = preprocessing
        self.do_symetric_output = do_symetric_output
        
        warnings.warn("Size not used yet on classif.")
        self.size = size  # WARNING: not used

        if n_inputs != "all":
            self.data = self.data[first_idx:n_inputs+first_idx]
            self.targets = self.targets[first_idx:n_inputs+first_idx]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input_ = (self.data[index].numpy() >= (self.threshold))[..., None]
        target_int = int(self.targets[index])
        target = torch.zeros(10)
        target[target_int] = 1

        if torch.rand(1) < self.invert_input_proba:
            input_ = 1 - input_

        input_ = torch.tensor(input_).float()

        input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        if self.transform is not None:
            input_ = self.transform(input_)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.do_symetric_output:
            input_ = 2 * input_ - 1

        return input_, target

    @property
    def processed_folder(self) -> str:
        return join(self.root, 'processed')

    @staticmethod
    def get_loader(
        batch_size, n_inputs, train, preprocessing, first_idx=0,
        threshold=.5, invert_input_proba=0, do_symetric_output=False, 
        size=(28, 28), **kwargs):
        if n_inputs == 0:
            return DataLoader([])
        return DataLoader(
            MnistClassifDataset(
                n_inputs=n_inputs, first_idx=first_idx,
                train=train, threshold=threshold, invert_input_proba=invert_input_proba,
                preprocessing=preprocessing, do_symetric_output=do_symetric_output, size=size,
            ), batch_size=batch_size, **kwargs)

    @staticmethod
    def get_train_val_test_loader(n_inputs_train, n_inputs_val, n_inputs_test, *args, **kwargs):
        trainloader = MnistClassifDataset.get_loader(first_idx=0, n_inputs=n_inputs_train, train=True, shuffle=True, *args, **kwargs)
        valloader = MnistClassifDataset.get_loader(first_idx=0, n_inputs=n_inputs_val, train=False, shuffle=False, *args, **kwargs)
        testloader = MnistClassifDataset.get_loader(first_idx=n_inputs_val, n_inputs=n_inputs_test, train=False, shuffle=False, *args, **kwargs)
        return trainloader, valloader, testloader
