from os.path import join
from typing import Tuple, Any, Optional, Callable, Union
import cv2
# import numpy as np

from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np

from deep_morpho.morp_operations import ParallelMorpOperations
from deep_morpho.gray_scale import level_sets_from_gray, gray_from_level_sets
from deep_morpho.datasets.collate_fn_gray import collate_fn_gray_scale
from deep_morpho.tensor_with_attributes import TensorGray
from .gray_dataset import GrayScaleDataset
# from general.utils import set_borders_to


ROOT_MNIST_DIR = join('/', 'hdd', 'datasets', 'MNIST')


class MnistMorphoDataset(MNIST):

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
        super().__init__(root, train, *kwargs)
        self.morp_operation = morp_operation
        self.threshold = threshold
        self.preprocessing = preprocessing
        self.size = size
        self.invert_input_proba = invert_input_proba
        self.do_symetric_output = do_symetric_output

        if n_inputs != "all":
            self.data = self.data[first_idx:n_inputs+first_idx]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input_ = (
            cv2.resize(self.data[index].numpy(), (self.size[1], self.size[0]), interpolation=cv2.INTER_CUBIC)
            >= (self.threshold)
        )[..., None]

        if torch.rand(1) < self.invert_input_proba:
            input_ = 1 - input_

        # input_[..., 0] = set_borders_to(input_[..., 0], np.array(self.morp_operation.max_selem_shape[0]) // 2, value=0)

        target = torch.tensor(self.morp_operation(input_)).float()
        input_ = torch.tensor(input_).float()

        input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)
        target = target.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        # target = target != self.data['value_bg'].iloc[idx]

        if self.preprocessing is not None:
            input_ = self.preprocessing(input_)
            target = self.preprocessing(target)

        if self.do_symetric_output:
            return 2 * input_ - 1, 2 * target - 1

        return input_.float(), target.float()


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
        trainloader = MnistMorphoDataset.get_loader(first_idx=0, n_inputs=n_inputs_train, train=True, *args, **kwargs)
        valloader = MnistMorphoDataset.get_loader(first_idx=0, n_inputs=n_inputs_val, train=False, *args, **kwargs)
        testloader = MnistMorphoDataset.get_loader(first_idx=n_inputs_val, n_inputs=n_inputs_test, train=False, *args, **kwargs)
        return trainloader, valloader, testloader


class MnistGrayScaleDataset(MNIST, GrayScaleDataset):

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
        invert_input_proba: bool = 0,
        do_symetric_output: bool = False,
        **kwargs,
    ) -> None:
        super(MNIST).__init__(root, train, *kwargs)
        super(GrayScaleDataset).__init__(n_gray_scale_values)
        self.morp_operation = morp_operation
        self.preprocessing = preprocessing
        self.n_inputs = n_inputs
        # self.n_gray_scale_values = n_gray_scale_values
        self.size = size
        self.invert_input_proba = invert_input_proba
        self.do_symetric_output = do_symetric_output

        if n_inputs != "all":
            self.data = self.data[first_idx:n_inputs+first_idx]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input_ = (
            cv2.resize(self.data[index].numpy(), (self.size[1], self.size[0]), interpolation=cv2.INTER_CUBIC)
        )[..., None]

        # target = torch.tensor(self.morp_operation(input_)).float()
        # input_ = torch.tensor(input_).float()
        target = TensorGray(self.morp_operation(input_)).float()
        input_ = TensorGray(input_).float()

        input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)
        target = target.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        if self.preprocessing is not None:
            input_ = self.preprocessing(input_)
            target = self.preprocessing(target)

        original_input = input_.detach()
        original_target = target.detach()

        input_, target = self.level_sets_from_gray(input_, target)

        input_.original = original_input
        target.original = original_target

        if torch.rand(1) < self.invert_input_proba:
            input_ = 1 - input_


        if self.do_symetric_output:
            return 2 * input_ - 1, 2 * target - 1

        return input_.float(), target.float()
        # debug
        # return or_in, or_tar, input_.float(), target.float()


    @property
    def processed_folder(self) -> str:
        return join(self.root, 'processed')


    @staticmethod
    def get_loader(
        batch_size, n_inputs, morp_operation, train, first_idx=0, threshold=.5, size=(50, 50),
        invert_input_proba=0, do_symetric_output=False, preprocessing=None, n_gray_scale_values="all",
    **kwargs):
        if n_inputs == 0:
            return DataLoader([])
        return DataLoader(
            MnistGrayScaleDataset(
                morp_operation=morp_operation, n_inputs=n_inputs, first_idx=first_idx,
                train=train, threshold=threshold, preprocessing=preprocessing,
                size=size, invert_input_proba=invert_input_proba,
                do_symetric_output=do_symetric_output, n_gray_scale_values=n_gray_scale_values,
            ), batch_size=batch_size, collate_fn=collate_fn_gray_scale, **kwargs)

    @staticmethod
    def get_train_val_test_loader(n_inputs_train, n_inputs_val, n_inputs_test, *args, **kwargs):
        trainloader = MnistGrayScaleDataset.get_loader(first_idx=0, n_inputs=n_inputs_train, train=True, *args, **kwargs)
        valloader = MnistGrayScaleDataset.get_loader(first_idx=0, n_inputs=n_inputs_val, train=False, *args, **kwargs)
        testloader = MnistGrayScaleDataset.get_loader(first_idx=n_inputs_val, n_inputs=n_inputs_test, train=False, *args, **kwargs)
        return trainloader, valloader, testloader

    def gray_from_level_sets(self, ar: Union[np.ndarray, torch.Tensor], values: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.do_symetric_output:
            return gray_from_level_sets((ar > 0).float(), values)
        return gray_from_level_sets(ar, values)

    # def level_sets_from_gray(self, input_: torch.Tensor, target: torch.Tensor):
    #     input_ls, input_values = level_sets_from_gray(input_, n_values=self.n_gray_scale_values)
    #     target_ls, _ = level_sets_from_gray(target, input_values)
    #     input_ls.gray_values = input_values

    #     return input_ls, target_ls

    # @staticmethod
    # def gray_batch_from_level_sets_batch(batch_tensor: torch.Tensor, values: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    #     """Given an input batch of level set tensor, the corresponding values and the corresponding indexes, recovers
    #     the gray scale batch of tensor.
    #     Usually, `values` and `indexes` are attributes of the inputs (batch[0]).

    #     Args:
    #         batch_tensor (torch.Tensor): shape (sum_{batch size}{nb level sets} , 1 , W , L)
    #         indexes (torch.Tensor): shape (batch size + 1,)
    #         values (torch.Tensor): shape (sum_{batch size}{nb level sets},)

    #     Returns:
    #         torch.Tensor: shape (batch size , 1 , W , L)
    #     """

    #     final_tensor = []

    #     for idx in range(1, len(indexes)):
    #         idx1 = indexes[idx - 1]
    #         idx2 = indexes[idx]
    #         input_tensor = gray_from_level_sets(batch_tensor[idx1:idx2, 0], values=values[idx1:idx2])

    #         final_tensor.append(input_tensor)

    #     return torch.stack(final_tensor)[:, None, ...]

    # @staticmethod
    # def gray_from_level_sets_batch_idx(index: int, batch_tensor: torch.Tensor, values: torch.Tensor, indexes: torch.Tensor,) -> torch.Tensor:
    #     """Get a gray image from its index in the batch tensor. The index must be below batch_size.

    #     Args:
    #         index(int): index of the tensor inside the original batch tensor.
    #         batch_tensor (torch.Tensor): shape (sum_{batch size}{nb level sets} , 1 , W , L)
    #         indexes (torch.Tensor): shape (batch size + 1,)
    #         values (torch.Tensor): shape (sum_{batch size}{nb level sets},)

    #     Returns:
    #         torch.Tensor: shape (W , L)
    #     """
    #     return gray_from_level_sets(
    #         batch_tensor[indexes[index]:indexes[index + 1], 0],
    #         values=values[indexes[index]:indexes[index + 1]]
    #     )

    # @staticmethod
    # def get_relevent_tensors_idx(idx: int, batch: torch.Tensor, preds: torch.Tensor) -> Tuple[torch.Tensor]:
    #     """ Given the batch and the predictions, outputs all the useful tensors for a given idx.

    #     Args:
    #         idx (int): index of the original image
    #         batch (torch.Tensor): batch of level sets
    #         preds (torch.Tensor): preds of level sets

    #     Returns:
    #         Tuple[torch.Tensor]: reconstructed image, prediction, reconstructed target, original img, original target
    #     """
    #     img = MnistGrayScaleDataset.gray_from_level_sets_batch_idx(
    #         index=idx,
    #         batch_tensor=batch[0],
    #         values=batch[0].gray_values,
    #         indexes=batch[0].indexes,
    #     )
    #     target = MnistGrayScaleDataset.gray_from_level_sets_batch_idx(
    #         index=idx,
    #         batch_tensor=batch[1],
    #         values=batch[0].gray_values,
    #         indexes=batch[0].indexes,
    #     )
    #     pred = MnistGrayScaleDataset.gray_from_level_sets_batch_idx(
    #         index=idx,
    #         batch_tensor=preds,
    #         values=batch[0].gray_values,
    #         indexes=batch[0].indexes,
    #     )

    #     original_img = batch[0].original[idx, 0]
    #     original_target = batch[1].original[idx, 0]

    #     return img, pred, target, original_img, original_target


    # @staticmethod
    # def get_relevent_tensors_batch(batch: torch.Tensor, preds: torch.Tensor) -> Tuple[torch.Tensor]:
    #     """ Given the batch and the predictions, outputs all the useful tensors for a given idx.

    #     Args:
    #         idx (int): index of the original image
    #         batch (torch.Tensor): batch of level sets
    #         preds (torch.Tensor): preds of level sets

    #     Returns:
    #         Tuple[torch.Tensor]: reconstructed image, prediction, reconstructed target, original img, original target
    #     """
    #     img = MnistGrayScaleDataset.gray_batch_from_level_sets_batch(
    #         batch_tensor=batch[0],
    #         values=batch[0].gray_values,
    #         indexes=batch[0].indexes,
    #     )
    #     target = MnistGrayScaleDataset.gray_batch_from_level_sets_batch(
    #         batch_tensor=batch[1],
    #         values=batch[0].gray_values,
    #         indexes=batch[0].indexes,
    #     )
    #     pred = MnistGrayScaleDataset.gray_batch_from_level_sets_batch(
    #         batch_tensor=preds,
    #         values=batch[0].gray_values,
    #         indexes=batch[0].indexes,
    #     )

    #     original_img = batch[0].original
    #     original_target = batch[1].original

    #     return img, pred, target, original_img, original_target


class MnistClassifDataset(MNIST):

    def __init__(
        self,
        root: str = ROOT_MNIST_DIR,
        n_inputs: int = "all",
        threshold: float = 30,
        first_idx: int = 0,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        invert_input_proba: bool = 0,
        download: bool = False,
    ) -> None:
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.n_inputs = n_inputs
        self.first_idx = first_idx
        self.threshold = threshold
        self.invert_input_proba = invert_input_proba
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

        # target = target != self.data['value_bg'].iloc[idx]

        if self.transform is not None:
            input_ = self.transform(input_)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input_, target

    @property
    def processed_folder(self) -> str:
        return join(self.root, 'processed')

    @staticmethod
    def get_loader(batch_size, n_inputs, train, first_idx=0, threshold=.5, invert_input_proba=0, **kwargs):
        if n_inputs == 0:
            return DataLoader([])
        return DataLoader(
            MnistClassifDataset(
                n_inputs=n_inputs, first_idx=first_idx,
                train=train, threshold=threshold, invert_input_proba=invert_input_proba,
            ), batch_size=batch_size, **kwargs)

    @staticmethod
    def get_train_val_test_loader(n_inputs_train, n_inputs_val, n_inputs_test, *args, **kwargs):
        trainloader = MnistClassifDataset.get_loader(first_idx=0, n_inputs=n_inputs_train, train=True, shuffle=True, *args, **kwargs)
        valloader = MnistClassifDataset.get_loader(first_idx=0, n_inputs=n_inputs_val, train=False, shuffle=False, *args, **kwargs)
        testloader = MnistClassifDataset.get_loader(first_idx=n_inputs_val, n_inputs=n_inputs_test, train=False, shuffle=False, *args, **kwargs)
        return trainloader, valloader, testloader
