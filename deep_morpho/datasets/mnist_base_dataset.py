from typing import Tuple, Any, Union
# import cv2
# import numpy as np
from PIL import Image

import torch
import numpy as np

from deep_morpho.morp_operations import ParallelMorpOperations
# from deep_morpho.gray_scale import level_sets_from_gray, gray_from_level_sets
from deep_morpho.tensor_with_attributes import TensorGray
from .gray_dataset import GrayScaleDataset
# from general.utils import set_borders_to


def resize_image(img: np.ndarray, size: Tuple) -> np.ndarray:
    img_int8 = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    return np.array(Image.fromarray(img_int8).resize((size[1], size[0]), Image.Resampling.BICUBIC))


class MnistBaseDataset:

    def __init__(
        self,
        morp_operation: ParallelMorpOperations,
        threshold: float = 30,
        size=(50, 50),
        preprocessing=None,
        first_idx: int = 0,
        n_inputs: int = "all",
        invert_input_proba: bool = 0,
        do_symetric_output: bool = False,
        **kwargs,
    ) -> None:
        self.morp_operation = morp_operation
        self.threshold = threshold
        self.preprocessing = preprocessing
        self.size = size
        self.invert_input_proba = invert_input_proba
        self.do_symetric_output = do_symetric_output

        if n_inputs != "all":
            self.data = self.data[first_idx:n_inputs+first_idx]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input_ = (resize_image(self.data[index].numpy(), self.size) >= (self.threshold))[..., None]

        if torch.rand(1) < self.invert_input_proba:
            input_ = 1 - input_

        target = torch.tensor(self.morp_operation(input_)).float()
        input_ = torch.tensor(input_).float()

        input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)
        target = target.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        if self.preprocessing is not None:
            input_ = self.preprocessing(input_)
            target = self.preprocessing(target)

        if self.do_symetric_output:
            return 2 * input_ - 1, 2 * target - 1

        return input_.float(), target.float()


    # @property
    # def processed_folder(self) -> str:
    #     return join(self.root, 'processed')


    # @staticmethod
    # def get_loader(batch_size, n_inputs, morp_operation, train, first_idx=0, threshold=.5, size=(50, 50), invert_input_proba=0, do_symetric_output=False, preprocessing=None, **kwargs):
    #     if n_inputs == 0:
    #         return DataLoader([])
    #     return DataLoader(
    #         MnistMorphoDataset(
    #             morp_operation=morp_operation, n_inputs=n_inputs, first_idx=first_idx,
    #             train=train, threshold=threshold, preprocessing=preprocessing,
    #             size=size, invert_input_proba=invert_input_proba,
    #             do_symetric_output=do_symetric_output,
    #         ), batch_size=batch_size, **kwargs)

    # @staticmethod
    # def get_train_val_test_loader(n_inputs_train, n_inputs_val, n_inputs_test, *args, **kwargs):
    #     trainloader = MnistMorphoDataset.get_loader(first_idx=0, n_inputs=n_inputs_train, train=True, *args, **kwargs)
    #     valloader = MnistMorphoDataset.get_loader(first_idx=0, n_inputs=n_inputs_val, train=False, *args, **kwargs)
    #     testloader = MnistMorphoDataset.get_loader(first_idx=n_inputs_val, n_inputs=n_inputs_test, train=False, *args, **kwargs)
    #     return trainloader, valloader, testloader


class MnistGrayScaleBaseDataset(GrayScaleDataset):

    def __init__(
        self,
        morp_operation: ParallelMorpOperations,
        n_inputs: int = "all",
        n_gray_scale_values: str = "all",
        size=(50, 50),
        first_idx: int = 0,
        preprocessing=None,
        do_symetric_output: bool = False,
        **kwargs,
    ) -> None:
        GrayScaleDataset.__init__(self, n_gray_scale_values)
        self.morp_operation = morp_operation
        self.preprocessing = preprocessing
        self.n_inputs = n_inputs
        # self.n_gray_scale_values = n_gray_scale_values
        self.size = size
        self.do_symetric_output = do_symetric_output

        if n_inputs != "all":
            self.data = self.data[first_idx:n_inputs+first_idx]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # input_ = (
        #     cv2.resize(self.data[index].numpy(), (self.size[1], self.size[0]), interpolation=cv2.INTER_CUBIC)
        # )[..., None]
        input_ = (resize_image(self.data[index].numpy(), self.size))[..., None]

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

        if self.do_symetric_output:
            gray_values = input_.gray_values
            input_ = 2 * input_ - 1
            target = 2 * target - 1
            input_.gray_values = gray_values

        input_.original = original_input
        target.original = original_target

        return input_.float(), target.float()


    # @property
    # def processed_folder(self) -> str:
    #     return join(self.root, 'processed')


    # @staticmethod
    # def get_loader(
    #     batch_size, n_inputs, morp_operation, train, first_idx=0, size=(50, 50),
    #     do_symetric_output=False, preprocessing=None, n_gray_scale_values="all",
    # **kwargs):
    #     if n_inputs == 0:
    #         return DataLoader([])
    #     return DataLoader(
    #         MnistGrayScaleDataset(
    #             morp_operation=morp_operation, n_inputs=n_inputs, first_idx=first_idx,
    #             train=train, preprocessing=preprocessing, size=size,
    #             do_symetric_output=do_symetric_output, n_gray_scale_values=n_gray_scale_values,
    #         ), batch_size=batch_size, collate_fn=collate_fn_gray_scale, **kwargs)

    # @staticmethod
    # def get_train_val_test_loader(n_inputs_train, n_inputs_val, n_inputs_test, *args, **kwargs):
    #     trainloader = MnistGrayScaleDataset.get_loader(first_idx=0, n_inputs=n_inputs_train, train=True, *args, **kwargs)
    #     valloader = MnistGrayScaleDataset.get_loader(first_idx=0, n_inputs=n_inputs_val, train=False, *args, **kwargs)
    #     testloader = MnistGrayScaleDataset.get_loader(first_idx=n_inputs_val, n_inputs=n_inputs_test, train=False, *args, **kwargs)
    #     return trainloader, valloader, testloader

    def gray_from_level_sets(self, ar: Union[np.ndarray, torch.Tensor], values: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.do_symetric_output:
            return super().gray_from_level_sets((ar > 0).float(), values)
        return super().gray_from_level_sets(ar, values)
