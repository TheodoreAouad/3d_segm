from typing import Tuple, Dict

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from skimage.morphology import disk
from .generate_forms import random_multi_rect
from general.array_morphology import array_dilation, array_erosion


def get_loader(batch_size, n_inputs, size, n_rectangles, device='cpu', **kwargs):
    return DataLoader(
        MultiRectDataset(size=size, n_rectangles=n_rectangles, device=device, len_dataset=n_inputs, **kwargs),
        batch_size=batch_size,
    )


class MultiRectDataset(Dataset):

    def __init__(
            self,
            size: Tuple,
            n_rectangles: int,
            max_shape: Tuple[int] = None,
            return_rects: bool = False,
            first_rect_args: Dict = {"centered": True},
            selem: "np.ndarray" = disk(3),
            morp_operation: str = "dilation",
            device: str = "cpu",
            len_dataset: int = 1000,
    ):
        self.size = size
        self.n_rectangles = n_rectangles
        self.max_shape = max_shape
        self.return_rects = return_rects
        self.first_rect_args = first_rect_args
        self.selem = selem
        self.device = device
        self.len_dataset = len_dataset
        self.morp_operation = morp_operation

        if morp_operation == 'dilation':
            self.morp_fn = array_dilation
        elif morp_operation == 'erosion':
            self.morp_fn = array_erosion

    @property
    def random_gen_args(self):
        return {attr: getattr(self, attr) for attr in ['size', 'n_rectangles', 'max_shape', 'return_rects', 'first_rect_args']}

    def __getitem__(self, idx):
        input_ = random_multi_rect(**self.random_gen_args)
        target = self.morp_fn(input_, self.selem, device=self.device, return_numpy_array=False).float()
        # input_ = format_for_conv(input_, device=self.device)
        input_ = torch.tensor(input_).unsqueeze(0).float().to(self.device)

        return input_, target

    def __len__(self):
        return self.len_dataset