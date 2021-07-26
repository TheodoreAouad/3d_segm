from typing import Tuple, Dict

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from skimage.morphology import disk
from general.array_morphology import array_dilation, array_erosion


def get_loader(batch_size, n_inputs, random_gen_fn, random_gen_args, device='cpu', **kwargs):
    return DataLoader(
        MultiRectDataset(random_gen_fn, random_gen_args, device=device, len_dataset=n_inputs, **kwargs),
        batch_size=batch_size,
    )


class MultiRectDataset(Dataset):

    def __init__(
            self,
            random_gen_fn,
            random_gen_args,
            selem: "np.ndarray" = disk(3),
            morp_operation: str = "dilation",
            device: str = "cpu",
            len_dataset: int = 1000,
    ):
        self.random_gen_fn = random_gen_fn
        self.random_gen_args = random_gen_args
        self.selem = selem
        self.device = device
        self.len_dataset = len_dataset
        self.morp_operation = morp_operation.lower()

        if self.morp_operation == 'dilation':
            self.morp_fn = array_dilation
        elif self.morp_operation == 'erosion':
            self.morp_fn = array_erosion
        elif self.morp_operation == 'opening':
            self.morp_fn = lambda x, *args, **kwargs: array_dilation(
                array_erosion(x, *args, **kwargs), *args, **kwargs
            )

    def __getitem__(self, idx):
        input_ = self.random_gen_fn(**self.random_gen_args)
        target = self.morp_fn(input_, self.selem, device=self.device, return_numpy_array=False).float()
        # input_ = format_for_conv(input_, device=self.device)
        input_ = torch.tensor(input_).unsqueeze(0).float().to(self.device)

        return input_, target

    def __len__(self):
        return self.len_dataset
