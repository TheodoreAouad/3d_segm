from typing import Tuple
import os
from os.path import join
import re
import pathlib

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from deep_morpho.dataloaders.custom_loaders import DataLoaderEpochTracker
from deep_morpho.morp_operations import ParallelMorpOperations
from general.utils import load_json, log_console
from .datamodule_base import DataModule
from .generate_forms3 import get_random_diskorect_channels

from .generator_dataset import GeneratorDataset

# def get_loader(batch_size, n_inputs, random_gen_fn, random_gen_args, morp_operation, device='cpu', **kwargs):
#     return DataLoader(
#         MultiRectDatasetGenerator(random_gen_fn, random_gen_args, morp_operation=morp_operation, device=device, n_inputs=n_inputs, ),
#         batch_size=batch_size,  **kwargs
#     )

# DEBUG
# N_DISKO = 0


class DiskorectDataset(GeneratorDataset):
    """ Random Generation behavior: generate a new input at each call of __getitem__. If num_workers=0, each epoch is independent
    and outputs a new input. If num_workers>0, each epoch is the same and outputs the same inputs.
    """
    def __init__(
        self,
        morp_operation: ParallelMorpOperations,
        random_gen_fn=get_random_diskorect_channels,
        random_gen_args={
            'size': (50, 50),
            'n_shapes': 20,
            'max_shape': (20, 20),
            'p_invert': .5,
            'n_holes': 10,
            'max_shape_holes': (10, 10),
            'noise_proba': 0.02,
            "border": (0, 0),
        },
        device: str = "cpu",
        do_symetric_output: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.random_gen_fn = random_gen_fn
        self.random_gen_args = random_gen_args
        self.device = device
        self.morp_fn = morp_operation
        self.do_symetric_output = do_symetric_output
        # self.rng = np.random.default_rng(seed)
        # print(seed)

        # DEBUG
        # self.nb_inp = 0
        # global N_DISKO
        # self.n_disko = N_DISKO
        # N_DISKO += 1
        # print(f"Reset. N_DISKO = {N_DISKO}, self.n_disko = {self.n_disko}, self.nb_inp = {self.nb_inp}")


    def generate_input_target(self):
        # # DEBUG
        # with open("todelete/worker_id.txt", "a") as f:
        #     info = torch.utils.data.get_worker_info()
        #     f.write(f"{info.id}\n")

        self.rng = self.get_rng()
        input_ = self.random_gen_fn(rng_float=self.rng.random, rng_int=self.rng.integers, **self.random_gen_args,)
        target = self.morp_fn(input_)

        target = torch.tensor(target).float()
        input_ = torch.tensor(input_).float()

        if input_.ndim == 2:
            input_ = input_.unsqueeze(-1)  # Must have at least one channel

        input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)
        target = target.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        # DEBUG
        # pathlib.Path(f"todelete/{self.n_disko}/input_{self.nb_inp}.png").parent.mkdir(parents=True, exist_ok=True)
        # plt.imsave(f"todelete/{self.n_disko}/input_{self.nb_inp}.png", input_.squeeze().cpu().numpy())
        # self.nb_inp += 1
        # print(f"todelete/{self.n_disko}/input_{self.nb_inp}.png")

        if self.do_symetric_output:
            return 2 * input_ - 1, 2 * target - 1
        return input_, target



class MultiRectDataset(Dataset):
    def __init__(
            self,
            inputs_path: str,
            targets_path: str,
            do_load_in_ram: bool = False,
            verbose: bool = True,
            n_inputs: int = None,
            logger=None,
    ):
        self.inputs_path = inputs_path
        self.targets_path = targets_path
        self.do_load_in_ram = do_load_in_ram
        self.verbose = verbose
        self.logger = logger
        self.n_inputs = n_inputs

        self.all_inputs_name = sorted(os.listdir(inputs_path), key=lambda x: int(re.findall(r'\d+', x)[0]))
        if self.n_inputs is not None:
            self.all_inputs_name = self.all_inputs_name[:self.n_inputs]

        if self.do_load_in_ram:
            self.all_inputs = []
            self.all_targets = []

            if verbose:
                log_console('Loading data in RAM...', logger=self.logger)
            for inpt in self.get_verbose_iterator(self.all_inputs_name):
                self.all_inputs.append(np.load(join(inputs_path, inpt)))
                self.all_targets.append(np.load(join(targets_path, inpt)))


    def get_verbose_iterator(self, iterator):
        if self.verbose:
            return tqdm(iterator)
        return iterator

    def __getitem__(self, idx):
        if self.do_load_in_ram:
            input_, target = self.all_inputs[idx], self.all_targets[idx]
        else:
            img_name = self.all_inputs_name[idx]
            input_, target = np.load(join(self.inputs_path, img_name)), np.load(join(self.targets_path, img_name))
        target = torch.tensor(target).float()
        input_ = torch.tensor(input_).unsqueeze(0).float()

        return input_, target

    def __len__(self):
        return len(self.all_inputs_name)

    @staticmethod
    def get_loader(batch_size, dataset_path, do_load_in_ram, morp_operation, logger=None, n_inputs=None, **kwargs):
        inputs_path = join(dataset_path, 'images')
        metadata = load_json(join(dataset_path, 'metadata.json'))
        targets_path = metadata["seqs"][morp_operation.get_saved_key()]['path_target']
        return DataLoader(
            MultiRectDataset(inputs_path, targets_path, do_load_in_ram=do_load_in_ram, verbose=True, logger=logger, n_inputs=n_inputs),
            batch_size=batch_size, **kwargs
        )
