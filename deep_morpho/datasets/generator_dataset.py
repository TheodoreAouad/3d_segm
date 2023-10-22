from abc import ABC, abstractmethod
from typing import Tuple

import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from .datamodule_base import DataModule
from deep_morpho.dataloaders.custom_loaders import DataLoaderEpochTracker


class GeneratorDataset(DataModule, Dataset, ABC):
    """ Random Generation behavior: generate a new input at each call of __getitem__. If num_workers=0, each epoch is independent
    and outputs a new input. If num_workers>0, each epoch is the same and outputs the same inputs.
    """
    def __init__(
        self,
        n_inputs: int = 1000,
        seed: int = None,
        max_generation_nb: int = 0,
    ):
        self.n_inputs = n_inputs
        self.max_generation_nb = max_generation_nb
        self.data = {}
        self.seed = seed
        self.rng = None
        self.epoch = None


    def get_rng(self):
        if self.rng is None:
            seed = self.seed if self.seed is not None else np.random.randint(0, 2 ** 32 - 1)

            # Different RNG for each worker and epoch. new_seed = seed * 2 ** epoch * 3 ** worker_id, ensuring that
            # each couple (worker, epoch) has a different seed.
            if self.epoch is not None:
                seed *= 2 ** self.epoch

            info = torch.utils.data.get_worker_info()
            if info is not None:
                seed *= 3 ** info.id

            self.rng = np.random.default_rng(seed)

        return self.rng

    def __getitem__(self, idx):
        if self.max_generation_nb == 0:
            return self.generate_input_target()

        idx = idx % self.max_generation_nb

        if idx not in self.data.keys():
            self.data[idx] = self.generate_input_target()

        return self.data[idx]

    @abstractmethod
    def generate_input_target(self):
        pass

    def __len__(self):
        return self.n_inputs

    @classmethod
    def get_loader(
        cls,
        batch_size,
        n_inputs: int,
        num_workers: int = 0,
        shuffle: bool = False,
        track_epoch: bool = True,
        **kwargs
    ):
        if track_epoch:
            loader = DataLoaderEpochTracker
        else:
            loader = DataLoader

        if n_inputs == 0:
            return loader([])
        return loader(
            cls(n_inputs=n_inputs, **kwargs), batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, )

    @classmethod
    def get_train_val_test_loader_from_experiment(cls, experiment: "ExperimentBase") -> Tuple[DataLoader, DataLoader, DataLoader]:
        args = experiment.args

        n_inputs_train = args[f"n_inputs{args.trainset_args_suffix}"]
        n_inputs_val = args[f"n_inputs{args.valset_args_suffix}"]
        n_inputs_test = args[f"n_inputs{args.testset_args_suffix}"]

        train_kwargs, val_kwargs, test_kwargs = cls.get_train_val_test_kwargs_pop_keys(
            experiment, keys=["n_inputs"]
        )

        train_loader = cls.get_loader(n_inputs=n_inputs_train, shuffle=True, batch_size=args["batch_size"], num_workers=args["num_workers"], **train_kwargs)
        val_loader = cls.get_loader(n_inputs=n_inputs_val, shuffle=False, batch_size=args["batch_size"], num_workers=args["num_workers"], **val_kwargs)
        test_loader = cls.get_loader(n_inputs=n_inputs_test, shuffle=False, batch_size=args["batch_size"], num_workers=args["num_workers"], **test_kwargs)

        return train_loader, val_loader, test_loader
