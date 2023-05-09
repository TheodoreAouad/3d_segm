from typing import Tuple

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from ..morp_operations import ParallelMorpOperations
from .generate_sticks_noised import get_sticks_noised_channels
from .utils import get_rect
from .datamodule_base import DataModule


class NoistiDataset(DataModule, Dataset):
    def __init__(
            self,
            size,
            angles: np.ndarray = np.linspace(0, 180, 10),
            n_shapes: int = 15,
            lengths_lim: Tuple = (12, 15),
            widths_lim: Tuple = (3, 3),
            p_invert: float = 0,
            border: Tuple = (0, 0),
            noise_proba: float = 0.1,
            n_inputs: int = 1000,
            seed: int = None,
            max_generation_nb: int = 0,
            do_symetric_output: bool = False,
    ):
        self.size = size
        self.angles = angles
        self.n_shapes = n_shapes
        self.lengths_lim = lengths_lim
        self.widths_lim = widths_lim
        self.p_invert = p_invert
        self.border = border
        self.noise_proba = noise_proba

        self.n_inputs = n_inputs
        self.max_generation_nb = max_generation_nb
        self.do_symetric_output = do_symetric_output
        self.data = {}
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, idx):
        if self.max_generation_nb == 0:
            return self.generate_input_target()

        idx = idx % self.max_generation_nb

        if idx not in self.data.keys():
            self.data[idx] = self.generate_input_target()

        return self.data[idx]

    @property
    def channels(self):
        return self.size[2]

    @staticmethod
    def get_default_morp_operation(lengths_lim, angles, **kwargs):
        L = lengths_lim[0] - 2
        selems = [get_rect(width=0, height=L, angle=theta) for theta in angles]
        kwargs["name"] = kwargs.get("name", 'sticks_noised')
        return ParallelMorpOperations(
            operations=[
                [
                    [("erosion", selem, False), 'union'] for selem in selems
                ],
                [
                    [("dilation", selem[::-1, ::-1] + 0, False) for selem in selems] + ['union']
                ]
            ],
            **kwargs
        )

    @property
    def default_morp_operation(self):
        return self.get_default_morp_operation(lengths_lim=self.lengths_lim, angles=self.angles)

    @property
    def generation_kwargs(self):
        return {
            key: getattr(self, key) for key in [
                "size", "angles", "n_shapes", "lengths_lim", "widths_lim", "p_invert", "border", "noise_proba",
            ]
        }

    def generate_input_target(self):
        target, input_ = get_sticks_noised_channels(
            rng_float=self.rng.random, rng_int=self.rng.integers,
            **self.generation_kwargs
        )

        target = torch.tensor(target).float()
        input_ = torch.tensor(input_).float()

        if input_.ndim == 2:
            input_ = input_.unsqueeze(-1)  # Must have at least one channel

        input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)
        target = target.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        if self.do_symetric_output:
            return 2 * input_ - 1, 2 * target - 1
        return input_, target

    def __len__(self):
        return self.n_inputs

    # TODO: factorize with DiskorectDataset
    @classmethod
    def get_loader(
        cls,
        batch_size,
        n_inputs: int = "all",
        num_workers: int = 0,
        shuffle: bool = False,
        **kwargs
    ):
        if n_inputs == 0:
            return DataLoader([])
        return DataLoader(
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
