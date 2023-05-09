from typing import Union, Optional, List, Callable, Tuple
from random import shuffle

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader

from .select_indexes_dataset import SelectIndexesDataset
from general.nn.dataloaders import ComposeDataloaders


with open('deep_morpho/datasets/root_vocsegmentation_dir.txt', 'r') as f:
    ROOT_VOCSEGMENTATION_DIR = f.read()

CLASSES = {
    0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat",
    9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "potted plant",
    17: "sheep", 18: "sofa", 19: "train", 20: "tv monitor", 255: "void"
}

CLASSES_INDEXES = {v: k for k, v in CLASSES.items()}
DEFAULT_PREPROCESSING_INPUT = transforms.ToTensor()
DEFAULT_PREPROCESSING_TARGET = transforms.PILToTensor()


class VOCSegmentationClassical(SelectIndexesDataset, VOCSegmentation):
    CLASSES = CLASSES
    CLASSES_INDEXES = CLASSES_INDEXES

    LEN_TRAIN = 1464
    LEN_TRAINVAL = 2913
    LEN_VAL = 1449

    def __init__(
        self,
        n_inputs: Union[int, str] = "all",
        first_idx: int = 0,
        indexes: Optional[List[int]] = None,
        preprocessing_input: Callable = DEFAULT_PREPROCESSING_INPUT,
        preprocessing_target: Callable = DEFAULT_PREPROCESSING_TARGET,
        do_replace_voids: bool = True,
        *args, **kwargs
    ):
        VOCSegmentation.__init__(
            self,
            root=ROOT_VOCSEGMENTATION_DIR,
            download=False,
            transform=preprocessing_input,
            target_transform=preprocessing_target,
            *args, **kwargs
        )

        self.targets = np.array(self.targets)
        self.images = np.array(self.images)
        self.data = np.array(self.images)  # Set up data for SelectIndexesDataset init
        SelectIndexesDataset.__init__(self, n_inputs=n_inputs, first_idx=first_idx, indexes=indexes, *args, **kwargs)

        self.images = self.data  # Update images after SelectIndexesDataset init
        self.n_classes = len(self.CLASSES) - 1 if do_replace_voids else len(self.CLASSES)  # -1 for void class
        self.do_replace_voids = do_replace_voids

    def __getitem__(self, index):
        img, tar = super().__getitem__(index)

        if self.do_replace_voids:
            tar[tar == self.CLASSES_INDEXES["void"]] = self.CLASSES_INDEXES["background"]

        return img, tar

    @staticmethod
    def load_resolutions(image_paths: List[str]):
        resolutions = np.empty((len(image_paths), 2), dtype=np.int)
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path)
            resolutions[i] = img.size[::-1]
        return resolutions

    @property
    def preprocessing_input(self):
        return self.transform

    @property
    def preprocessing_target(self):
        return self.target_transform

    @property
    def preprocessing(self):
        return self.transforms

    @classmethod
    def get_loader(
        cls,
        batch_size,
        n_inputs: Union[int, str] = "all",
        num_workers: int = 0,
        shuffle: bool = False,
        **kwargs
    ):
        if n_inputs == 0:
            return DataLoader([])

        all_dataset = cls(n_inputs=n_inputs, **kwargs)
        resolutions = cls.load_resolutions(all_dataset.images)

        indexes = all_dataset.indexes

        if "indexes" in kwargs:
            kwargs.pop("indexes")

        dataloaders = []
        len_tot = 0  # DEBUG
        for resolution in np.unique(resolutions, axis=0):
            new_indexes = indexes[(resolutions == resolution).prod(1).astype(bool)]
            dataloaders.append(DataLoader(
                cls(indexes=new_indexes, **kwargs),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
            ))
            len_tot += len(new_indexes)  # DEBUG


        return ComposeDataloaders(dataloaders, shuffle=shuffle)


    @classmethod
    def get_train_val_test_loader_from_experiment(cls, experiment: "ExperimentBase", ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        args: "Parser" = experiment.args

        n_inputs_train = min(args[f"n_inputs{args.trainset_args_suffix}"], cls.LEN_TRAIN)
        n_inputs_val = min(args[f"n_inputs{args.valset_args_suffix}"], cls.LEN_TRAINVAL)
        n_inputs_test = min(args[f"n_inputs{args.testset_args_suffix}"], cls.LEN_VAL)

        train_kwargs, val_kwargs, test_kwargs = cls.get_train_val_test_kwargs_pop_keys(
            experiment, keys=["n_inputs", "indexes", "shuffle", "first_idx", "image_set"]
        )

        all_train_idxs = list(range(n_inputs_train))
        shuffle(all_train_idxs)
        train_idxes = all_train_idxs[:n_inputs_train]


        trainloader = cls.get_loader(indexes=train_idxes, image_set="train", shuffle=True, batch_size=args["batch_size"], num_workers=args["num_workers"], **train_kwargs)
        valloader = cls.get_loader(first_idx=0, n_inputs=n_inputs_val, image_set="trainval", shuffle=False, batch_size=args["batch_size"], num_workers=args["num_workers"], **val_kwargs)
        testloader = cls.get_loader(first_idx=0, n_inputs=n_inputs_test, image_set="val", shuffle=False, batch_size=args["batch_size"], num_workers=args["num_workers"], **test_kwargs)

        return trainloader, valloader, testloader
