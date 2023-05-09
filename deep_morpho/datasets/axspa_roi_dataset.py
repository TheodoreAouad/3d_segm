import torch
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from deep_morpho.morp_operations import ParallelMorpOperations

from general.utils import one_hot_array
from general.nn.utils import train_val_test_split
from general.nn.dataloaders import dataloader_resolution_from_df

from .datamodule_base import DataModule


class AxspaROIDataset(DataModule, Dataset):

    def __init__(self, data, preprocessing=transforms.ToTensor(), unique_resolution=True):
        self.data = data
        if unique_resolution:
            self.keep_only_one_res()
        self.preprocessing = preprocessing


    def keep_only_one_res(self):
        max_res = self.data['resolution'].value_counts(sort=True, ascending=False).index[0]
        self.data = self.data[self.data['resolution'] == max_res]


    def __getitem__(self, idx):
        input_ = np.load(self.data['path_segm'].iloc[idx])
        target = np.load(self.data['path_roi'].iloc[idx])

        # input_ = np.stack([input_, target], axis=-1)
        input_ = one_hot_array(input_, nb_chans=2)
        target = target != self.data['value_bg'].iloc[idx]

        if self.preprocessing is not None:
            input_ = self.preprocessing(input_)
            target = self.preprocessing(target)

        return input_.float(), target.float()


    def __len__(self):
        return len(self.data)


    @staticmethod
    def get_loader(data, batch_size, preprocessing=transforms.ToTensor(), **kwargs):
        return dataloader_resolution_from_df(
            df=data,
            dataset=AxspaROIDataset,
            dataset_args={"preprocessing": preprocessing},
            batch_size=batch_size,
            **kwargs
        )

    @staticmethod
    def get_train_val_test_loader(data_train, data_val, data_test, *args, **kwargs):
        trainloader = AxspaROIDataset.get_loader(data_train, *args, **kwargs)
        valloader = AxspaROIDataset.get_loader(data_val, *args, **kwargs)
        testloader = AxspaROIDataset.get_loader(data_test, *args, **kwargs)
        return trainloader, valloader, testloader


class AxspaROISimpleDataset(DataModule, Dataset):

    def __init__(self, data, morp_operations: ParallelMorpOperations = None, preprocessing=None, do_symetric_output: bool = False):
        self.data = data
        self.preprocessing = preprocessing
        if morp_operations is None:
            morp_operations = self.get_default_morp_operation()
        self.morp_operations = morp_operations
        self.do_symetric_output = do_symetric_output

    def __getitem__(self, idx):
        input_ = np.load(self.data['path_segm'].iloc[idx])
        # target = np.load(self.data['path_roi'].iloc[idx])

        # input_ = np.stack([input_, target], axis=-1)
        input_ = one_hot_array(input_, nb_chans=2)
        target = torch.tensor(self.morp_operations(input_)).float()
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


    def __len__(self):
        return len(self.data)


    @classmethod
    def get_loader(cls, data, batch_size, morp_operations=None, preprocessing=None, do_symetric_output=False, num_workers=0,
            shuffle=False, **kwargs):
        return dataloader_resolution_from_df(
            df=data,
            dataset=cls,
            dataset_args={"morp_operations": morp_operations, "preprocessing": preprocessing, "do_symetric_output": do_symetric_output},
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            # **kwargs
        )

    # @classmethod
    # def get_train_val_test_loader(cls, n_inputs_train, n_inputs_val, n_inputs_test, *args, **kwargs):
    #     data = pd.read_csv(kwargs['dataset_path'])
    #     max_res = data['resolution'].value_counts(sort=True, ascending=False).index[0]
    #     data = data[data['resolution'] == max_res]

    #     data_train, data_val, data_test = train_val_test_split(
    #         data,
    #         train_size=n_inputs_train,
    #         val_size=n_inputs_val,
    #         test_size=n_inputs_test,
    #     )

    #     if "data" in kwargs:
    #         kwargs.pop("data")

    #     trainloader = cls.get_loader(data=data_train, shuffle=True, *args, **kwargs)
    #     valloader = cls.get_loader(data=data_val, shuffle=False, *args, **kwargs)
    #     testloader = cls.get_loader(data=data_test, shuffle=False, *args, **kwargs)
    #     return trainloader, valloader, testloader

    @classmethod
    def get_train_val_test_loader_from_experiment(cls, experiment: "ExperimentBase"):
        args = experiment.args

        n_inputs_train = args[f"n_inputs{args.trainset_args_suffix}"]
        n_inputs_val = args[f"n_inputs{args.valset_args_suffix}"]
        n_inputs_test = args[f"n_inputs{args.testset_args_suffix}"]

        train_kwargs, val_kwargs, test_kwargs = cls.get_train_val_test_kwargs_pop_keys(experiment, keys=["data"])

        data = pd.read_csv(args['dataset_path'])
        max_res = data['resolution'].value_counts(sort=True, ascending=False).index[0]
        data = data[data['resolution'] == max_res]

        data_train, data_val, data_test = train_val_test_split(
            data,
            train_size=n_inputs_train,
            val_size=n_inputs_val,
            test_size=n_inputs_test,
        )

        trainloader = cls.get_loader(data=data_train, shuffle=True, **train_kwargs)
        valloader = cls.get_loader(data=data_val, shuffle=False, **val_kwargs)
        testloader = cls.get_loader(data=data_test, shuffle=False, **test_kwargs)

        return trainloader, valloader, testloader

    def get_default_morp_operation(self, **kwargs):
        return ParallelMorpOperations(
            name="roi_detection",
            operations=[
                [[('dilation', ('hstick', 41), False), ('dilation', ('hstick', 41), False), 'intersection']]
            ],
            **kwargs
        )
