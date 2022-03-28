import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from deep_morpho.morp_operations import ParallelMorpOperations

from general.utils import one_hot_array
from general.nn.dataloaders import dataloader_resolution


class AxspaROIDataset(Dataset):

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
        return dataloader_resolution(
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


class AxspaROISimpleDataset(Dataset):

    def __init__(self, data, morp_operations: ParallelMorpOperations = None, preprocessing=None, ):
        self.data = data
        self.preprocessing = preprocessing
        if morp_operations is None:
            morp_operations = self.get_default_morp_operation()
        self.morp_operations = morp_operations

    def __getitem__(self, idx):
        input_ = np.load(self.data['path_segm'].iloc[idx])
        # target = np.load(self.data['path_roi'].iloc[idx])

        # input_ = np.stack([input_, target], axis=-1)
        input_ = one_hot_array(input_, nb_chans=2)
        target = self.morp_operations(input_).float()
        input_ = torch.tensor(input_).float()

        input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)
        target = target.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        # target = target != self.data['value_bg'].iloc[idx]

        if self.preprocessing is not None:
            input_ = self.preprocessing(input_)
            target = self.preprocessing(target)

        return input_.float(), target.float()


    def __len__(self):
        return len(self.data)


    @staticmethod
    def get_loader(data, batch_size, morp_operations=None, preprocessing=None, **kwargs):
        return dataloader_resolution(
            df=data,
            dataset=AxspaROISimpleDataset,
            dataset_args={"morp_operations": morp_operations, "preprocessing": preprocessing},
            batch_size=batch_size,
            **kwargs
        )

    @staticmethod
    def get_train_val_test_loader(data_train, data_val, data_test, *args, **kwargs):
        trainloader = AxspaROISimpleDataset.get_loader(data_train, *args, **kwargs)
        valloader = AxspaROISimpleDataset.get_loader(data_val, *args, **kwargs)
        testloader = AxspaROISimpleDataset.get_loader(data_test, *args, **kwargs)
        return trainloader, valloader, testloader

    def get_default_morp_operation(self, **kwargs):
        return ParallelMorpOperations(
            name="roi_detection",
            operations=[
                [[('dilation', ('hstick', 41), False), ('dilation', ('hstick', 41), False), 'intersection']]
            ],
            **kwargs
        )
