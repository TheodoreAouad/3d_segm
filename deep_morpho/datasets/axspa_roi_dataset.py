import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms

from general.nn.dataloaders import dataloader_resolution


class AxspaROIDataset(Dataset):

    def __init__(self, data, preprocessing=transforms.ToTensor()):
        self.data = data
        self.preprocessing = preprocessing


    def __getitem__(self, idx):
        input_ = np.load(self.data['path_segm'].iloc[idx])
        target = np.load(self.data['path_roi'].iloc[idx])

        input_ = input_ != 0
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
