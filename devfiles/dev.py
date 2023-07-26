import torch
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __getitem__(self, idx):
        return torch.ones(1)

    def __len__(self):
        return 5


loader = DataLoader(MyDataset(), num_workers=2)
next(iter(loader))
