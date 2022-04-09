import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


from deep_morpho.utils import set_seed


class DummyDataset(Dataset):
    def __init__(self, length=30):
        super().__init__()
        self.length = length

    def __getitem__(self, idx):
        return torch.rand(20, 20) + idx

    def __len__(self):
        return self.length


class TestSetSeed:

    @staticmethod
    def test_random_rand():
        seed = set_seed()
        a = random.randrange(0, 1)
        set_seed(seed)
        assert random.randrange(0, 1) == a

    @staticmethod
    def test_numpy_rand():
        seed = set_seed()
        a = np.random.rand()
        set_seed(seed)
        assert np.random.rand() == a

    @staticmethod
    def test_torch_rand():
        seed = set_seed()
        a = torch.rand(1)
        set_seed(seed)
        assert torch.rand(1) == a

    @staticmethod
    def test_init_conv():
        seed = set_seed()
        a = nn.Conv2d(3, 3, 3)
        set_seed(seed)
        assert (a.weight == nn.Conv2d(3, 3, 3).weight).all()

    @staticmethod
    def test_dataloader():
        seed = set_seed()
        loader = DataLoader(DummyDataset(), batch_size=3, shuffle=True, num_workers=3)
        tensors1 = []
        for tensor in loader:
            tensors1.append(tensor)

        set_seed(seed)
        loader2 = DataLoader(DummyDataset(), batch_size=3, shuffle=True, num_workers=3)
        tensors2 = []
        for tensor in loader2:
            tensors2.append(tensor)


        for tensor1, tensor2 in zip(tensors1, tensors2):
            assert (tensor1 == tensor2).all()
