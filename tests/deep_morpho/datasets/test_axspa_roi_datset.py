from os.path import join

import pandas as pd

from deep_morpho.datasets.axspa_roi_dataset import AxspaROIDataset, AxspaROISimpleDataset
from deep_morpho.morp_operations import ParallelMorpOperations


class TestAxspaROIDataset:

    @staticmethod
    def test_init_dataset():
        data = pd.read_csv(join('data', 'deep_morpho', 'axspa_roi', 'axspa_roi.csv'))
        dataset = AxspaROIDataset(data)
        dataset[0]

    @staticmethod
    def test_dataloader():
        data = pd.read_csv(join('data', 'deep_morpho', 'axspa_roi', 'axspa_roi.csv'))
        dataloader = AxspaROIDataset.get_loader(data, batch_size=20)
        for img, target in dataloader:
            pass


class TestAxspaROISimpleDataset:

    @staticmethod
    def test_init_dataset():
        data = pd.read_csv(join('data', 'deep_morpho', 'axspa_roi', 'axspa_roi.csv'))
        morp_operations = ParallelMorpOperations.erosion(('disk', 3))
        dataset = AxspaROISimpleDataset(data, morp_operations=morp_operations)
        dataset[0]

    @staticmethod
    def test_dataloader():
        data = pd.read_csv(join('data', 'deep_morpho', 'axspa_roi', 'axspa_roi.csv'))
        morp_operations = ParallelMorpOperations.erosion(('disk', 3))
        dataloader = AxspaROISimpleDataset.get_loader(data, morp_operations=morp_operations, batch_size=20)
        for img, target in dataloader:
            pass
