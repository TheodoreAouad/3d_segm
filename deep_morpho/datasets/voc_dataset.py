from typing import Union, Optional, List, Callable, Tuple


import numpy as np
from torchvision.datasets import VOCSegmentation
from .select_indexes_dataset import SelectIndexesDataset


with open('deep_morpho/datasets/root_vocsegmentation_dir.txt', 'r') as f:
    ROOT_VOCSEGMENTATION_DIR = f.read()


class VOCSegmentationClassical(SelectIndexesDataset, VOCSegmentation):
    def __init__(
        self,
        n_inputs: Union[int, str] = "all",
        first_idx: int = 0,
        indexes: Optional[List[int]] = None,
        preprocessing: Callable = None,
        *args, **kwargs
    ):
        VOCSegmentation.__init__(self, root=ROOT_VOCSEGMENTATION_DIR, download=False, *args, **kwargs)
        self.targets = np.array(self.targets)
        self.images = np.array(self.images)
        self.data = np.array(self.images)
        self.preprocessing = preprocessing


    @property
    def targets(self):
        return np.array(super().targets)