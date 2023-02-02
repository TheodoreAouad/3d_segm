from typing import Union, Optional, List

import numpy as np

from .datamodule_base import DataModule


class SelectIndexesDataset(DataModule):

    def __init__(
        self,
        n_inputs: Union[int, str] = "all",
        first_idx: int = 0,
        indexes: Optional[List[int]] = None,
        *args, **kwargs
    ) -> None:
        self.n_inputs = n_inputs
        self.first_idx = first_idx
        self.indexes = indexes

        if self.n_inputs != "all" or self.indexes is not None:
            if self.indexes is None:
                self.n_inputs = min(n_inputs, len(self.data))
                self.indexes = list(range(first_idx, min(first_idx + n_inputs, len(self))))

            self.data = self.data[self.indexes]
            self.targets = np.array(self.targets)[self.indexes]

        self.n_classes = len(np.unique(self.targets))
