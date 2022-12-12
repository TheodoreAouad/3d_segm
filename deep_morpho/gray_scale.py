from typing import Tuple, Union

import torch
import numpy as np

from deep_morpho.tensor_with_attributes import TensorGray


def undersample(min_value: float, max_value: float, n_value: int):
    return np.round(np.linspace(min_value, max_value, n_value)).astype(int)


def level_sets_from_gray(ar: Union[np.ndarray, torch.Tensor], values: Union[np.ndarray, torch.Tensor] = None, n_values: int = "all") -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
    """ Given an array with any number of values, outputs an array with one more dimension
    with level sets as binary arrays.
    """
    if isinstance(ar, np.ndarray):
        values = np.unique(ar) if values is None else values
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        constructor_fn = np.zeros

    elif isinstance(ar, torch.Tensor):
        values = torch.unique(ar) if values is None else values
        if not isinstance(values, torch.Tensor):
            values = TensorGray(values, device=ar.device)
        # constructor_fn = partial(torch.zeros, device=ar.device)

        def constructor_fn(x):
            res = TensorGray(size=x)
            res.to(ar.device)
            return res

    else:
        raise ValueError("ar type must be numpy.ndarray or torch.Tensor")

    if isinstance(n_values, int) and n_values > 0:
        values = values[undersample(0, len(values)-1, n_values)]

    res = constructor_fn((len(values),) + ar.shape)
    for idx, v in enumerate(values):
        res[idx] = ar >= v

    return res, values


def gray_from_level_sets(ar: Union[np.ndarray, torch.Tensor], values: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """ Given level sets (in {0, 1}) and the corresponding values, outputs the corresponding gray scale array
    """
    v2 = values[1:] - values[:-1]

    if isinstance(v2, np.ndarray):
        v2 = np.concatenate([[values[0]], v2])
    elif isinstance(v2, torch.Tensor):
        v2 = torch.cat([TensorGray([values[0]]).to(v2.device), v2])
        # v2 = v2.to(ar.device)
    else:
        raise ValueError("value type must be numpy.ndarray or torch.Tensor")

    return (ar * v2[:, None, None]).sum(0)
