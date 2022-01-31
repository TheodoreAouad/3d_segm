from typing import Union, Callable, Union, List

import numpy as np
import torch
import torch.nn.functional as F


def format_for_conv(ar: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Formats the input array as a tensor in the format expected by the network and returns it.

    Args:
        ar: write your description
        np: write your description
        ndarray: write your description
        device: write your description
        torch: write your description
        device: write your description
    """
    return torch.tensor(ar).unsqueeze(0).unsqueeze(0).float().to(device)


def array_erosion(ar: np.ndarray, selem: np.ndarray, device: torch.device = "cpu", return_numpy_array: bool = True) -> Union[np.ndarray, torch.Tensor]:
    """
    Oscillator for the convolutional operator.

    Args:
        ar: write your description
        np: write your description
        ndarray: write your description
        selem: write your description
        np: write your description
        ndarray: write your description
        device: write your description
        torch: write your description
        device: write your description
        return_numpy_array: write your description
    """
    conv_fn = {2: F.conv2d, 3: F.conv3d}[ar.ndim]

    torch_array = (conv_fn(
        format_for_conv(ar, device=device), format_for_conv(selem, device=device),
        padding=selem.shape[0] // 2,
    ) == selem.sum()).squeeze()

    if return_numpy_array:
        return torch_array.to("cpu").int().numpy()
    return torch_array


def array_dilation(ar: np.ndarray, selem: np.ndarray, device: torch.device = "cpu", return_numpy_array: bool = True) -> Union[np.ndarray, torch.Tensor]:
    """
    Dilation of an array with a scalar element.

    Args:
        ar: write your description
        np: write your description
        ndarray: write your description
        selem: write your description
        np: write your description
        ndarray: write your description
        device: write your description
        torch: write your description
        device: write your description
        return_numpy_array: write your description
    """
    conv_fn = {2: F.conv2d, 3: F.conv3d}[ar.ndim]

    torch_array = (conv_fn(
        format_for_conv(ar, device=device), format_for_conv(selem, device=device),
        padding=selem.shape[0] // 2,
    ) > 0).squeeze()

    if return_numpy_array:
        return torch_array.to("cpu").int().numpy()
    return torch_array


def intersection(ars: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Intersection of arrays.

    Args:
        ars: write your description
        np: write your description
        ndarray: write your description
        axis: write your description
    """
    return ars.sum(axis) == ars.shape[axis]


def union(ars: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Returns the union of the ars along the given axis.

    Args:
        ars: write your description
        np: write your description
        ndarray: write your description
        axis: write your description
    """
    return ars.sum(axis) > 0


def fn_chans(ar: np.ndarray, fn: Callable, chans: Union[str, List[int]] = 'all') -> np.ndarray:
    """
    Apply function to each channel of ar

    Args:
        ar: write your description
        np: write your description
        ndarray: write your description
        fn: write your description
        chans: write your description
    """
    if chans == 'all':
        chans = range(ar.shape[-1])
    return torch.tensor(fn(np.stack([ar[..., chan] for chan in chans], axis=-1)))


def intersection_chans(ar, chans: Union[str, List[int]] = 'all') -> np.ndarray:
    """
    Return the channels that are in the intersection with the input array.

    Args:
        ar: write your description
        chans: write your description
    """
    return fn_chans(ar, intersection, chans)


def union_chans(ar, chans: Union[str, List[int]] = 'all') -> np.ndarray:
    """
    Return the union of chans in the array ar.

    Args:
        ar: write your description
        chans: write your description
    """
    return fn_chans(ar, union, chans)
