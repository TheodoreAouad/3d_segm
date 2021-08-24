from typing import List

import torch
import torch.nn.functional as F


def format_for_conv(ar, device):
    return torch.tensor(ar).unsqueeze(0).unsqueeze(0).float().to(device)


def array_erosion(ar, selem, device="cpu", return_numpy_array: bool = True):
    conv_fn = {2: F.conv2d, 3: F.conv3d}[ar.ndim]

    torch_array = (conv_fn(
        format_for_conv(ar, device=device), format_for_conv(selem, device=device),
        padding=selem.shape[0] // 2,
    ) == selem.sum()).squeeze()

    if return_numpy_array:
        return torch_array.to("cpu").int().numpy()
    return torch_array


def array_dilation(ar, selem, device="cpu", return_numpy_array: bool = True):
    conv_fn = {2: F.conv2d, 3: F.conv3d}[ar.ndim]

    torch_array = (conv_fn(
        format_for_conv(ar, device=device), format_for_conv(selem, device=device),
        padding=selem.shape[0] // 2,
    ) > 0).squeeze()

    if return_numpy_array:
        return torch_array.to("cpu").int().numpy()
    return torch_array


def sequentiel_morp_operations(
    operations: List['str'], selems: List["np.ndarray"], device="cpu", return_numpy_array: bool = True
):
    str_to_fn = {'dilation': array_dilation, 'erosion': array_erosion}

    def morp_fn(ar):
        res = ar + 0
        for op, selem in zip(operations, selems):
            res = str_to_fn[op](ar=res, selem=selem, device=device, return_numpy_array=return_numpy_array)

        return res
    return morp_fn
