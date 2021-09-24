from typing import Union, Tuple, List, Callable, Any

import torch
import torch.nn.functional as F
from numpy import ndarray

from general.structuring_elements import *


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



class SequentialMorpOperations:
    str_to_selem_fn = {
        'disk': disk, 'vstick': vstick, 'hstick': hstick, 'square': square, 'dcross': dcross, 'scross': scross,
        'vertical_stick': vstick, 'horizontal_stick': hstick, 'diagonal_cross': dcross, 'straight_cross': scross,
    }
    str_to_fn = {'dilation': array_dilation, 'erosion': array_erosion}

    def __init__(
        self,
        operations: List['str'],
        selems: List[Union[ndarray, Tuple[Union[str, Callable], Any]]],
        device="cpu",
        return_numpy_array: bool = False,
        name: str = None,
    ):
        self.operations = [op.lower() for op in operations]
        self._selems_original = selems
        self.selems = self._init_selems(selems)
        assert len(self.operations) == len(self.selems), "Must have same number of operations and selems"
        self.device = device
        self.return_numpy_array = return_numpy_array
        self.name = name



    def _init_selems(self, selems):
        res = []

        self._selem_fn = []
        self._selem_arg = []

        self._repr = "SequentialMorpOperations("
        for selem_idx, selem in enumerate(selems):
            if isinstance(selem, ndarray):
                res.append(selem)
                self._repr += f"{self.operations[selem_idx]}{selem.shape} => "
                self._selem_fn.append(None)
                self._selem_arg.append(None)
            elif isinstance(selem, tuple):
                selem_fn, selem_arg = selem
                if isinstance(selem[0], str):
                    selem_fn = self.str_to_selem_fn[selem_fn]
                res.append(selem_fn(selem_arg))

                self._repr += f"{self.operations[selem_idx]}({selem_fn.__name__}({selem_arg})) => "
                self._selem_fn.append(selem_fn)
                self._selem_arg.append(selem_arg)

        self._repr = self._repr[:-4] + ")"
        return res



    def morp_fn(self, ar):
        res = ar + 0
        for op, selem in zip(self.operations, self.selems):
            res = self.str_to_fn[op](ar=res, selem=selem, device=self.device, return_numpy_array=self.return_numpy_array)

        return res


    def __call__(self, ar):
        return self.morp_fn(ar)


    def __len__(self):
        return len(self.selems)

    def __repr__(self):
        # ops = ""
        # for op, selem in zip(self.operations, self.selems):
        #     ops += f"{op}{selem.shape}) "
        # ops = ops[:-1]
        # return f"SequentialMorpOperations({ops})"
        return self._repr


    def get_saved_key(self):
        return (
            '=>'.join(self.operations) +
            ' -- ' +
            "=>".join([f'{fn.__name__}({arg})' for fn, arg in zip(self._selem_fn, self._selem_arg)])
        )
