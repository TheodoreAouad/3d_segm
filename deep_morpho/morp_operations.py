from typing import Union, Tuple, List, Callable, Any

import numpy as np

from general.structuring_elements import *
from general.array_morphology import array_erosion, array_dilation, union_chans, intersection_chans


class SequentialMorpOperations:
    str_to_selem_fn = {
        'disk': disk, 'vstick': vstick, 'hstick': hstick, 'square': square, 'dcross': dcross, 'scross': scross,
        'vertical_stick': vstick, 'horizontal_stick': hstick, 'diagonal_cross': dcross, 'straight_cross': scross,
    }
    str_to_fn = {'dilation': array_dilation, 'erosion': array_erosion}

    def __init__(
        self,
        operations: List['str'],
        selems: List[Union[np.ndarray, Tuple[Union[str, Callable], Any]]],
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
            if isinstance(selem, np.ndarray):
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


class ParallelMorpOperations:
    str_to_selem_fn = {
        'disk': disk, 'vstick': vstick, 'hstick': hstick, 'square': square, 'dcross': dcross, 'scross': scross,
        'vertical_stick': vstick, 'horizontal_stick': hstick, 'diagonal_cross': dcross, 'straight_cross': scross,
    }
    str_to_fn = {'dilation': array_dilation, 'erosion': array_erosion}
    str_to_ui_fn = {'union': union_chans, 'intersection': intersection_chans}

    def __init__(
        self,
        operations: List[List[List[Union[Tuple[Union[Callable, str], Union[Callable, Tuple[str, int]]], 'str', Callable]]]],
        device="cpu",
        return_numpy_array: bool = False,
        name: str = None,
    ):
        self.operations_original = operations
        self.device = device
        self.return_numpy_array = return_numpy_array
        self.name = name

        self.operations = None
        self.operation_names = None
        self.selem_names = None
        self.selem_args = None
        self.layers_in_channels = None
        self.layers_out_channels = None
        self.selems = None

        self.convert_ops(operations)


    def _erodila_selem_converter(self, args):
        selem_name = None
        selem_args = None
        if not isinstance(args, tuple):
            return args, selem_name, selem_args
        if isinstance(args[0], str):
            selem_name = args[0]
            selem_op = self.str_to_selem_fn[selem_name]
        else:
            selem_op = args[0]

        selem_args = args[1]
        return selem_op(selem_args), selem_name, selem_args


    def _erodila_op_converter(self, args):
        op_name = None
        selem_name = None
        selem_args = None
        selem = None

        if not isinstance(args, tuple):
            return args, op_name, selem_name, selem_args, selem

        if isinstance(args[0], str):
            op_name = args[0]
            op_fn = self.str_to_fn[args[0]]
        else:
            op_fn = args[0]

        selem, selem_name, selem_args = self._erodila_selem_converter(args[1])
        return lambda x: op_fn(x, selem=selem), op_name, selem_name, selem_args, selem


    def _ui_converter(self, args):

        ui_name = None
        ui_args = "all"

        if isinstance(args, tuple):
            ui_fn = args[0]
            ui_args = args[1]
        else:
            ui_fn = args

        if isinstance(ui_fn, str):
            ui_name = ui_fn
            ui_fn = self.str_to_ui_fn[ui_name]

        return lambda x: ui_fn(x, ui_args), ui_name, ui_args


    def convert_ops(self, all_op_str):
        alls = {key: [] for key in ["op_fn", "op_names", "selem_names", "selem_args", "selems"]}
        layers = {key: [] for key in ["op_fn", "op_names", "selem_names", "selem_args", "selems"]}
        chans = {key: [] for key in ["op_fn", "op_names", "selem_names", "selem_args", "selems"]}

        in_channels = []
        out_channels = []

        for layer_str in all_op_str:
            for key in layers.keys():
                layers[key] = []
            out_channels.append(len(layer_str))
            in_channels.append(len(layer_str[0]) - 1)

            for chan_str in layer_str:

                for key in chans.keys():
                    chans[key] = []

                for cur_op_str in chan_str[:-1]:
                    op_fn, op_name, selem_name, selem_args, selem = self._erodila_op_converter(cur_op_str)

                    chans["op_fn"].append(op_fn)
                    chans["op_names"].append(op_name)
                    chans['selem_names'].append(selem_name)
                    chans['selem_args'].append(selem_args)
                    chans['selems'].append(selem)

                ui_fn, ui_name, ui_args = self._ui_converter(chan_str[-1])
                chans["op_fn"].append(ui_fn)
                chans["op_names"].append(ui_name)
                chans['selem_names'].append("channels")
                chans["selem_args"].append(ui_args)

                for key in layers.keys():
                    layers[key].append(chans[key])

            for key in alls.keys():
                alls[key].append(layers[key])

        self.operations = alls['op_fn']
        self.operation_names = alls['op_names']
        self.selem_names = alls['selem_names']
        self.selem_args = alls['selem_args']
        self.selems = alls['selems']

        assert in_channels[1:] == out_channels[:-1]

        self.in_channels = in_channels
        self.out_channels = out_channels

        return alls


    def apply_ops(self, ar):
        x = ar + 0
        for layer in self.operations:
            next_x = np.zeros(x.shape[:-1] + (len(layer),))
            for chan_idx, chan in enumerate(layer):
                morps, ui = chan[:-1], chan[-1]
                next_x[..., chan_idx] = ui(
                    np.stack([morps[idx](x[..., idx]) for idx in range(len(morps))], axis=-1)
                )
            x = next_x
        return x


    def __call__(self, ar):
        return self.apply_ops(ar)


    def __len__(self):
        return len(self.selems)

    def __repr__(self):
        repr_ = f'{self.__class__.__name__}(in_channels={self.in_channels[0]}, out_channels={self.out_channels[-1]})'
        for layer_idx in range(len(self.operation_names)):
            layer = self.operation_names[layer_idx]
            repr_ += f"\n{' '*4}Layer{layer_idx}(in_channels={self.in_channels[layer_idx]}, out_channels={self.out_channels[layer_idx]})"
            for chan_idx in range(len(layer)):
                chan = layer[chan_idx]

                if chan[-1] == 'intersection':
                    ui_name = 'inter'
                else:
                    ui_name = chan[-1]

                repr_ += f"\n{' '*8}Out{chan_idx}: {ui_name}(chans={self.selem_args[layer_idx][chan_idx][-1]}) |"
                for op_idx in range(len(self.operation_names[layer_idx][chan_idx]) - 1):
                    repr_ += f" {chan[op_idx]}({self.selem_names[layer_idx][chan_idx][op_idx]}({self.selem_args[layer_idx][chan_idx][op_idx]}))"

        return repr_


    def get_saved_key(self):
        return (
            '=>'.join(self.operations) +
            ' -- ' +
            "=>".join([f'{fn.__name__}({arg})' for fn, arg in zip(self._selem_fn, self._selem_arg)])
        )
