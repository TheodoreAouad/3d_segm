import numpy as np
import torch
from general.structuring_elements import *

from deep_morpho.morp_operations import ParallelMorpOperations

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

morp_operations = []
selems = []



for ops in [
    ("dilation", "erosion", "dilation"),
]:
    for selems in [
        (('dcross', 7), ('hstick', 7), ("scross", 7)),
    ]:
        for aggreg in ["union", "intersection"]:

            morp_operations.append(ParallelMorpOperations(
                name=f'{aggreg}',
                operations=[
                    [
                        # [('dilation', ('hstick', 5)), ('dilation', ('hstick', 5)), 'intersection'],
                        [(op, selem) for op, selem in zip(ops, selems)] + [aggreg]
                    ]
                ]
            ))

for ops in [
    ("dilation", "erosion"),
]:
    for selems in [
        (('dcross', 7), ('hstick', 7)),
        (('disk', 3), ('vstick', 7)),
    ]:
        for aggreg in ["union", "intersection"]:

            morp_operations.append(ParallelMorpOperations(
                name=f'{aggreg}',
                operations=[
                    [
                        # [('dilation', ('hstick', 5)), ('dilation', ('hstick', 5)), 'intersection'],
                        [(op, selem) for op, selem in zip(ops, selems)] + [aggreg]
                    ]
                ]
            ))
