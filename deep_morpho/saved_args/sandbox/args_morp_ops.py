import numpy as np
import torch
from general.structuring_elements import *

from deep_morpho.morp_operations import SequentialMorpOperations, ParallelMorpOperations

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

morp_operations = []
selems = []

morp_operations.append(ParallelMorpOperations(
    name='dilation_union',
    operations=[
        [
            [('dilation', ('hstick', 5)), ('dilation', ('hstick', 5)), 'intersection'],
            # [('dilation', ('hstick', 5)), 'union']
        ]
    ]
))

# sizes = [5, 7]
# for size in [7]:
#     for se in [
#         # disk,
#         # hstick,
#         # vstick,
#         # diagonal_cross,
#         straight_cross,
#         # square
#     ]:
#         cur_size = size if se.__name__ != 'disk' else size // 2
#         for op in [
#             'erosion',
#             # 'dilation'
#         ]:
#             morp_operations.append(SequentialMorpOperations(
#                 name=op + f"_size_{size}x{size}",
#                 selems=[(se, cur_size)],
#                 operations=[op],
#                 return_numpy_array=False,
#                 device='cpu'
#             ))
        # morp_operations.append(SequentialMorpOperations(
        #     name=f"opening_size_{size}x{size}",
        #     selems=[(se, cur_size) for _ in range(2)],
        #     operations=['erosion', 'dilation'],
        #     device='cpu'
        # ))
        # morp_operations.append(SequentialMorpOperations(
        #     name=f"closing_size_{size}x{size}",
        #     selems=[(se, cur_size) for _ in range(2)],
        #     operations=['dilation', 'erosion'],
        #     device='cpu'
        # ))

# morp_operations.append(SequentialMorpOperations(
#     name="sandbox_closing",
#     selems=[("square", 7), ("square", 7)],
#     operations=['dilation', 'erosion'],
#     device="cpu",
# ))

# morp_operations.append(SequentialMorpOperations(
#     name="dilation",
#     selems=[("disk", 3)],
#     operations=['dilation'],
#     device="cpu",
# ))
