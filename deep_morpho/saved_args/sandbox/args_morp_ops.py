import numpy as np
import torch
from general.structuring_elements import *

from general.array_morphology import SequentialMorpOperations

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

morp_operations = []
selems = []

# vstick
# selem = np.zeros((7, 7))
# selem[:, 3] = 1
# selems.append(selem)

# hstick
# selem = np.zeros((7, 7))
# selem[3, :] = 1
# selems.append(selem)

# for se in [
#     disk(7//2),
#     hstick(7),
#     vstick(7),
#     diagonal_cross(7), straight_cross(7), square(7),
# ]:
#     for op in ['dilation']:
#         morp_operations.append(SequentialMorpOperations(
#             name=op + "_7x7",
#             selems=[se],
#             operations=[op],
#             return_numpy_array=False,
#             device='cpu'
#         ))

# sizes = [5, 7]
for size in [7]:
    for se in [
        disk,
        # hstick,
        # vstick,
        # diagonal_cross,
        # straight_cross,
        # square
    ]:
        cur_size = size if se.__name__ != 'disk' else size // 2
        # for op in [
        #     'erosion',
        #     'dilation'
        # ]:
        #     morp_operations.append(SequentialMorpOperations(
        #         name=op + f"_size_{size}x{size}",
        #         selems=[(se, cur_size)],
        #         operations=[op],
        #         return_numpy_array=False,
        #         device='cpu'
        #     ))
        # morp_operations.append(SequentialMorpOperations(
        #     name=f"opening_size_{size}x{size}",
        #     selems=[(se, cur_size) for _ in range(2)],
        #     operations=['erosion', 'dilation'],
        #     device='cpu'
        # ))
        morp_operations.append(SequentialMorpOperations(
            name=f"closing_size_{size}x{size}",
            selems=[(se, cur_size) for _ in range(2)],
            operations=['dilation', 'erosion'],
            device='cpu'
        ))

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
