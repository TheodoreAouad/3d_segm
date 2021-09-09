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

# for se in [disk(3), hstick(7), vstick(7), diagonal_cross(7), straight_cross(7), square(7)]:
#     for op in ['erosion', 'dilation']:
#         morp_operations.append(SequentialMorpOperations(
#             name=op + "_7x7",
#             selems=[se],
#             operations=[op],
#             return_numpy_array=False,
#             device=device
#         ))
#
# for se in [disk(2), hstick(5), vstick(5), diagonal_cross(5), straight_cross(5), square(5)]:
#     for op in ['erosion', 'dilation']:
#         morp_operations.append(SequentialMorpOperations(
#             name=op + "_5x5",
#             selems=[se],
#             operations=[op],
#             return_numpy_array=False,
#             device=device
#         ))
    # morp_operations.append(SequentialMorpOperations(
    #     name="opening",
    #     selems=[se, se],
    #     operations=['erosion', 'dilation'],
    #     return_numpy_array=False,
    #     device=device
    # ))
    # morp_operations.append(SequentialMorpOperations(
    #     name="closing",
    #     selems=[se, se],
    #     operations=['dilation', 'erosion'],
    #     return_numpy_array=False,
    #     device=device
    # ))

morp_operations.append(SequentialMorpOperations(
    name="dilation",
    selems=[disk(3)],
    operations=['dilation'],
    device=device,
))
morp_operations.append(SequentialMorpOperations(
    name="erosion",
    selems=[disk(3)],
    operations=['erosion'],
    device=device,
))
#
# morp_operations.append(SequentialMorpOperations(
#     name="Bimonn_closing",
#     selems=[disk(3), disk(3)],
#     operations=['dilation', 'erosion',],
#     return_numpy_array=False,
#     device=device,
# ))

# morp_operations.append(SequentialMorpOperations(
#     name="Bimonn_opening",
#     selems=[disk(3), disk(3)],
#     operations=['erosion', 'dilation'],
#     return_numpy_array=False,
#     device=device,
# ))

# seq_morp = SequentialMorpOperations(
#     name='Bimonn_network/experiment_14',
#     selems=[disk(3), disk(3), straight_cross(7), straight_cross(7)],
#     operations=['erosion', 'dilation', 'erosion', 'dilation'],
#     return_numpy_array=False,
#     device=device,
# )
# morp_operations.append(seq_morp)
#
# seq_morp = SequentialMorpOperations(
#     name='Bimonn_network/experiment_14',
#     selems=[disk(2), diagonal_cross(5), straight_cross(5), straight_cross(5)],
#     operations=['dilation', 'dilation', 'erosion', 'dilation'],
#     return_numpy_array=False,
#     device=device,
# )
# morp_operations.append(seq_morp)
