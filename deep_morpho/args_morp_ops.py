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

for se in [disk(3), hstick(7), vstick(7), diagonal_cross(7), straight_cross(7), square(7)]:
    for op in ['erosion', 'dilation']:
        seq_morp = SequentialMorpOperations(
            name='Bimonn_'+op,
            selems=[se],
            operations=[op],
            return_numpy_array=False,
            device=device
        )
        morp_operations.append(seq_morp)

# morp_operations.append(SequentialMorpOperations(
#     name="Bimonn_opening",
#     selems=[disk(2), disk(2)],
#     operations=['erosion', 'dilation'],
#     return_numpy_array=False,
#     device=device,
# ))
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
#     name='Bimonn_erosion',
#     selems=[disk(2)],
#     operations=['erosion'],
#     return_numpy_array=False,
#     device=device,
# )
# morp_operations.append(seq_morp)

