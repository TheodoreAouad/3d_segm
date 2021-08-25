import numpy as np
import torch
from skimage.morphology import disk

from general.array_morphology import SequentialMorpOperations

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

morp_operations = []

# selems = []

# selem = np.zeros((7, 7))
# selem[3, :] = 1
# selems.append(selem)

# selem = np.zeros((7, 7))
# selem[:, 3] = 1
# selems.append(selem)

seq_morp = SequentialMorpOperations(
    name='Bimonn_erosion',
    selems=[disk(3)],
    operations=['erosion'],
    return_numpy_array=False,
    device=device,
)
morp_operations.append(seq_morp)

# seq_morp = SequentialMorpOperations(
#     name='Bimonn_closing',
#     selems=[disk(3), disk(3)],
#     operations=['dilation', 'erosion'],
#     return_numpy_array=False,
#     device=device,
# )
# morp_operations.append(seq_morp)
