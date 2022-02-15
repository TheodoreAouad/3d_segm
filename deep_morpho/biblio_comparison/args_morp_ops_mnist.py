import numpy as np
import torch
from general.structuring_elements import *

from deep_morpho.morp_operations import ParallelMorpOperations

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

morp_operations = []

for op in [
    # 'disk', 'hstick', 'vstick', 'scross',
    # 'dcross', 'square'
    "disk", "hstick", "dcross",
]:
    size1 = 5
    size2 = 7
    if op == "disk":
        size1 = size1 // 2
        size2 = size2 // 2

    morp_operations.append(ParallelMorpOperations.dilation((op, size1)))
    morp_operations.append(ParallelMorpOperations.erosion((op, size1)))
    morp_operations.append(ParallelMorpOperations.closing((op, size2)))
    morp_operations.append(ParallelMorpOperations.opening((op, size2)))
