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
    # 'disk', 'hstick', 'vstick', 'scross', 'dcross', 'square'
    "disk",
    "hstick",
    "dcross",
]:
    size = 7
    if op == "disk":
        size = size // 2

    # morp_operations.append(ParallelMorpOperations.dilation((op, size)))
    # morp_operations.append(ParallelMorpOperations.erosion((op, size)))
    # morp_operations.append(ParallelMorpOperations.closing((op, size)))
    morp_operations.append(ParallelMorpOperations.opening((op, size)))