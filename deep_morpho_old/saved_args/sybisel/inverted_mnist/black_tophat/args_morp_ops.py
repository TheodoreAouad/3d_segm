import numpy as np
import torch
from general.structuring_elements import *

from deep_morpho_old.morp_operations import ParallelMorpOperations, ParallelMorpOperationsGrayTopHats

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

morp_operations = []
selems = []


for op in [
    'disk',
    "hstick",
    "dcross",
    "bsquare",
    "bdiamond",
    "bcomplex",
    "scross"
]:
    size = 7
    if op == "disk":
        size = size // 2

    # morp_operations.append(ParallelMorpOperations.dilation((op, size), name=f"dilation/{op}"))
    # morp_operations.append(ParallelMorpOperations.erosion((op, size), name=f"erosion/{op}"))
    # morp_operations.append(ParallelMorpOperations.closing((op, size), name=f"closing/{op}"))
    # morp_operations.append(ParallelMorpOperations.opening((op, size), name=f"opening/{op}"))
    # morp_operations.append(ParallelMorpOperations.white_tophat((op, size), name=f"white_tophat/{op}"))
    morp_operations.append(ParallelMorpOperations.black_tophat((op, size), name=f"black_tophat/{op}"))
