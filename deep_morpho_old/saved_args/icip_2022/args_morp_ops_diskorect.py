import numpy as np
import torch
from general.structuring_elements import *

from deep_morpho_old.morp_operations import ParallelMorpOperations, ParallelMorpOperationsGrayTopHats

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

morp_operations = []

for op in [
    # 'disk', 'hstick', 'vstick', 'scross', 'dcross', 'square'
    # "disk",
    # "hstick",
    "dcross",
]:
    size = 7
    if op == "disk":
        size = size // 2

    morp_operations.append(ParallelMorpOperations.dilation((op, size)))
    # morp_operations.append(ParallelMorpOperations.erosion((op, size)))
    # morp_operations.append(ParallelMorpOperations.closing((op, size)))
    # morp_operations.append(ParallelMorpOperations.opening((op, size)))


# for ops in [
#     ("dilation", "dilation", "dilation", "intersection"),
#     ("erosion", "erosion", "erosion", "union"),
# ]:
#     aggreg = ops[-1]
#     ops = ops[:-1]
#     for selems in [
#         (('dcross', 7), ('hstick', 7), ('vstick', 7)),
#         (('scross', 7), ('vstick', 7), ('square', 7)),
#     ]:
#         morp_operations.append(ParallelMorpOperations(
#             name=f'{aggreg}',
#             operations=[
#                 [
#                     [('dilation', ('identity', 7), False), 'union'] for _ in range(len(ops))
#                 ],
#                 [
#                     # [('dilation', ('hstick', 5)), ('dilation', ('hstick', 5)), 'intersection'],
#                     [(op, selem) for op, selem in zip(ops, selems)] + [aggreg]
#                 ],
#             ]
#         ))
#         morp_operations.append(ParallelMorpOperations(
#             name=f'{aggreg}',
#             operations=[
#                 # [
#                 #     [('dilation', ('identity', 7), False), 'union'] for _ in range(len(ops))
#                 # ],
#                 [
#                     # [('dilation', ('hstick', 5)), ('dilation', ('hstick', 5)), 'intersection'],
#                     [(op, selem) for op, selem in zip(ops, selems)] + [aggreg]
#                 ],
#             ]
#         ))
