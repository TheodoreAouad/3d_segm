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

# morp_operations.append(ParallelMorpOperations.white_tophat(('disk', 3)))
# morp_operations.append(ParallelMorpOperations.white_tophat(('hstick', 7)))
# morp_operations.append(ParallelMorpOperations.white_tophat(('dcross', 7)))
# morp_operations.append(ParallelMorpOperations.white_tophat(('disk', 3)))



# morp_operations.append(ParallelMorpOperations.closing(('hstick', 7)))
# morp_operations.append(ParallelMorpOperations.closing(('dcross', 7)))
# morp_operations.append(ParallelMorpOperations.opening(('hstick', 7)))
# morp_operations.append(ParallelMorpOperations.opening(('dcross', 7)))
# morp_operations.append(ParallelMorpOperations.dilation(('disk', 3)))
# morp_operations.append(ParallelMorpOperations.dilation(('hstick', 7)))

morp_operations.append(ParallelMorpOperations.closing(('disk', 3)))
morp_operations.append(ParallelMorpOperations.closing(('hstick', 7)))
morp_operations.append(ParallelMorpOperations.closing(('dcross', 7)))


# for op in [
#     'disk', "hstick", "dcross",
#     # 'hstick', 'vstick', 'scross', 'dcross', 'square'
# ]:
#     # size1 = 5
#     size2 = 7
#     if op == "disk":
# #         size1 = size1 // 2
#         size2 = size2 // 2

#     morp_operations.append(ParallelMorpOperations.dilation((op, size2)))
#     morp_operations.append(ParallelMorpOperations.erosion((op, size2)))
#     morp_operations.append(ParallelMorpOperations.closing((op, size2)))
#     morp_operations.append(ParallelMorpOperations.opening((op, size2)))
#     morp_operations.append(ParallelMorpOperations.white_tophat((op, size2)))
#     morp_operations.append(ParallelMorpOperations.black_tophat((op, size2)))
# # morp_operations.append(ParallelMorpOperations.dilation(('disk', 2)))
# morp_operations.append(ParallelMorpOperations.dilation(('disk', 2)))

# morp_operations.append(ParallelMorpOperations.complementation(size=5))
# morp_operations.append(ParallelMorpOperations.identity(size=7))
# morp_operations.append(ParallelMorpOperations(
#     name="complementation",
#     operations=[[[('dilation', ('identity', 7), True), 'union']]]
# ))


# for ops in [
#     ("dilation", "erosion", "dilation"),
# ]:
#     for selems in [
#         (('dcross', 7), ('hstick', 7), ("scross", 7)),
#     ]:
#         for aggreg in ["union", "intersection"]:

#             morp_operations.append(ParallelMorpOperations(
#                 name=f'{aggreg}',
#                 operations=[
#                     [
#                         # [('dilation', ('hstick', 5)), ('dilation', ('hstick', 5)), 'intersection'],
#                         [(op, selem) for op, selem in zip(ops, selems)] + [aggreg]
#                     ]
#                 ]
#             ))

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
#         # for aggreg in [
#         #     "union",
#         #     "intersection"
#         # ]:
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
