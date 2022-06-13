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

# morp_operations.append(ParallelMorpOperations.closing(('disk', 3)))
# morp_operations.append(ParallelMorpOperations.closing(('disk', 3)))


# morp_operations.append(ParallelMorpOperations.remove_isolated_points())

# morp_operations.append(ParallelMorpOperations.closing(('disk', 3)))
# morp_operations.append(ParallelMorpOperations.opening(('disk', 3)))
# morp_operations.append(ParallelMorpOperations.closing(('hstick', 7)))
# morp_operations.append(ParallelMorpOperations.closing(('dcross', 7)))
# morp_operations.append(ParallelMorpOperations.opening(('disk', 3)))
# morp_operations.append(ParallelMorpOperations.opening(('dcross', 7)))
# morp_operations.append(ParallelMorpOperations.white_tophat(('disk', 3)))
# morp_operations.append(ParallelMorpOperations.white_tophat(('hstick', 7)))
# morp_operations.append(ParallelMorpOperations.black_tophat(('dcross', 7)))



# morp_operations.append(ParallelMorpOperations.dilation(('hstick', 7)))
# morp_operations.append(ParallelMorpOperations.translation(1, 2))
# morp_operations.append(ParallelMorpOperations.concatenate(
#     ParallelMorpOperations.translation(1, 2),
#     ParallelMorpOperations.dilation(('hstick', 7)),
#     # ParallelMorpOperations.dilation(('disk', 3)),
#     # ParallelMorpOperations.dilation(('dcross', 7)),
# ))
# morp_operations.append(ParallelMorpOperations.concatenate(
#     ParallelMorpOperations.translation(1, 2),
#     # ParallelMorpOperations.dilation(('hstick', 7)),
#     # ParallelMorpOperations.dilation(('disk', 3)),
#     ParallelMorpOperations.dilation(('dcross', 7)),
# ))
# morp_operations.append(ParallelMorpOperations.translation(2, 1))
# morp_operations.append(ParallelMorpOperations.translation(0, 3))
# morp_operations.append(ParallelMorpOperations.translation(3, 0))
# morp_operations.append(ParallelMorpOperations.concatenate(
#     ParallelMorpOperations.translation(1, 1),
#     [
#         ParallelMorpOperations.translation(0, 1),
#         ParallelMorpOperations.translation(1, 0),
#         ParallelMorpOperations.translation(1, 1),
#         ParallelMorpOperations.translation(1, 1),
#         ParallelMorpOperations.translation(1, 1),
#         ParallelMorpOperations.translation(1, 1),
#         ParallelMorpOperations.translation(1, 1),
#         ParallelMorpOperations.translation(1, 1),
#         ParallelMorpOperations.translation(1, 1),
#     ],
#     name="mutli_translation",
#     # ParallelMorpOperations.dilation((op, size), name=f"dilation/{op}")
# ))


for op in [
    'disk',
    "hstick",
    "dcross",
    # 'hstick', 'vstick', 'scross', 'dcross', 'square'
]:
    size = 5
    if op == "disk":
        size = size // 2

    # morp_operations.append(ParallelMorpOperations.dilation((op, size), name=f"dilation/{op}"))
    # morp_operations.append(ParallelMorpOperations.erosion((op, size), name=f"erosion/{op}"))
    # morp_operations.append(ParallelMorpOperations.closing((op, size), name=f"closing/{op}"))
    # morp_operations.append(ParallelMorpOperations.opening((op, size), name=f"opening/{op}"))
    morp_operations.append(ParallelMorpOperations.white_tophat((op, size), name=f"white_tophat/{op}"))
    # morp_operations.append(ParallelMorpOperations.black_tophat((op, size), name=f"black_tophat/{op}"))

    # morp_operations.append(ParallelMorpOperations.concatenate(ParallelMorpOperations.translation(2, 1), ParallelMorpOperations.dilation((op, size), name=f"dilation/{op}")))
    # morp_operations.append(ParallelMorpOperations.concatenate(ParallelMorpOperations.translation(2, 1), ParallelMorpOperations.erosion((op, size), name=f"erosion/{op}")))
    # morp_operations.append(ParallelMorpOperations.concatenate(ParallelMorpOperations.translation(2, 1), ParallelMorpOperations.closing((op, size), name=f"closing/{op}")))
    # morp_operations.append(ParallelMorpOperations.concatenate(ParallelMorpOperations.translation(2, 1), ParallelMorpOperations.opening((op, size), name=f"opening/{op}")))
    # morp_operations.append(ParallelMorpOperations.concatenate(ParallelMorpOperations.translation(2, 1), ParallelMorpOperations.white_tophat((op, size), name=f"white_tophat/{op}")))
    # morp_operations.append(ParallelMorpOperations.concatenate(ParallelMorpOperations.translation(2, 1), ParallelMorpOperations.black_tophat((op, size), name=f"black_tophat/{op}")))
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
