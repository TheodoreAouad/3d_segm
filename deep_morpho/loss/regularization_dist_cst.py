from typing import List, Dict

import torch
import torch.nn as nn

from ..models import BiSEBase
from ..binarization.bise_closest_selem import BiseClosestMinDistOnCst
from ..binarization.projection_constant_set import ProjectionConstantSet


class RegularizationProjConstant(nn.Module):
    """ Adds a regularization loss to encourage the bimonn to be morphological.
    For a structuring element $S$, $A(S) = \{\theta \cdot S | \theta > 0\}$. We compute
    $$ \min_S d(A(S), (W, B)) $$.
    """
    def __init__(self, model: nn.Module = None, bise_modules: List[BiSEBase] = None):
        super().__init__()
        self.model = model
        self.bise_modules = bise_modules

    def forward(self, *args, pl_module=None, **kwargs) -> torch.Tensor:
        if self.model is None:
            self.set_model(pl_module.model)

        loss = 0
        for handler in self.bise_closest_handlers:
            _, _, dist = handler(return_np_array=False, verbose=False)
            loss += dist.sum()
        return loss

    def set_model(self, model: nn.Module):
        self.model = model
        if self.bise_modules is None:
            self.bise_modules = [m for m in self.model.modules() if isinstance(m, BiSEBase)]

        self.bise_closest_handlers = [BiseClosestMinDistOnCst(bise) for bise in self.bise_modules]
        return self


# DEPRECATED: the cat does not share memory, the gradient are not passed correctly. See how to define the projectors
#  at each iteration without being too slow.
# class RegularizationProjConstantVectorized(nn.Module):
#     """ Adds a regularization loss to encourage the bimonn to be morphological.
#     For a structuring element $S$, $A(S) = \{\theta \cdot S | \theta > 0\}$. We compute
#     $$ \min_S d(A(S), (W, B)) $$.
#     """
#     def __init__(self, model: nn.Module = None):
#         super().__init__()
#         self.model = model
#         self.weights_dict: Dict[int, torch.Tensor] = dict()
#         self.bias_dict: Dict[int, torch.Tensor] = dict()
#         self.projectors: List[ProjectionConstantSet] = []

#     def forward(self, *args, pl_module=None, **kwargs) -> torch.Tensor:
#         if self.model is None:
#             self.set_model(pl_module.model)

#         loss = 0
#         for proj in self.projectors:
#             proj.compute(verbose=False)
#             loss += proj.final_dist.sum()
#         return loss

#     def set_model(self, model: nn.Module):
#         self.model = model
#         for bise_module in self.model.modules():
#             if not isinstance(bise_module, BiSEBase):
#                 continue
#             W = bise_module.weights.reshape(bise_module.weights.shape[0], -1)
#             dim = W.shape[1]
#             self.weights_dict[dim] = self.weights_dict.get(dim, []) + [W]
#             self.bias_dict[dim] = self.bias_dict.get(dim, []) + [bise_module.bias]

#         self.projectors = [ProjectionConstantSet(
#             weights=torch.cat(self.weights_dict[dim], axis=0), bias=torch.cat(self.bias_dict[dim], axis=0)
#         ) for dim in self.weights_dict.keys()]

#         # self.bise_closest_handlers = [BiseClosestMinDistOnCst(bise) for bise in self.bise_modules]
#         return self


# class RegularizationProjConstant(RegularizationProjConstantOld):
# # class RegularizationProjConstant(RegularizationProjConstantNew):
#     pass
