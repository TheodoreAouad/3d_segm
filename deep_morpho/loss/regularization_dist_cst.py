from typing import List

import torch
import torch.nn as nn

from ..models import BiSEBase
from ..binarization.bise_closest_selem import BiseClosestMinDistOnCst


class RegularizationProjConstant(nn.Module):
    """ Adds a regularization loss to encourage the bimonn to be morphological.
    For a structuring element $S$, $A(S) = \{\theta \cdot S | \theta > 0\}$. We compute
    $$ \min_S d(A(S), (W, B)) $$.
    """
    def __init__(self, model: nn.Module = None, bise_modules: List[BiSEBase] = None):
        super().__init__()
        self.model = model

        if bise_modules is None:
            bise_modules = [m for m in self.model.modules() if isinstance(m, BiSEBase)]

        self.bise_modules = bise_modules
        self.bise_closest_handlers = [BiseClosestMinDistOnCst(bise) for bise in bise_modules]


    def forward(self, *args, **kwargs) -> torch.Tensor:
        loss = 0
        for handler in self.bise_closest_handlers:
            loss += handler(verbose=False)
        return loss
