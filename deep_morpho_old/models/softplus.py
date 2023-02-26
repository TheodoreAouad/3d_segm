import torch.nn as nn

from deep_morpho_old.threshold_fn import softplus_threshold_inverse


class Softplus(nn.Softplus):

    def forward_inverse(self, x):
        return softplus_threshold_inverse(x, self.beta)
