import torch.nn as nn

from .bise import BiSE


class OpeningNet(nn.Module):

    def __init__(self, share_weights=False, **bise_args):
        super().__init__()
        self.share_weights = share_weights
        self.dilations1 = BiSE(**bise_args)
        if self.share_weights:
            self.dilations2 = BiSE(shared_weights=self.dilations1.weights, shared_weight_P=self.dilations1.weight_P, **bise_args)
        else:
            self.dilations2 = BiSE(**bise_args)

        self.dilations = [self.dilations1, self.dilations2]

    def forward(self, x):
        output = self.dilations[0](x)
        output = self.dilations[1](output)
        return output
