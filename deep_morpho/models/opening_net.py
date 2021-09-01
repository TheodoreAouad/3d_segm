import torch.nn as nn

from .bise import BiSE, LogicalNotBiSE


class OpeningNet(nn.Module):

    def __init__(self, logical_not: bool = False, share_weights: bool = True, **bise_args):
        super().__init__()
        self.share_weights = share_weights
        self.logical_not = logical_not

        layer = BiSE if not logical_not else LogicalNotBiSE

        self.dilations1 = layer(**bise_args)
        if self.share_weights:
            self.dilations2 = layer(shared_weights=self.dilations1.weights, shared_weight_P=self.dilations1.weight_P, **bise_args)
        else:
            self.dilations2 = layer(**bise_args)

        self.bises = [self.dilations1, self.dilations2]

    def forward(self, x):
        output = self.bises[0](x)
        output = self.bises[1](output)
        return output
