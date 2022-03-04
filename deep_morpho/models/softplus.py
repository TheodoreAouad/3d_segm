import torch
import torch.nn as nn


class Softplus(nn.Softplus):

    def forward_inverse(self, x):
        return 1 / self.beta * torch.log(torch.exp(self.beta * x) - 1)
