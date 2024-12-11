import torch
import torch.nn as nn


class PConv2d(nn.Conv2d):

    def __init__(self, P_: float, *args, **kwargs):
        """
        Initialize the model with a single parameter.

        Args:
            self: write your description
            P_: write your description
        """
        super().__init__(bias=0, *args, **kwargs)
        self.P_ = nn.Parameter(torch.tensor([P_]).float())

    def forward(self, x: torch.Tensor):
        """
        Forward propagate the L1 norm of x

        Args:
            self: write your description
            x: write your description
            torch: write your description
            Tensor: write your description
        """
        return self.forward(x**(self.P_ + 1)) / self.forward(x**(self.P_))
