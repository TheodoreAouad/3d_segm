import torch
import torch.nn as nn

from ..threshold_fn import *
from .threshold_layer import dispatcher


class ComplementationLayer(nn.Module):

    def __init__(self, threshold_mode='sigmoid', alpha_init: float = 0, invert_thresholded_alpha: bool = False):
        """
        Initialize the threshold layer

        Args:
            self: write your description
            threshold_mode: write your description
            alpha_init: write your description
            invert_thresholded_alpha: write your description
        """
        super().__init__()
        self.invert_thresholded_alpha = invert_thresholded_alpha
        if isinstance(alpha_init, nn.Parameter):
            self.alpha = alpha_init
        else:
            self.alpha = nn.Parameter(torch.tensor([alpha_init]).float())
        # self.threshold_fn = getattr(self, threshold_mode + '_threshold')
        self.threshold_layer = dispatcher[threshold_mode](P_=1, constant_P=True)

    def forward(self, x):
        """
        Forward Rosenblatt transformation.

        Args:
            self: write your description
            x: write your description
        """
        return self.thresholded_alpha * x + (1 - self.thresholded_alpha) * (1 - x)


    @property
    def thresholded_alpha(self):
        """
        Returns the thresholded alpha.

        Args:
            self: write your description
        """
        if self.invert_thresholded_alpha:
            return 1 - self.threshold_layer(self.alpha)
        return self.threshold_layer(self.alpha)

    def sigmoid_threshold(self, x):
        """
        Return the sigmoid threshold for the given value.

        Args:
            self: write your description
            x: write your description
        """
        return sigmoid_threshold(x)

    def arctan_threshold(self, x):
        """
        Return the arctan threshold for the given value.

        Args:
            self: write your description
            x: write your description
        """
        return arctan_threshold(x)

    def tanh_threshold(self, x):
        """
        Return the tanh threshold for the given value.

        Args:
            self: write your description
            x: write your description
        """
        return tanh_threshold(x)

    def erf_threshold(self, x):
        """
        Threshold for the ERF function.

        Args:
            self: write your description
            x: write your description
        """
        return erf_threshold(x)
