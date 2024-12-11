from typing import List

from ..models import BiSE


class ThresholdPenalization:

    def __init__(self, bise_layers: List[BiSE], coef: float = .5, degree: int = 2, detach_weights: bool = True,
                 epsilon: float = .5):
        """
        Initialize the loss function.

        Args:
            self: write your description
            bise_layers: write your description
            coef: write your description
            degree: write your description
            detach_weights: write your description
            epsilon: write your description
        """
        self.bise_layers = bise_layers
        self.coef = coef
        self.loss_fn = getattr(self, f"polynome_{degree}")
        self.detach_weights = detach_weights
        self.epsilon = epsilon

    def __call__(self):
        """
        Compute the loss.

        Args:
            self: write your description
        """
        loss = 0
        for dilation in self.bise_layers:
            sum_weights = (dilation._normalized_weight > .5).sum()
            if self.detach_weights:
                sum_weights = sum_weights.detach()
            loss += self.coef * self.loss_fn(dilation.bias, 1 - self.epsilon, sum_weights - self.epsilon)
        return loss

    @staticmethod
    def polynome_2(bias, x, y):
        """
        Second order polynomial.

        Args:
            bias: write your description
            x: write your description
            y: write your description
        """
        return (x + bias).abs() * (y + bias).abs()

    @staticmethod
    def polynome_4(bias, x, y):
        """
        Solve the equation 4 page 970.

        Args:
            bias: write your description
            x: write your description
            y: write your description
        """
        return (x + bias) ** 2 * (y + bias) ** 2
