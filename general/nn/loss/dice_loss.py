import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, reduction='mean'):
        """
        Initialize the estimator.

        Args:
            self: write your description
            reduction: write your description
        """
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, ypred, ytrue):
        """
        Dice score computation

        Args:
            self: write your description
            ypred: write your description
            ytrue: write your description
        """
        ypred = ypred.squeeze()
        ytrue = ytrue.squeeze()
        assert len(ypred.shape) == 3
        assert len(ytrue.shape) == 3

        dice_score = 2 * (ytrue * ypred).sum(1) / ((ytrue ** 2).sum(1) + (ypred ** 2).sum(1) + self.eps)

        return self.reduction_fn(1 - dice_score)

    def reduction_fn(self, x):
        """
        Reduce function.

        Args:
            self: write your description
            x: write your description
        """
        if isinstance(self.reduction, str):
            if self.reduction == "sum":
                return x.sum()
            if self.reduction == "mean":
                return x.mean()
            if self.reduction == "none":
                return x

        return self.reduction(x)
