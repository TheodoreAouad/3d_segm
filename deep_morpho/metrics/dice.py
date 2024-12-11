import numpy as np


def dice(y_true, y_pred, threshold=.5, SMOOTH=1e-6,):
    """
    Dice test for the given threshold.

    Args:
        y_true: write your description
        y_pred: write your description
        threshold: write your description
        SMOOTH: write your description
    """
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    if y_true.ndim == 4:
        return np.stack([dice(y_true[:, k, ...], y_pred[:, k, ...], threshold, SMOOTH) for k in range(y_true.shape[1])], axis=0).mean(0)

    targets = (y_true != 0)
    if threshold is None:
        outputs = y_pred != 0
    else:
        outputs = y_pred > threshold

    intersection = (outputs & targets).float().sum((1, 2))

    return (
        (2*intersection + SMOOTH) / (targets.sum((1, 2)) + outputs.sum((1, 2)) + SMOOTH)
    ).detach().cpu().numpy()
