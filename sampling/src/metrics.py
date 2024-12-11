from numpy import ndarray


def dice(ytrue: ndarray, ypred: ndarray, SMOOTH: float = 1e-6,):
    """
    Calculates the dice test statistic.

    Args:
        ytrue: write your description
        ypred: write your description
        SMOOTH: write your description
    """
    ytrue = ytrue.astype(bool)
    ypred = ypred.astype(bool)
    intersection = (ytrue & ypred).sum()

    return (
        (2*intersection + SMOOTH) / (ytrue.sum() + ypred.sum() + SMOOTH)
    )
