from numpy import ndarray


def dice(ytrue: ndarray, ypred: ndarray, SMOOTH: float = 1e-6,):
    """
    Dice coefficient.

    Args:
        ytrue: (array): write your description
        ypred: (array): write your description
        SMOOTH: (int): write your description
    """
    ytrue = ytrue.astype(bool)
    ypred = ypred.astype(bool)
    intersection = (ytrue & ypred).sum()

    return (
        (2*intersection + SMOOTH) / (ytrue.sum() + ypred.sum() + SMOOTH)
    )
