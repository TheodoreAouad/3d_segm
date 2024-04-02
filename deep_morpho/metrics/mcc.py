import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, f1_score

def mcc_quotient(tp, tn, fp, fn):
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den == 0:
        den = 1
    return (tp * tn - fp * fn) / np.sqrt(den)

def matthewscc_custom(y_pred, y_true):
    """Computes Matthew Correlation Coefficient.

    Args:
        outputs_orig (torch.tensor): arbitrary size. Tensor of 0s and 1s. Predictions.
        targets_orig (torch.tensor): same size as outputs_orig. Tensor of 0s and 1s. True values.

    Returns:
        torch.tensor: tensor size (0) of accuracy.
    """

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    return mcc_quotient(tp, tn, fp, fn)

def matthewscc(y_pred, y_true):
    if type(y_pred) == torch.Tensor:
        y_pred = y_pred.cpu().numpy()

    if type(y_true) == torch.Tensor:
        y_true = y_true.cpu().numpy()

    if y_pred.max() <= 1 and y_true.max() <= 1:
        return matthewscc_custom(y_pred, y_true)

    return matthews_corrcoef(y_pred, y_true)
