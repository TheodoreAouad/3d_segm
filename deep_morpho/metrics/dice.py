def dice(y_true, y_pred, threshold=.5, SMOOTH=1e-6,):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    targets = (y_true != 0)
    if threshold is None:
        outputs = y_pred != 0
    else:
        outputs = y_pred > threshold

    intersection = (outputs & targets).float().sum((1, 2))

    return (
        (2*intersection + SMOOTH) / (targets.sum((1, 2)) + outputs.sum((1, 2)) + SMOOTH)
    ).detach().cpu().numpy()
