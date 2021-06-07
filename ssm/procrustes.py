import numpy as np


def get_transform_procrustes(ar1, ar2):
    """
    Solves min_T || ar1 @ T - ar2||
    """
    return np.linalg.inv(ar1.T @ ar1) @ ar1.T @ ar2
