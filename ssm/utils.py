import re

import numpy as np


def get_norm_transform(mean: np.ndarray, std: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Args:
        mean (np.ndarray): d matrix, with d the number of dimension
        std (np.ndarray): d matrix, with d the number of dimension
        invert (bool): undo the normalization

    Returns:
        np.ndarray: (d+1) x (d+1) matrix. The linear operation to apply to normalize by mean and std.
    """
    Tn = np.eye(4)

    if invert:
        Tn[:-1, -1] += mean
        Tn[:3, :3] *= std
    else:
        Tn[:-1, -1] -= mean/std
        Tn[:3, :3] /= std
    return Tn


def transform_cloud(T: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """
    Applies the (d+1) x (d+1) linear transform matrix to an array.

    Args:
        T (np.ndarray): size (d+1) x (d+1). The transform matrix.
        mat (np.ndarray): size (Nxd). The transform matrix.

    Returns:
        np.ndarray: size (Nxd). the points of mat with transform T applied.
    """
    return ((T @ np.hstack((mat, np.ones((mat.shape[0], 1)))).T).T)[:, :-1]


def sort_by_regex(lis, regex=r'labels-(\d+)'):
    return sorted(lis, key=lambda x: int(re.findall(regex, x)[0]))


def random_color_generator(size: int, alpha: float = 1, RGB: bool = False) -> np.ndarray:
    colors = np.ones((size, 4)) * alpha
    if RGB:
        colors[:, :-1] = np.random.rand(size, 3)
    else:
        grey_values = np.random.rand(size, 1)
        colors[:, :-1] = np.concatenate([grey_values for _ in range(3)], axis=1)
    # colors[:, :-1] /= colors[:, :-1].sum(1)
    return colors
