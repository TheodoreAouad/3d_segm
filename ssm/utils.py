import re

import numpy as np


def array_wo_idx(ar: np.ndarray, idx: int) -> np.ndarray:
    totake = np.ones(len(ar)).astype(bool)
    totake[idx] = False
    return ar[totake]


def is_outlier_1d(ar: np.ndarray, ratio_inter_quartile: float = 1.5) -> np.ndarray:
    q1, q3 = np.quantile(ar, [0.25, 0.75])
    return np.maximum(ar - q3, q1 - ar) - ratio_inter_quartile * (q3 - q1) > 0


# def best_fit_transform(A, B, allow_reflection=False):
#     '''
#     Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
#     Input:
#       A: Nxm numpy array of corresponding points
#       B: Nxm numpy array of corresponding points
#     Returns:
#       T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
#       R: mxm rotation matrix
#       t: mx1 translation vector
#     '''
#
#     assert A.shape == B.shape
#
#     # get number of dimensions
#     m = A.shape[1]
#
#     # translate points to their centroids
#     centroid_A = np.mean(A, axis=0)
#     centroid_B = np.mean(B, axis=0)
#     AA = A - centroid_A
#     BB = B - centroid_B
#
#     # rotation matrix
#     H = np.dot(AA.T, BB)
#     U, S, Vt = np.linalg.svd(H)
#     R = np.dot(Vt.T, U.T)
#
#     # special reflection case
#     if not allow_reflection and np.linalg.det(R) < 0:
#         Vt[m-1, :] *= -1
#         R = np.dot(Vt.T, U.T)
#
#     # translation
#     t = centroid_B.T - np.dot(R, centroid_A.T)
#
#     # homogeneous transformation
#     T = np.identity(m+1)
#     T[:m, :m] = R
#     T[:m, m] = t
#
#     return T, R, t


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
