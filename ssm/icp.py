''' Taken from https://github.com/ClayFlannigan/icp/blob/master/icp.py
'''
import random

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .utils import get_norm_transform, transform_cloud


def best_fit_transform(A, B, allow_reflection=False):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if not allow_reflection and np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Mxm array of points
    Output:
        distances (size Mx1): Euclidean distances of the nearest neighbor
        indices (size Mx1): dst indices of the nearest neighbor
    '''

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, n_points=None, allow_reflection=False, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        n_points: number of random points. If none given, the max will be taken.
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    # assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    if n_points is None:
        n_points = min(A.shape[0], B.shape[0])
    n_points = min(A.shape[0], B.shape[0], n_points)

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points

        src_points = np.array(sorted(random.sample(range(src.shape[1]), n_points)))
        dst_points = np.array(sorted(random.sample(range(dst.shape[1]), n_points)))

        small_src = src[:m, src_points]
        small_dst = dst[:m, dst_points]

        distances, indices = nearest_neighbor(small_src.T, small_dst.T)


        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(small_src.T, small_dst[:, indices].T, allow_reflection=allow_reflection)


        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T, allow_reflection=allow_reflection)

    return T, distances, i


def register_icp(
    ref_verts: np.ndarray,
    src_verts: np.ndarray,
    allow_reflection: bool = False,
    init_pose=None,
    max_iterations: int = 1000,
    tolerance: float = 1e-5,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Performs ICP registration onto ref_verts (Shape = T @ ref)
    Args:
        src_verts (np.ndarray): size (Mxd). Coint cloud to register onto
        ref_verts (np.ndarray): size (Nxd). Coint cloud to register onto
        allow_reflection (bool): if False, will remove reflections
        init_pose (np.ndarray): initial transform
        max_iterations (int): max number of iterations for ICP algorithm
        tolerance (float): min accepted difference of tansform between two iterations of ICP to converge

    Returns:
        np.ndarray, np.ndarray, int: Transform of size (d+1) x (d+1).
                                    errors of size (N). number of iterations to converge.
    """
    Tnorm_ref = get_norm_transform(ref_verts.mean(0), ref_verts.std(0))
    nref_verts = transform_cloud(Tnorm_ref, ref_verts)

    src_mean, src_std = src_verts.mean(0), src_verts.std(0)
    nverts = (src_verts - src_mean) /src_std
    Tnorm_inv = get_norm_transform(src_mean, src_std, invert=True)


    T, errs, n_iters = icp(
        nref_verts, nverts, allow_reflection=allow_reflection, init_pose=None, max_iterations=max_iterations, tolerance=tolerance
    )
    Tref = Tnorm_inv @ T @ Tnorm_ref

    return Tref, errs, n_iters
