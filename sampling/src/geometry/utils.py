from typing import List, Union, Tuple, Optional

import numpy as np


def get_e2(e1: np.ndarray):
    """
    Gives a vector of norm 1 orthogonal to e1.
    """
    return np.array([e1[2], e1[2], -(e1[0]+e1[1])]) / np.sqrt(2*e1[2]**2 + (e1[0] + e1[1])**2)


def get_e3(e1: np.ndarray, e2: np.ndarray):
    """
    Gives the vector of norm 1 orthogonal to e1 and e2 to make a positive orthonormal basis
    """
    return np.cross(e1, e2)


def get_on_basis(v1: np.ndarray):
    """
    Returns 3 vectors of orthogonal basis given v1, where the first vector has the same orentation as v1
    """
    e1 = v1 / np.linalg.norm(v1)
    e2 = get_e2(e1)
    return e1, e2, get_e3(e1, e2)


def get_rotation_matrix(vs: Union[List, Tuple, np.ndarray]):
    """
    Get a rotation matrix.
    """
    # If vs is one array, we get a full orthonormal basis from it.
    if type(vs) == np.ndarray:
        e1, e2, e3 = get_on_basis(vs)
    # If vs is two arrays, we get complete it into an orthonormal basis.
    if len(vs) == 1:
        e1, e2, e3 = get_on_basis(vs[0])
    elif len(vs) == 2:
        e1, e2 = vs
        e1 /= np.linalg.norm(e1)
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
    # If vs is already 3 vectors, we get the rotation matrix directly.
    else:
        e1, e2, e3 = vs
        e1 /= np.linalg.norm(e1)
        e2 /= np.linalg.norm(e2)
        e3 /= np.linalg.norm(e3)
    return np.vstack((e1, e2, e3)).T


def sample_plane_according_to_vector(
        n: int,
        vs: np.ndarray,
        position: float,
        width: int,
        length: int,
        height: int,
        edge_size: Optional[float] = None
):
    """
    Sample uniformly n 3D points along the plane of normal vector v. The plane is centered around the width and height.
    The position according to axis z is given. This function's purpose is to preceed the intersection of the plane and
    an image cuboid.

    Args:
        n (int): number of samples for each coordinate
        vs (ndarray): 3d vector. Normal vector to the plane.
        position (float): position according to z-axis (last dimension of the cuboid image)
        width (int): width of the targeted image
        length (int): length of the targeted image
        height (int): height of the targeted image
        edge_size (float, optional): size of the edge of the plane. Defaults to None.

    Returns:
        ndarray, ndarray, ndarray: size (3, n**2): coordinates of the samples, size(n**2):
        x-indexes of the samples. size (n**2): y-indexes of the samples.
    """
    if edge_size is None:
        edge_size = np.sqrt(length**2 + height**2) * 1.5

    R = get_rotation_matrix(vs)
    # samples = np.random.rand(2, n) * np.sqrt(3) * np.array([[width], [length]])
    X, Y = np.mgrid[0:n, 0:n]
    samples = np.c_[X[X == X] * edge_size / (n-1), Y[Y == Y] * edge_size / (n-1)].T
    samples = np.vstack((samples, np.zeros(n**2)))
    samples = R @ samples
    # samples = samples + (d / v.sum())
    hor_correction = np.array([
        [width // 2 - samples[0].mean()],
        [length // 2 - samples[1].mean()],
        [height // 2 - samples[2].mean()]])
    normal_vec = R[:, -1][:, np.newaxis]

    samples += (hor_correction + normal_vec * position)
    return samples, X[X == X], Y[Y == Y]



def get_pos_point_on_segment(x: np.ndarray, u1: np.ndarray, u2: np.ndarray):
    """
    Gets the position t of point x on the segment [u1, u2] such that
    x = u1 + t * (u2 - u1)

    Args:
        x (ndarray): point which we want position of. Can be one vector of multiple vectors.
        u1 (ndarray): beginning of the segment
        u2 (ndarray): end of the segment

    Returns:
        float: float between 0 and 1 giving the position of x. If multiple vectors given, returns array of float.
    """
    if len(x.shape) == 1:
        return (x - u1) @ (u2 - u1) / ((u2 - u1) @ (u2 - u1))

    dimu = u1.shape[0]
    if type(x) == list:
        Xs = np.stack(x, 0)
    elif x.shape == (dimu, dimu):
        Xs = x + 0
    elif x.shape[0] == dimu:
        Xs = x.T
    elif x.shape[1] == dimu:
        Xs = x.T
    else:
        assert False, 'Wrong dimensions given: {} incompatible with {}'.format(x.shape, u1.shape)
    assert Xs.shape[1] == dimu

    return (Xs - u1) @ (u2 - u1) / ((u2 - u1) @ (u2 - u1))


def deg(theta: float):
    """
    Convert angle in degrees to degrees.

    Args:
        theta: write your description
    """
    return theta * 360/(2*np.pi)


def rad(theta: float):
    """
    Convert angle in degrees to radians.

    Args:
        theta: write your description
    """
    return (theta * 2*np.pi / 360) % (2*np.pi)


def get_mean_nonzero(ar: np.ndarray):
    """
    Get the mean of the non - zero values in an array.

    Args:
        ar: write your description
        np: write your description
        ndarray: write your description
    """
    xs = np.where(ar != 0)[0]
    return (xs.max() + xs.min()) / 2


def get_barycenter_nonzero_3d(ar: np.ndarray):
    """
    Calculate the barycenter of the nonzeros in 3d

    Args:
        ar: write your description
        np: write your description
        ndarray: write your description
    """
    res = np.zeros(3)
    res[0] = get_mean_nonzero(ar.sum(1).sum(1))
    res[1] = get_mean_nonzero(ar.sum(0).sum(1))
    res[2] = get_mean_nonzero(ar.sum(0).sum(0))
    
    return res


def spheric_coord(r: float, theta: float, phi: float):
    """
    Return the spheric coordinates of a spheric curve.

    Args:
        r: write your description
        theta: write your description
        phi: write your description
    """
    assert r > 0
    assert 0 <= phi <= np.pi

    return np.array([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ])
