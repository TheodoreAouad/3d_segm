import numpy as np


def get_ellipse_fn(center, matrix, radius):
    """
    Returns a function that returns the angular distances between the ellipse and the specified center and

    Args:
        center: write your description
        matrix: write your description
        radius: write your description
    """
    def fn(*x):
        """
        Function to calculate the center and radius of a sphere.

        Args:
            x: write your description
        """
        W, L, H = x[0].shape
        Z = np.zeros_like(x[0])
        for i in range(W):
            for j in range(L):
                for k in range(H):
                    coord = np.array([x[0][i, j, k], x[1][i, j, k], x[2][i, j, k]]) - center
                    coord = coord[:, np.newaxis]
                    Z[i, j, k] = np.sqrt(coord.T @ matrix @ coord) - radius
        return Z
    return fn


def get_ellipsoid_points(center, matrix, radius, shape=(40, 40, 40), eps=1e-1):
    """
    Get points in ellipse centered at center with radius eps.

    Args:
        center: write your description
        matrix: write your description
        radius: write your description
        shape: write your description
        eps: write your description
    """

    XX, YY, ZZ = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    fn_ellipse = get_ellipse_fn(center, matrix, radius)
    lvset = fn_ellipse(XX, YY, ZZ)
    mask = np.zeros_like(lvset)
    mask[(-eps < lvset) & (lvset < eps)] = 1
    Xs, Ys, Zs = np.where(mask)
    return np.concatenate((Xs[:, np.newaxis], Ys[:, np.newaxis], Zs[:, np.newaxis]), axis=1)
