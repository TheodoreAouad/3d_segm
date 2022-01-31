from typing import Tuple, final

import numpy as np
from PIL import Image, ImageDraw
from numba import njit
from general.numba_utils import numba_randint, numba_rand, numba_rand_shape_2d


@njit
def numba_straight_rect(width, height):
    """
    A straight line of a rectangle in numba.

    Args:
        width: write your description
        height: write your description
    """
    return np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])


@njit
def numba_rotation_matrix(theta):
    """
    Generates a rotation matrix for a numba system.

    Args:
        theta: write your description
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


@njit
def numba_transform_rect(rect: np.ndarray, R: np.ndarray, offset: np.ndarray):
    """
    Transform a rectangle with a matrix R using the numba library.

    Args:
        rect: write your description
        np: write your description
        ndarray: write your description
        R: write your description
        np: write your description
        ndarray: write your description
        offset: write your description
        np: write your description
        ndarray: write your description
    """
    return np.dot(rect, R) + offset


@njit
def numba_correspondance(ar: np.ndarray) -> np.ndarray:
    """
    Correlation matrix for numba.

    Args:
        ar: write your description
        np: write your description
        ndarray: write your description
    """
    return 1 - ar


@njit
def numba_invert_proba(ar: np.ndarray, p_invert: float) -> np.ndarray:
    """
    Inverts the probabilities of ar.

    Args:
        ar: write your description
        np: write your description
        ndarray: write your description
        p_invert: write your description
    """
    if numba_rand() < p_invert:
        return numba_correspondance(ar)
    return ar


def get_rect(x, y, width, height, angle):
    """
    Get a rectangle from the x y width height and angle.

    Args:
        x: write your description
        y: write your description
        width: write your description
        height: write your description
        angle: write your description
    """
    rect = numba_straight_rect(width, height)
    theta = (np.pi / 180.0) * angle
    R = numba_rotation_matrix(theta)
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    # transformed_rect = numba_transform_rect(rect.astype(float), R.astype(float), offset.astype(float))
    return transformed_rect

# def get_rect(x, y, width, height, angle):
#     rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
#     theta = (np.pi / 180.0) * angle
#     R = np.array([[np.cos(theta), -np.sin(theta)],
#                   [np.sin(theta), np.cos(theta)]])
#     offset = np.array([x, y])
#     transformed_rect = np.dot(rect, R) + offset
#     return transformed_rect


def draw_poly(draw, poly, fill_value=1):
    """
    Draws a polygon.

    Args:
        draw: write your description
        poly: write your description
        fill_value: write your description
    """
    draw.polygon([tuple(p) for p in poly], fill=fill_value)


def draw_ellipse(draw, center, radius, fill_value=1):
    """
    Draws an ellipse with the given center and radius.

    Args:
        draw: write your description
        center: write your description
        radius: write your description
        fill_value: write your description
    """
    bbox = (center[0] - radius[0], center[1] - radius[1], center[0] + radius[0], center[1] + radius[1])
    draw.ellipse(bbox, fill=fill_value)


def get_random_rotated_diskorect(
    size: Tuple, n_shapes: int = 30, max_shape: Tuple[int] = (15, 15), p_invert: float = 0.5,
        border=(4, 4), n_holes: int = 15, max_shape_holes: Tuple[int] = (5, 5), noise_proba=0.05, **kwargs
):
    """
    Generate a diskorect with randomly rotated diskorect.

    Args:
        size: write your description
        n_shapes: write your description
        max_shape: write your description
        p_invert: write your description
        border: write your description
        n_holes: write your description
        max_shape_holes: write your description
        noise_proba: write your description
    """
    diskorect = np.zeros(size)
    img = Image.fromarray(diskorect)
    draw = ImageDraw.Draw(img)

    def draw_shape(max_shape, fill_value):
        """
        Draw a shape.

        Args:
            max_shape: write your description
            fill_value: write your description
        """
        x = numba_randint(0, size[0] - 2)
        y = numba_randint(0, size[0] - 2)

        if np.random.rand() < .5:
            W = numba_randint(1, max_shape[0])
            L = numba_randint(1, max_shape[1])

            angle = numba_rand() * 45
            draw_poly(draw, get_rect(x, y, W, L, angle), fill_value=fill_value)

        else:
            rx = numba_randint(1, max_shape[0]//2)
            ry = numba_randint(1, max_shape[1]//2)
            draw_ellipse(draw, np.array([x, y]), (rx, ry), fill_value=fill_value)

    for _ in range(n_shapes):
        draw_shape(max_shape=max_shape, fill_value=1)

    for _ in range(n_holes):
        draw_shape(max_shape=max_shape_holes, fill_value=0)

    diskorect = np.asarray(img) + 0
    diskorect[numba_rand_shape_2d(*diskorect.shape) < noise_proba] = 1
    diskorect = numba_invert_proba(diskorect, p_invert)

    diskorect[:border[0], :] = 0
    diskorect[-border[0]:, :] = 0
    diskorect[:, :border[0]] = 0
    diskorect[:, -border[0]:] = 0

    return diskorect


def get_random_diskorect_channels(size: Tuple, squeeze: bool = False, *args, **kwargs):
    """Applies diskorect to multiple channels.

    Args:
        size (Tuple): (W, L, H)
        squeeze (bool, optional): If True, squeeze the output: if H = 1, returns size (W, L). Defaults to False.

    Raises:
        ValueError: size must be of len 2 or 3, either (W, L) or (W, L, H) with H number of channels.

    Returns:
        np.ndarray: size (W, L) or (W, L, H)
    """
    if len(size) == 3:
        W, L, H = size
    elif len(size) == 2:
        W, L = size
        H = 1
    else:
        raise ValueError(f"size argument must have 3 or 2 values, not f{len(size)}.")

    final_img = np.zeros((W, L, H))
    for chan in range(H):
        final_img[..., chan] = get_random_rotated_diskorect((W, L), *args, **kwargs)

    if squeeze:
        return np.squeeze(final_img)
    return final_img
