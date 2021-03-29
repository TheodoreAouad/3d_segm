import numpy as np
import skimage.morphology as morp

import src.utils


def get_markers_nd(segl, label_slices, selem, ):
    """
    Return the n - segmenters that are in segments

    Args:
        segl: (todo): write your description
        label_slices: (str): write your description
        selem: (todo): write your description
    """
    markers = np.zeros_like(segl).astype(int)
    markers[:, label_slices, ...] = -1
    # dil_segl = morp.dilation(segl, selem).astype(int) - segl.astype(int)
    dil_segl = segl
    markers[dil_segl == 1] = 0
    markers[segl == 1] = 1
    return markers


def get_markers_2d(segl, label_slices, margin=.05):
    """
    Return a 2d between segments

    Args:
        segl: (todo): write your description
        label_slices: (str): write your description
        margin: (str): write your description
    """
    return get_markers_nd(segl, label_slices, selem=morp.disk(
        max(margin * segl.shape[0], 1)))


def get_markers_3d(segl, label_slices, margin=.05):
    """
    Returns a 2d between segments

    Args:
        segl: (todo): write your description
        label_slices: (str): write your description
        margin: (str): write your description
    """
    return get_markers_nd(segl, label_slices, selem=morp.ball(
        max(margin * segl.shape[0], 1)))


def apply_watershed(img, segl, label_slices, margin=.02):
    """
    Apply an nd to an image.

    Args:
        img: (array): write your description
        segl: (todo): write your description
        label_slices: (todo): write your description
        margin: (todo): write your description
    """
    markers = get_markers_nd(segl, label_slices, margin)
    grad_img = np.abs(src.utils.grad_img(img))
    return morp.watershed(grad_img, markers)