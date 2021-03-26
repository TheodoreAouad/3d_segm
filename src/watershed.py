import numpy as np
import skimage.morphology as morp

import src.utils


def get_markers_nd(segl, label_slices, selem, ):
    markers = np.zeros_like(segl).astype(int)
    markers[:, label_slices, ...] = -1
    # dil_segl = morp.dilation(segl, selem).astype(int) - segl.astype(int)
    dil_segl = segl
    markers[dil_segl == 1] = 0
    markers[segl == 1] = 1
    return markers


def get_markers_2d(segl, label_slices, margin=.05):
    return get_markers_nd(segl, label_slices, selem=morp.disk(
        max(margin * segl.shape[0], 1)))


def get_markers_3d(segl, label_slices, margin=.05):
    return get_markers_nd(segl, label_slices, selem=morp.ball(
        max(margin * segl.shape[0], 1)))


def apply_watershed(img, segl, label_slices, margin=.02):
    markers = get_markers_nd(segl, label_slices, margin)
    grad_img = np.abs(src.utils.grad_img(img))
    return morp.watershed(grad_img, markers)