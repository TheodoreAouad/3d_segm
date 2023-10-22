from numba import njit

from skimage.morphology import binary_erosion, binary_dilation


@njit
def numba_binary_erosion(image, selem):
    return binary_erosion(image, selem)


@njit
def numba_binary_dilation(image, selem):
    return binary_dilation(image, selem)
