import pathlib
from typing import Tuple, Optional

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.morphology import disk, dilation, erosion


def convert_to_nii(ar: np.ndarray, affine: np.ndarray):
    return nib.Nifti1Image(ar, affine)


def save_as_nii(path: str, ar: np.ndarray, affine: np.ndarray, dtype: Optional[type] = None):
    if dtype is not None:
        ar = ar.astype(dtype)
    nib_ar = convert_to_nii(ar, affine)
    pathlib.Path(path).parent.mkdir(exist_ok=True, parents=True)
    nib.save(nib_ar, path)
    return path


def apply_crop(ar: np.ndarray, crop_xs: Tuple, crop_ys: Tuple, crop_zs: Optional[Tuple] = None):
    if len(ar.shape) == 2:
        return ar[crop_xs[0]:crop_xs[1], crop_ys[0]:crop_ys[1]]

    if len(ar.shape) == 3:
        return ar[crop_xs[0]:crop_xs[1], crop_ys[0]:crop_ys[1], crop_zs[0]:crop_zs[1]]


def reverse_crop(croped_ar: np.ndarray, size: Tuple, crop_xs: Tuple, crop_ys: Tuple, crop_zs: Optional[Tuple] = None,
                 fill_value: float = 0):
    ar = np.zeros(size) + fill_value
    if len(size) == 2:
        ar[crop_xs[0]:crop_xs[1], crop_ys[0]:crop_ys[1]] = croped_ar

    if len(size) == 3:
        ar[crop_xs[0]:crop_xs[1], crop_ys[0]:crop_ys[1], crop_zs[0]:crop_zs[1]] = croped_ar

    return ar.astype(croped_ar.dtype)


def grad_morp(img: np.ndarray, selem: np.ndarray = disk(1)):
    """
    Gives the morphological gradient of an image.

    Args:
        img (ndarray): image to give gradient of. Shape depends on dimension.
        selem (ndarray, optional): Local region. See dilation or erosion. Defaults to disk(1).

    Returns:
        ndarray: same shape as img.
    """
    return dilation(img, selem=selem) - erosion(img, selem=selem)


def grad_img(img: np.ndarray, mode: str = 'constant'):
    """
    Gives the sobel gradient of an image.

    Args:
        img (ndarray): shape (W, L)
        mode (str, optional): See ndimage.sobel. Defaults to 'constant'.

    Returns:
        ndarray: shape (W, L).
    """
    sx = ndimage.sobel(img, axis=0, mode=mode)
    sy = ndimage.sobel(img, axis=1, mode=mode)

    return np.hypot(sx, sy)


def center_and_crop(img: np.ndarray, mask: np.ndarray, size: Tuple, fill_background: int = 0):
    """
    Center and crop an img around the mask.

    Args:
        img (ndarray): shape (W, L)
        mask (ndarray): shape (W, L). Array of 0, 1 and 2.
        size (tuple): Size of the region.
        fill_background (int): value to fill the background with.

    Returns:
        ndarray: the img cropped around the mask of size size.
    """
    assert set(np.unique(mask)).issubset({0, 1, 2}), 'mask must have for values only 0, 1, 2. Values: {}'.format(np.unique(mask))
    x, y = np.where(mask != 0)

    if type(size) == int:
        size = (size, size)
    if len(x) == 0:
        x, y = (img.shape[0] - 1)/2, (img.shape[1] - 1)/2
    else:
        x = x.mean()
        y = y.mean()

    sx = (size[0] - 1) / 2
    sy = (size[1] - 1) / 2

    res = np.zeros(size) + fill_background

    res_x0 = max(ceil_(sx) - ceil_(x), 0)
    res_x1 = ceil_(sx) + min(floor_(sx) + 1, floor_(img.shape[0] - x))
    res_y0 = max(ceil_(sy) - ceil_(y), 0)
    res_y1 = ceil_(sy) + min(floor_(sy) + 1, floor_(img.shape[1] - y))


    res[res_x0:res_x1, res_y0:res_y1] = img[
        max(ceil_(x) - ceil_(sx), 0): ceil_(x) + floor_(sx) + 1,
        max(ceil_(y) - ceil_(sy), 0): ceil_(y) + floor_(sy) + 1,
    ]

    return res


def floor_(x: float):
    return np.int(np.floor(x))


def ceil_(x: float):
    return np.int(np.ceil(x))


def uniform_sampling(all_slices: np.ndarray, n_slices: int, dtype: type = int):
    if n_slices > 2:
        return np.linspace(all_slices.min(), all_slices.max(), n_slices).astype(dtype)

    if n_slices == 1:
        return [dtype(all_slices.min() + all_slices.max())]

    return np.array([
        (all_slices.min()*.67 + all_slices.max()*.33),
        (all_slices.min()*.33 + all_slices.max()*.67),
    ]).astype(dtype)



def get_arrangements(n):
    """
    Get arrangements of n elements.
    Examples:
        if n = 3, returns
        [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    """
    def f(A, S):
        cur_set = set(A).difference(S)
        if len(cur_set) == 0:
            return [S]
        res = []
        for i in cur_set:
            res += f(A, S + [i])
        return res

    return f(range(n), [])
