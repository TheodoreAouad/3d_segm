import pathlib
from typing import Tuple, Optional, List
from time import time

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.morphology import disk, dilation, erosion, label
from skimage.transform import warp



def convert_to_nii(ar: np.ndarray, affine: np.ndarray):
    """
    Convert the input array and affine to a NIfti1Image object.

    Args:
        ar: write your description
        np: write your description
        ndarray: write your description
        affine: write your description
        np: write your description
        ndarray: write your description
    """
    return nib.Nifti1Image(ar, affine)


def save_as_nii(path: str, ar: np.ndarray, affine: np.ndarray, dtype: Optional[type] = None):
    """
    Saves the given AR and affine to a NNI file.

    Args:
        path: write your description
        ar: write your description
        np: write your description
        ndarray: write your description
        affine: write your description
        np: write your description
        ndarray: write your description
        dtype: write your description
    """
    if dtype is not None:
        ar = ar.astype(dtype)
    nib_ar = convert_to_nii(ar, affine)
    pathlib.Path(path).parent.mkdir(exist_ok=True, parents=True)
    nib.save(nib_ar, path)
    return path


def apply_crop(ar: np.ndarray, crop_xs: Tuple, crop_ys: Tuple, crop_zs: Optional[Tuple] = None):
    """
    Apply crop to an array array.

    Args:
        ar: write your description
        np: write your description
        ndarray: write your description
        crop_xs: write your description
        crop_ys: write your description
        crop_zs: write your description
    """
    if len(ar.shape) == 2:
        return ar[crop_xs[0]:crop_xs[1], crop_ys[0]:crop_ys[1]]

    if len(ar.shape) == 3:
        return ar[crop_xs[0]:crop_xs[1], crop_ys[0]:crop_ys[1], crop_zs[0]:crop_zs[1]]


def reverse_crop(croped_ar: np.ndarray, size: Tuple, crop_xs: Tuple, crop_ys: Tuple, crop_zs: Optional[Tuple] = None,
                 fill_value: float = 0):
    """
    Reverse crop of an array.

    Args:
        croped_ar: write your description
        np: write your description
        ndarray: write your description
        size: write your description
        crop_xs: write your description
        crop_ys: write your description
        crop_zs: write your description
        fill_value: write your description
    """
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
    """
    Round a float to the nearest integer.

    Args:
        x: write your description
    """
    return np.int(np.floor(x))


def ceil_(x: float):
    """
    Round a float to an integer.

    Args:
        x: write your description
    """
    return np.int(np.ceil(x))


def uniform_sampling(all_slices: np.ndarray, n_slices: int, dtype: type = int):
    """
    Sample uniformly from a list of slices.

    Args:
        all_slices: write your description
        np: write your description
        ndarray: write your description
        n_slices: write your description
        dtype: write your description
    """
    if n_slices > 2:
        return np.linspace(all_slices.min(), all_slices.max(), n_slices).astype(dtype)

    if n_slices == 1:
        return [dtype(all_slices.min() + all_slices.max())]

    return np.array([
        (all_slices.min()*.67 + all_slices.max()*.33),
        (all_slices.min()*.33 + all_slices.max()*.67),
    ]).astype(dtype)


def get_arrangements(n: int) -> List[List]:
    """
    Get arrangements of n elements.
    Examples:
        if n = 3, returns
        [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    """
    def f(A, S):
        """
        Recursively find all elements in S that are not in A.

        Args:
            A: write your description
            S: write your description
        """
        cur_set = set(A).difference(S)
        if len(cur_set) == 0:
            return [S]
        res = []
        for i in cur_set:
            res += f(A, S + [i])
        return res

    return f(range(n), [])


def get_most_important_labels(labels: np.ndarray, weights: np.ndarray, scope: int = 1, return_weights: bool = False) -> np.ndarray:
    """Returns the most weighted elements in labels.

    Args:
        labels (ndarray): array of elements
        weights (ndarray): array of weights for each elements
        scope (int, optional): number of labels to return. Defaults to 1.
        return_weights (bool, optional): If True, returns weights. Defaults to False.

    Returns:
        ndarray: shape (scope,). Labels with biggest weight.
    """
    labels_sorted = labels[weights.argsort()][::-1]
    if return_weights:
        weights_sorted = weights[weights.argsort()][::-1]
        return labels_sorted[:scope], weights_sorted[:scope]
    return labels_sorted[:scope]


def get_most_important_regions(regions: np.ndarray, weights: np.ndarray = 1, scope: int = 1, background: float = 0) -> np.ndarray:
    """Returns a mask containing only the biggest regions. The mask is labelled.
    The number of regions is the scope.

    Args:
        regions (ndarray): Mask of regions. Either 0 and 1, or already labeled.
        weights (ndarray, optional): weights to give to each label. Defaults to 1.
        scope (int, optional): number of regions. Defaults to 1.
        background (int, optional): pixels that are not part of the most important
                                    regions. Defaults to 0.

    Returns:
        ndarray: same shape as regions. Array like regions but with the less
                important regions being put to background.
    """
    if len(np.unique(regions)) == 2:
        regions = label(regions)
    labels, count = np.unique(regions, return_counts=True)
    count[labels == background] = 0
    weighted = count * weights
    biggest_labels = get_most_important_labels(labels, weighted, scope=scope)
    regions[~np.isin(regions, biggest_labels)] = background
    return regions


def apply_transform(
    ar,
    T,
    center='mid',
    order=1,
    do_scipy_func=True,
    show_time=False,
    **kwargs
):
    """
    Apply a transformation array T to each element of array ar. Returns
    ar[T(i)] for indexes i.

    Args:
        ar (ndarray): array to transform
        T (ndarray): square array of shape len(ar.shape) + 1
        order (int, optional): Interpolation order of B-spline. See skimage.transform.warp for more details.
                               Defaults to 1 for linear interpolation.
        show_time (bool, optional): Show computation time. Defaults to False.

    Returns:
        ndarray: same shape as ar. ar[T(i)] for all indexes i, interpolated.
    """
    start = time()
    if center == "mid":
        center = np.array([(shp -1)/2 for shp in ar.shape])

    if do_scipy_func:
        rot_matrix = T[:-1, :-1]
        trans_matrix = T[:-1, -1]
        if type(center) in [int, float]:
            center = np.zeros_like(trans_matrix) + center
        offset = trans_matrix - rot_matrix @ center + center
        res = ndimage.affine_transform(ar, rot_matrix, offset=offset, order=order, **kwargs)
        if show_time:
            print('Applying scipy transform:', time() - start)

    else:
        if type(center) == np.ndarray:
            center = center[:, np.newaxis]
        coords = np.meshgrid(*[np.arange(n) for n in ar.shape], indexing='ij')
        samples = np.c_[[ar[ar==ar] for ar in coords]] - center
        samples = np.vstack((samples, np.ones(samples.shape[1])))
        new_indexs = (T @ samples)[:-1] + center
        t1 = time()
        if show_time:
            print('Apply transform to indexes:', t1 - start)
        res = warp(ar, new_indexs, order=order, **kwargs).reshape(*ar.shape)
        t2 = time()
        if show_time:
            print('Apply warping:', t2-t1)
    return res
