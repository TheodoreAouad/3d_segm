import numpy as np
from scipy import ndimage


def ceil_(x):
    return np.int(np.ceil(x))


def floor_(x):
    return np.int(np.floor(x))


def get_order_from_name(filename: str):
    """
    Extracts the order of one plane of MRI

     against the others.
    Args:
        filename (str): name of the file.
    Returns:
        int: order of the file. Note: the order of 'VERSION' is unknwon.
    """
    letters_to_replace = ['I', 'E', 'B', 'A', 'C', 'D']
    if filename[-1] == '1':
        return 10
    # elif filename == 'VERSION':
    #     return -1
    else:
        res = filename.replace('0', '')
        for letter in letters_to_replace:
            res = res.replace(letter, '')
        try:
            return int(res)
        except ValueError:
            return -1


def get_volume_from_mri(mri, axis_to_stack=0):
    """
    Stacks all the pixel_array of the lines of mri to form a 3D cube.

    Args:
        mri (pd.DataFrame): dataframe containing the pixel arrays
        axis_to_stack (int, optional): Axis to stack to. Defaults to 0.

    Returns:
        ndarray: 3D cube of the slices put together.
    """
    volume = []
    for img in mri.pixel_array:
        volume.append(img)
    return np.stack(volume, axis_to_stack)


def center_and_crop(img, mask, size, fill_background=0):
    """
    Center and crop an img around the mask.

    Args:
        img (ndarray): shape (W, L)
        mask (ndarray): shape (W, L). Array of 0, 1 and 2.
        size (tuple): Size of the region.

    Returns:
        ndarray: the img cropped around the mask of size size.
    """
    assert set(np.unique(mask)).issubset(set([0, 1, 2])), 'mask must have for values only 0, 1, 2. Values: {}'.format(np.unique(mask))
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



def grad_img(img, mode='constant'):
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
