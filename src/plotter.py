import os

import matplotlib.pyplot as plt
import numpy as np
import pathlib
from PIL import Image, ImageEnhance
from matplotlib import cm


def plot_img_mask_on_ax(ax, img, mask, alpha=.7):

    masked = np.ma.masked_where(mask == 0, mask)
    ax.imshow(img, cmap='gray')
    ax.imshow(masked, cmap='jet', alpha=alpha)

    return ax


def save_figs(figs, path, filename=''):

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    all_imgs = [int(img.split('--')[-1][:-4]) for img in os.listdir(path) if ('.png' in img) and (filename in img)]
    first = max(all_imgs) + 1 if len(all_imgs) != 0 else 0
    for i, fig in enumerate(figs):
        fig.savefig(os.path.join(path, '{}--{}.png'.format(filename.split('--')[0], i + first)))
        print('saved in ', os.path.join(path, '{}--{}.png'.format(filename.split('--')[0], i + first)))


def plot_three(img, target, pred, titles=['Input', 'True', 'Pred'], **kwargs):
    """
    Plot the image, the true mask and predicted mask

    Args:
        img (ndarray): shape (width, length)
        target (ndarray): shape (width, length), true mask in [0, 1, 2]
        pred (ndarray): shape (width, length), predicted mask in [0, 1, 2]
    """

    fig, axs = plt.subplots(1, 3, **kwargs)
    axs[0].imshow(img, cmap='gray')
    plot_img_mask_on_ax(axs[1], img, target)
    plot_img_mask_on_ax(axs[2], img, pred)

    for i in range(3):
        axs[i].set_title(titles[i])

    return fig


def plot_rectangle_on_ax(ax, P, **kwargs):
    """
    Plot a rectangle on ax.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): ax to plot rectangle on
        P (ndarray): shape (3, 4). The corners of the rectangle, in "circle" order.
    """
    for i in range(4):
        ax.plot(
            [P[0, i], P[0, (i+1) % 4]],
            [P[1, i], P[1, (i+1) % 4]],
            [P[2, i], P[2, (i+1) % 4]], **kwargs)



def create_pillow_array_mask(rows, masks=[], pixels_sep=20):

    dim0_rows = [max([ar.shape[0] for ar in row if ar is not None]) for row in rows]
    dim0 = sum(dim0_rows)
    dim1 = max([sum([ar.shape[1] for ar in row if ar is not None]) for row in rows])



    sum([max([ar.shape[0] for ar in row if ar is not None]) for row in rows])

    nrows = len(rows)
    ncols = max([len(row) for row in rows])

    big_ar = np.ones((dim0 + pixels_sep*(nrows-1), dim1 + pixels_sep * (ncols - 1)))
    big_mask = np.zeros((dim0 + pixels_sep*(nrows-1), dim1 + pixels_sep * (ncols - 1))) - 1

    prev_i = 0
    for i in range(nrows):
        prev_j = 0
        for j in range(ncols):
            if j < len(rows[i]):

                if rows[i][j] is None:
                    continue

                ar = rows[i][j]
                if i < len(masks) and j < len(masks[i]) and masks[i][j] is not None:
                    big_mask[prev_i:prev_i + ar.shape[0], prev_j:prev_j + ar.shape[1]] = masks[i][j]
                else:
                    big_mask[prev_i:prev_i + ar.shape[0], prev_j:prev_j + ar.shape[1]] = 0

                if ar.max() != ar.min():
                    ar = (ar - ar.min()) / (ar.max() - ar.min())

                big_ar[prev_i:prev_i + ar.shape[0], prev_j:prev_j + ar.shape[1]] = ar
                prev_j = prev_j + ar.shape[1] + pixels_sep
        prev_i = prev_i + dim0_rows[i] + pixels_sep

    return big_ar, big_mask


def create_pillow_image(rows, masks=[], cmap_img=cm.gray, cmap_mask=cm.jet, alpha=.5, enhance=None, **kwargs):
    big_ar, big_mask = create_pillow_array_mask(rows, masks, **kwargs)

    pil_ar = array_to_pil(big_ar, mask=big_mask==-1, cmap=cmap_img)
    pil_mask = array_to_pil(big_mask, mask=np.isin(big_mask, [-1, 0]), cmap=cmap_mask)

    pil_mask.putalpha(int(alpha * 255))

    pil_to_save = Image.alpha_composite(pil_ar, pil_mask)

    if enhance:
        enhancer = ImageEnhance.Brightness(pil_to_save)
        pil_to_save = enhancer.enhance(enhance)

    return pil_to_save


def array_to_pil(ar, mask=None, cmap=cm.gray,):
    ar = (ar - ar.min()) / (ar.max() - ar.min())
    if mask is not None:
        ar = np.ma.masked_where(mask, ar)

    return Image.fromarray(np.uint8(cmap(ar)*255))
