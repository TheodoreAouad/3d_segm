
# %%

import random
from importlib import reload
from time import time
from os.path import join
import pathlib
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import skimage.morphology as morp
import skimage.segmentation as sksegm
from tqdm import tqdm
import vpython as vp

import all_paths as ap

reload(ap)
import src.utils as u
import src.data_manager.utils as du
import src.data_manager.get_data as gd
import src.plotter as p
import src.animer as an
import src.metrics as m
import src.watershed as ws


def reload_modules():
    for mudl in [p, an, u, du, gd, m, ws]:
        reload(mudl)


reload_modules()

print('Done.')



# %%

seg3d = du.get_segm_atlas()
vol = du.get_vol_atlas()
vol_nib = du.get_nib_atlas()

print(seg3d.shape, vol.shape)
# %%
u.save_as_nii('segmentations/true_segm.nii.gz', seg3d, vol_nib.affine)
# %%

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

p.plot_img_mask_on_ax(ax, vol[..., 150], seg3d[..., 150])
fig.show()



# %%

reload_modules()

slice_idx = 150
img = vol[..., slice_idx]
# seg = morp.label(seg3d[..., slice_idx] == 2) == 1
seg = seg3d[..., slice_idx] == 2

img = u.center_and_crop(img, seg, (300, 200))
seg = u.center_and_crop(seg, seg, (300, 200))

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_title('True Segm')
p.plot_img_mask_on_ax(ax, img, seg)
fig.show()
# %%

n_slices = 7
margin = .1 * img.shape[0]

all_slices = np.where(seg.sum(0))[0]
# label_slices = random.sample(list(all_slices), n_slices)
label_slices = np.linspace(all_slices.min() + 1, all_slices.max() - 1, n_slices).astype(int)
segl = np.zeros_like(seg)
segl[:, label_slices] = seg[:, label_slices]
# segl[:, np.array(label_slices) + 1] = seg[:, np.array(label_slices) + 1]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
p.plot_img_mask_on_ax(ax, img, segl)
fig.show()
# %%

markers = ws.get_markers_2d(segl, label_slices, 0.01)
fig, ax = plt.subplots(1, 1, figsize=(5, 10))
p.plot_img_mask_on_ax(ax, img[:, all_slices], markers[:, all_slices])
fig.show()
# %%

reload_modules()
# grad_img = np.abs(du.grad_img(img))
# labels = morp.watershed(grad_img, markers)
labels = ws.apply_watershed(img, segl, label_slices)

# %%

fig, axs = plt.subplots(1, 3, figsize=(15, 10))
p.plot_img_mask_on_ax(axs[0], img[:, all_slices], markers[:, all_slices])
p.plot_img_mask_on_ax(axs[1], img[:, all_slices], seg[:, all_slices])
p.plot_img_mask_on_ax(axs[2], img, labels == 1)

# %%
### 3d

# %%

crop_xs = (100, 300)
crop_ys = (150, 270)
crop_zs = (0, seg3d.shape[-1])

# cvol = vol[crop_xs[0]:crop_xs[1], crop_ys[0]:crop_ys[1]]
# cseg3d = seg3d[crop_xs[0]:crop_xs[1], crop_ys[0]:crop_ys[1]]
cvol = u.apply_crop(vol, crop_xs, crop_ys, crop_zs)
cseg3d = u.apply_crop(seg3d, crop_xs, crop_ys, crop_zs)

# %%

SACRUM = 1
ILIAC = 2
nb_to_bone = ['', 'SACRUM', 'ILIAC']
BONE = SACRUM

n_slices = 3
margin = .1 * cvol.shape[0]
cseg = cseg3d == BONE

all_slices = np.where(cseg.sum(0).sum(1))[0]

# label_slices = np.linspace(all_slices.min() + 1, all_slices.max() - 1, n_slices).astype(int)
# label_slices = np.linspace(all_slices.min() - 1, all_slices.max() + 1, len(all_slices) + 2).astype(int)
label_slices = u.uniform_sampling(all_slices, n_slices)

csegl = np.zeros_like(cseg)
csegl[:, label_slices, :] = cseg[:, label_slices, :]

markers = ws.get_markers_3d(csegl, label_slices, margin=0)
grad_vol = u.grad_morp(cvol, morp.ball(1))

t1 = time()
labels_pred = morp.watershed(grad_vol, markers)
print(time() - t1)

m.dice(cseg, labels_pred == 1)

# %%
u.save_as_nii(
    f'segmentations/straight/{n_slices}/annotated_slices.nii.gz',
    u.reverse_crop(csegl.astype(int), seg3d.shape, crop_xs, crop_ys, crop_zs),
    vol_nib.affine,
    type=np.uint8
)

u.save_as_nii(
    f'segmentations/straight/{n_slices}/markers.nii.gz',
    u.reverse_crop(np.where(markers==-1, 2, markers), seg3d.shape, crop_xs, crop_ys, crop_zs),
    vol_nib.affine,
    type=np.uint8,
)

u.save_as_nii(
    f'segmentations/straight/{n_slices}/final_pred.nii.gz',
    u.reverse_crop((labels_pred == 1).astype(int), seg3d.shape, crop_xs, crop_ys, crop_zs),
    vol_nib.affine,
    type=np.uint8
)

# %%

slice_idx = 150
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
p.plot_img_mask_on_ax(ax, cvol[..., slice_idx], csegl[..., slice_idx])
fig.show()



# %%
reload_modules()
slice_z = 150
slice_y = 104

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
p.plot_img_mask_on_ax(axs[0], cvol[..., slice_z], markers[..., slice_z])
# p.plot_img_mask_on_ax(axs[1], cvol[:, slice_y, :], cseg[:, slice_y, :])
p.plot_img_mask_on_ax(axs[1], cvol[..., slice_z], cseg[..., slice_z])
fig.show()

# %%
slice_x, slice_y, slice_z = 147, 66, 150

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
p.plot_img_mask_on_ax(axs[0], cvol[..., slice_z], labels_pred[..., slice_z] == 1)
p.plot_img_mask_on_ax(axs[1], cvol[:, slice_y, :], labels_pred[:, slice_y, :] == 1)
p.plot_img_mask_on_ax(axs[2], cvol[slice_x, ...], labels_pred[slice_x, ...] == 1)
fig.show()
