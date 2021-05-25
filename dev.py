# %%
from importlib import reload
import os
from time import time

import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as morp
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F

import general.utils as u
import general.array_morphology as am

def reload_modules():
    for modl in [u, am]:
        reload(modl)

reload_modules()

# %%

path_segm = os.path.abspath("/home/safran/theodore/datasets/CT-ORG/labels and README/labels-1.nii.gz")
path_img = os.path.abspath("/home/safran/theodore/datasets/CT-ORG/volumes 0-49/volume-1.nii.gz")

seg3n = nib.load(path_segm)
img3n = nib.load(path_img)

seg3 = np.round(seg3n.get_fdata())

# %%
t1 = time()

def horizontal_bar(size):
    selem = np.zeros((3, size, 3))
    selem[1, :, 1] = 1
    return selem

selem = horizontal_bar(50)

oseg3 = morp.opening(seg3 == 5, selem)

print(time() - t1)

oseg3[:selem.shape[0], :selem.shape[1], :selem.shape[2]] = 1
# oseg3[:40, :40, :40] = 1
# %%

u.save_as_nii(
    "segmentations/CT_ORG/labels-1-opened.nii.gz",
    oseg3,
    seg3n.affine,
    dtype=np.uint8
)

# %%

path_segm = os.path.abspath("/hdd/datasets/CT-ORG/raw/labels_and_README/labels-1.nii.gz")
seg3n = nib.load(path_segm)
seg3 = np.round(seg3n.get_fdata()) == 5

selem = morp.ball(3)

t1 = time()
c1 = F.conv3d(
    torch.tensor(seg3).unsqueeze(0).unsqueeze(0).float().cuda(),
    torch.tensor(selem).unsqueeze(0).unsqueeze(0).float().cuda(),
    padding=selem.shape[0]//2
) > 0
c1 = c1.squeeze().cpu().numpy()
time_conv = time() - t1

t2 = time()
c2 = morp.dilation(seg3, selem)
time_morp = time() - t2

print(time_conv, time_morp)
# %%

print(seg3.shape, c1.shape, c2.shape)
print((c2 != c1).sum())

# %%
path_segm = os.path.abspath("/hdd/datasets/CT-ORG/raw/labels_and_README/labels-1.nii.gz")
seg3n = nib.load(path_segm)
seg3 = np.round(seg3n.get_fdata()) == 5

selem = morp.ball(3)

t1 = time()
c1 = F.conv3d(
    torch.tensor(seg3).unsqueeze(0).unsqueeze(0).float().cuda(),
    torch.tensor(selem).unsqueeze(0).unsqueeze(0).float().cuda(),
    padding=selem.shape[0]//2
) == selem.sum()
c1 = c1.squeeze().cpu().numpy()
time_conv = time() - t1

t2 = time()
c2 = morp.erosion(seg3, selem)
time_morp = time() - t2

print(time_conv, time_morp)

# %%

print(seg3.shape, c1.shape, c2.shape)
print((c2 != c1).sum())


# %%
reload_modules()
plt.close()
# path_segm = os.path.abspath("/hdd/datasets/CT-ORG/raw/labels_and_README/labels-1.nii.gz")
# seg3n = nib.load(path_segm)
# seg3 = np.round(seg3n.get_fdata()) == 5
# seg2 = seg3[..., 60]
seg2 = np.zeros((100, 100))
seg2[47:53, 20:80] = 1

# selem = morp.disk(2)
selem = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
])

to_show = seg2 + 0
to_show[:selem.shape[0], :selem.shape[1]] = selem
plt.imshow(to_show)
plt.show()

t1 = time()
# c1 = F.conv2d(
#     torch.tensor(seg2).unsqueeze(0).unsqueeze(0).float(), torch.tensor(selem).unsqueeze(0).unsqueeze(0).float(),
#     padding=1,
# ) > 0
# c1 = c1.squeeze().numpy()
c1 = am.array_dilation(seg2, selem, "2d")
time_conv = time() - t1

t2 = time()
c2 = morp.dilation(seg2, selem)
time_morp = time() - t2

print(time_conv, time_morp)
print((c1 != c2).sum())

# %%
plt.close()
fig, axs = plt.subplots(1, 4)
axs[0].imshow(to_show, interpolation='nearest')
axs[1].imshow(c1, interpolation='nearest')
axs[2].imshow(c2, interpolation='nearest')
axs[3].imshow(np.abs(c2[:c1.shape[0], :c1.shape[1]] != c1), interpolation='nearest')
fig.show()

# %%
x0s, y0s = np.where(seg2 != 0)
x1s, y1s = np.where(c1 != 0)
x2s, y2s = np.where(c2 != 0)

print(seg2.shape, c1.shape, c2.shape)
print(x0s.min(), x1s.min(), x2s.min())
print(x0s.max(), x1s.max(), x2s.max())

print(y0s.min(), y1s.min(), y2s.min())
print(y0s.max(), y1s.max(), y2s.max())


# %%
plt.close()
# path_segm = os.path.abspath("/hdd/datasets/CT-ORG/raw/labels_and_README/labels-1.nii.gz")
# seg3n = nib.load(path_segm)
# seg3 = np.round(seg3n.get_fdata()) == 5
# seg2 = seg3[..., 60]
seg2 = np.zeros((100, 100))
seg2[47:53, 20:80] = 1

# selem = morp.disk(2)
selem = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
])

to_show = seg2 + 0
to_show[:selem.shape[0], :selem.shape[1]] = selem
plt.imshow(to_show)
plt.show()


t1 = time()
c1 = am.array_erosion(seg2, selem, "2d")
time_conv = time() - t1

t2 = time()
c2 = morp.erosion(seg2, selem)
time_morp = time() - t2

print(time_conv, time_morp)
print((c1 != c2).sum())

# %%
plt.close()
fig, axs = plt.subplots(1, 4)
axs[0].imshow(to_show, interpolation='nearest')
axs[1].imshow(c1, interpolation='nearest')
axs[2].imshow(c2, interpolation='nearest')
axs[3].imshow(np.abs(c2[:c1.shape[0], :c1.shape[1]] != c1), interpolation='nearest')
fig.show()

# %%
x0s, y0s = np.where(seg2 != 0)
x1s, y1s = np.where(c1 != 0)
x2s, y2s = np.where(c2 != 0)

print(seg2.shape, c1.shape, c2.shape)
print(x0s.min(), x1s.min(), x2s.min())
print(x0s.max(), x1s.max(), x2s.max())

print(y0s.min(), y1s.min(), y2s.min())
print(y0s.max(), y1s.max(), y2s.max())

