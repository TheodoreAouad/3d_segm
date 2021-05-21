# %%
import os
from time import time

import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as morp
import nibabel as nib

import general.utils as u

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
