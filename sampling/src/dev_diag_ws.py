
# %%

from importlib import reload
from time import time
import itertools

import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as morp
from mpl_toolkits.mplot3d import Axes3D

import all_paths as ap

reload(ap)
import sampling.src.utils as u
import sampling.src.data_manager.utils as du
import sampling.src.data_manager.get_data as gd
import sampling.src.plotter as p
import sampling.src.animer as an
import sampling.src.metrics as m
import sampling.src.geometry.entities as e
import sampling.src.geometry.utils as gu
import sampling.src.watershed as ws
import sampling.src.diag_sampling as ds


def reload_modules():
    for mudl in [p, an, u, du, gd, m, ws, e, gu, ds]:
        reload(mudl)


reload_modules()

print('Done.')



# %%

seg3d = du.get_segm_atlas()
vol = du.get_vol_atlas()
vol_nib = du.get_nib_atlas()

print(seg3d.shape, vol.shape)
# %%
# u.save_as_nii('segmentations/true_segm.nii.gz', seg3d, vol_nib.affine)
# %%

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

p.plot_img_mask_on_ax(ax, vol[..., 150], seg3d[..., 150])
fig.show()

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

margin = .1 * cvol.shape[0]
cseg = cseg3d == BONE

# %%
ntheta = 4
# nphi = 3

thetas = u.uniform_sampling(np.linspace(0, 2 * np.pi, 2*ntheta), ntheta, dtype=float)
# phis = u.uniform_sampling(np.linspace(0, np.pi, nphi), nphi, dtype=float)
phis = [np.pi/2]

vecns = np.array([gu.spheric_coord(50, theta, phi) for (theta, phi) in itertools.product(thetas, phis)]).T
n_slices = vecns.shape[1]


# %%


t1 = time()
cov, markers, annots = ds.diag_sampling_annot(cseg, vecns)
print("Sampling time:", time() - t1)

grad_vol = u.grad_morp(cvol, morp.ball(1))

t1 = time()
labels_pred = morp.watershed(grad_vol, markers)
print("Watershed time:", time() - t1)

dice_score = m.dice(cseg, labels_pred == 1)
print(dice_score)

# %%
u.save_as_nii(
    f'segmentations/diag/{n_slices}/coverage.nii.gz',
    u.reverse_crop(cov, seg3d.shape, crop_xs, crop_ys, crop_zs),
    vol_nib.affine,
    type=np.uint8
)

u.save_as_nii(
    f'segmentations/diag/{n_slices}/annotations.nii.gz',
    u.reverse_crop(annots, seg3d.shape, crop_xs, crop_ys, crop_zs),
    vol_nib.affine,
    type=np.uint8
)

u.save_as_nii(
    f'segmentations/diag/{n_slices}/markers.nii.gz',
    u.reverse_crop(np.where(markers==-1, 2, markers), seg3d.shape, crop_xs, crop_ys, crop_zs),
    vol_nib.affine,
    type=np.uint8
)

u.save_as_nii(
    f'segmentations/diag/{n_slices}/final_pred.nii.gz',
    u.reverse_crop(labels_pred==1, seg3d.shape, crop_xs, crop_ys, crop_zs),
    vol_nib.affine,
    type=np.uint8
)

#%%
fig = plt.figure()
ax = Axes3D(fig)

cube = e.StraightCube(size=cseg.shape)

# cube.plot_on_ax(ax)
cube.plot_centered_plane(ax, e.Plane(vecns[:, 2]))
cube.plot_centered_plane(ax, e.Plane(vecns[:, 3]))
# ax.scatter(samples[0,:], samples[1,:], samples[2,:])
# ax.scatter(*vecns)
fig.show()
