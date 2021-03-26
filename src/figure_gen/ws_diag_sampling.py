
# %%

from importlib import reload
from time import time
import itertools

import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as morp
import skimage.segmentation as sksegm
import pandas as pd
from tqdm import tqdm

import all_paths as ap

reload(ap)
import src.utils as u
import src.data_manager.utils as du
import src.metrics as m
import src.geometry.utils as gu
import src.diag_sampling as ds


print('Done.')



# %%

seg3d = du.get_segm_atlas()
vol = du.get_vol_atlas()
vol_nib = du.get_nib_atlas()

crop_xs = (100, 300)
crop_ys = (150, 270)
crop_zs = (0, seg3d.shape[-1])

cvol = u.apply_crop(vol, crop_xs, crop_ys, crop_zs)
cseg3d = u.apply_crop(seg3d, crop_xs, crop_ys, crop_zs)

# %%

SACRUM = 1
ILIAC = 2
nb_to_bone = ['', 'SACRUM', 'ILIAC']
BONE = SACRUM

margin = .1 * cvol.shape[0]
all_metrics = {
    'l2_norm': lambda x, y: np.sqrt(((x.astype(float) - y.astype(float)) ** 2).mean()),
    'dice': lambda x, y: m.dice(x, y),
}
# %%
nthetas = np.arange(1, 60)
# nphi = 3

all_dfs = []

for BONE in [SACRUM, ILIAC]:
    cseg = cseg3d == BONE

    for ntheta in tqdm(nthetas):
        thetas = u.uniform_sampling(np.linspace(0, 2 * np.pi, 2*ntheta), ntheta, dtype=float)
        # phis = u.uniform_sampling(np.linspace(0, np.pi, nphi), nphi, dtype=float)
        phis = [np.pi/2]

        vecns = np.array([gu.spheric_coord(50, theta, phi) for (theta, phi) in itertools.product(thetas, phis)]).T
        n_slices = vecns.shape[1]

        t1 = time()
        cov, markers, annots = ds.diag_sampling_annot(cseg, vecns)
        sampling_time = time() - t1

        grad_vol = u.grad_morp(cvol, morp.ball(1))

        t1 = time()
        labels_pred = sksegm.watershed(grad_vol, markers)
        ws_time = time() - t1

        dice_score = m.dice(cseg, labels_pred == 1)

        all_dfs.append(pd.DataFrame(dict(
            {
                'n_slice': [n_slices],
                'vecns': [vecns],
                'ws_duration': [ws_time],
                'sampling_duration': [sampling_time],
                'BONE': [nb_to_bone[BONE]],
            },
            **{fn_name: fn(cseg == 1, labels_pred == 1) for (fn_name, fn) in all_metrics.items()}
        )))

all_dfs = pd.concat(all_dfs)

#%%

for BONE in ['SACRUM', 'ILIAC']:
    plt.plot(all_dfs.loc[all_dfs['BONE'] == BONE, 'n_slice'], all_dfs.loc[all_dfs['BONE'] == BONE, 'dice'], label=BONE)

plt.legend()
plt.title('Dice score vs number of slices for each bone')
plt.xlabel('n_slices')
plt.ylabel('dice')
plt.grid(b=True, which='major')
# plt.savefig('/home/safran/theodore/These/manuscrit/latex/watershed_tests/ws-summary.png')
plt.show()

