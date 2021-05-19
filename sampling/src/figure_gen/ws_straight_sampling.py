import random
from importlib import reload
from time import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as morp
import skimage.segmentation as sksegm
from tqdm import tqdm

import all_paths as ap
from sampling.src.utils import uniform_sampling

reload(ap)
import sampling.src.utils as u
import sampling.src.data_manager.utils as du
import sampling.src.data_manager.get_data as gd
import sampling.src.plotter as p
import sampling.src.animer as an
import sampling.src.metrics as m
import sampling.src.watershed as ws


def reload_modules():
    for mudl in [p, an, u, du, gd, m, ws]:
        reload(mudl)


reload_modules()

print('Done.')


# %% Compute time - metric
reload_modules()

seg3d = du.get_segm_atlas()
vol = du.get_vol_atlas()


all_df_metrics = {}
#%%

SACRUM = 1
ILIAC = 2
nb_to_bone = ['', 'SACRUM', 'ILIAC']

cvol = vol[100:300, 150:270]
cseg3d = seg3d[100:300, 150:270]
grad_vol = np.abs(u.grad_morp(cvol, morp.ball(1)))

samplers = {
    'uniform': uniform_sampling,
    'random': lambda all_slices, n_slices: random.sample(list(all_slices), n_slices)
}

all_metrics = {
    'l2_norm': lambda x, y: np.sqrt(((x.astype(float) - y.astype(float)) ** 2).mean()),
    'dice': lambda x, y: m.dice(x, y),
}

for BONE in [SACRUM, ILIAC]:
    for sampling, sampling_fn in samplers.items():
        print(BONE, sampling)
        cseg = cseg3d == BONE
        all_slices = np.where(cseg.sum(0).sum(1))[0]

        n_n_slices = all_slices.max() - all_slices.min() + 1
        # all_n_slices = np.linspace(1, (max_all_slices - min_all_slices - 1), n_n_slices).astype(int)
        all_n_slices = np.arange(1, n_n_slices + 1)
        # all_n_slices = [n_n_slices+2]

        df_metrics = []
        for n_slices in tqdm(all_n_slices):
            label_slices = sampling_fn(all_slices, n_slices)
            # label_slices = np.linspace(all_slices.min()-1, all_slices.max()+1, n_slices).astype(int)

            csegl = np.zeros_like(cseg)
            csegl[:, label_slices, :] = cseg[:, label_slices, :]

            t1 = time()
            markers = ws.get_markers_3d(csegl, label_slices, margin=0)
            labels_pred = sksegm.watershed(grad_vol, markers)
            t2 = time()

            df_metrics.append(pd.DataFrame(dict(
                **{
                    'n_slice': [n_slices],
                    'label_slices': [label_slices],
                    'duration': [t2 - t1],
                    'sampling': [sampling],
                    'BONE': [nb_to_bone[BONE]],
                    # 'labels_pred': [labels_pred],
                },
                **{fn_name: fn(cseg == 1, labels_pred == 1) for (fn_name, fn) in all_metrics.items()}
            )))

        df_metrics = pd.concat(df_metrics)

        all_df_metrics[BONE] = all_df_metrics.get(BONE, []) + [df_metrics]

#%%
plt.plot(all_df_metrics[1][0]['n_slice'], all_df_metrics[1][0]['dice'],
         label=all_df_metrics[1][0]['sampling'].iloc[0] + '-' + all_df_metrics[1][0]['BONE'].iloc[0])
plt.plot(all_df_metrics[2][0]['n_slice'], all_df_metrics[2][0]['dice'],
         label=all_df_metrics[2][0]['sampling'].iloc[0] + '-' + all_df_metrics[2][0]['BONE'].iloc[0])
plt.plot(all_df_metrics[1][1]['n_slice'], all_df_metrics[1][1]['dice'],
         label=all_df_metrics[1][1]['sampling'].iloc[0] + '-' + all_df_metrics[1][1]['BONE'].iloc[0])
plt.plot(all_df_metrics[2][1]['n_slice'], all_df_metrics[2][1]['dice'],
         label=all_df_metrics[2][1]['sampling'].iloc[0] + '-' + all_df_metrics[2][1]['BONE'].iloc[0])
plt.legend()
plt.title('Dice score vs number of slices for each bone')
plt.xlabel('n_slices')
plt.ylabel('dice')
# plt.savefig('/home/safran/theodore/These/manuscrit/latex/watershed_tests/ws-summary.png')
plt.show()

#%%

