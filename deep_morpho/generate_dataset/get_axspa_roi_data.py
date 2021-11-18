import os
from os.path import join
import pathlib

from tqdm import tqdm
import pandas as pd
import numpy as np


PATH_DATA_SRC = "/hdd/aouadt/projets/SpondiDetect/SpondiDetect/data/desir-segmented"
PATH_PATIENTS = "data/deep_morpho/axspa_roi/patients.txt"
PATH_CSV = "data/deep_morpho/axspa_roi/axspa_roi.csv"
PATH_DATA_DST = "data/deep_morpho/axspa_roi/data"


all_df = []

with open(PATH_PATIENTS, 'r') as f:
    all_patients = f.read().splitlines()

# all_patients = all_patients[100:150]

for patient in tqdm(sorted(set(all_patients).intersection(os.listdir(PATH_DATA_SRC)))):
    path = join(PATH_DATA_SRC, patient)

    if os.path.exists(join(path, 'true_target3d.npy')):
        # segm_src = join(path, 'true_target3d.npy')
        segm = np.load(join(path, 'true_target3d.npy'))
    elif os.path.exists(join(path, 'dl_pred_target3d.npy')):
        segm = np.load(join(path, 'dl_pred_target3d.npy'))
        # segm_src = join(path, 'dl_pred_target3d.npy')

    roi_src = join(PATH_DATA_SRC, patient, 'no_wings', 'roi_iliac_sacrum', 'rois_stir.npy')
    roi = np.load(join(PATH_DATA_SRC, patient, 'no_wings', 'roi_iliac_sacrum', 'rois_stir.npy'))
    if segm.shape != roi.shape:
        print(patient, "failed")
        continue

    pathlib.Path(join(PATH_DATA_DST, patient)).mkdir(exist_ok=True, parents=True)

    for idx in range(segm.shape[-1]):
        path_segm = join(PATH_DATA_DST, patient, f"input_{idx}.npy")
        path_roi = join(PATH_DATA_DST, patient, f"target_{idx}.npy")

        # shutil.copyfile(segm_src, path_segm)
        # shutil.copyfile(roi_src, path_roi)

        np.save(path_segm, segm[..., idx])
        np.save(path_roi, roi[..., idx])

        all_df.append(pd.DataFrame({
            'patient': [patient],
            'resolution': [segm.shape[:-1]],
            'slice_idx': [idx],
            'path_segm': [path_segm],
            'path_roi': [path_roi],
            'value_bg': [-1],
        }))

all_df = pd.concat(all_df)
all_df.to_csv(PATH_CSV, index=False)
