import os
from os.path import join
import pathlib
import re
from pytorch_lightning.core.saving import convert

from tqdm import tqdm
import pandas as pd
import numpy as np


PATH_DATA_SRC = "/hdd/aouadt/projets/SpondiDetect/SpondiDetect/data/desir-segmented"
PATH_LABEL = "/hdd/aouadt/projets/SpondiDetect/SpondiDetect/data/per_slice_label.csv"
PATH_PATIENTS = "data/deep_morpho/axspa_roi/patients.txt"
PATH_CSV = "data/deep_morpho/axspa_roi/axspa_roi.csv"
PATH_DATA_DST = "data/deep_morpho/axspa_roi/data"

only_new = True


def convert_instance_number_idx(patient, path_slices="/hdd/aouadt/projets/SpondiDetect/SpondiDetect/data/desir_slices"):
    dic = {}
    for slice_ in os.listdir(join(path_slices, patient, 'npy')):
        slice_idx = re.findall(r'slice(\d+)-', slice_)[0]
        inst = int(re.findall(r'inst(\d+)\.', slice_)[0])
        dic[inst] = int(slice_idx)
    return dic


df_label = pd.read_csv(PATH_LABEL)

all_df = []

with open(PATH_PATIENTS, 'r') as f:
    all_patients = f.read().splitlines()

# all_patients = all_patients[50:]

for patient in tqdm(sorted(set(all_patients).intersection(os.listdir(PATH_DATA_SRC)))):

    if only_new and os.path.exists(join(PATH_DATA_DST, patient)):
        continue

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

    all_slices = df_label.loc[df_label['patient_id'] == patient, 'instance_number'].unique().astype(int)
    converter = convert_instance_number_idx(patient)

    pathlib.Path(join(PATH_DATA_DST, patient)).mkdir(exist_ok=True, parents=True)

    for inst in all_slices:
        if inst not in converter.keys():
            continue
        idx = converter[inst]
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
