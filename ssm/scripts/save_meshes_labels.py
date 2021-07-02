import os
import pathlib
import re

import numpy as np
from tqdm import tqdm
import nibabel as nib
import skimage.measure as meas

from general.utils import get_most_important_regions
from ssm.utils import sort_by_regex

CUR_IDX = 4
STEP_SIZE = 4
only_new = False

LABELS = ['background', 'liver', 'bladder', 'lungs', 'kidneys', 'bone', 'brain']
LABELS_DIR = "/hdd/datasets/CT-ORG/raw/labels_and_README/"
DEST_DIR = f"/hdd/datasets/CT-ORG/meshes/{LABELS[CUR_IDX]}/labels/"

all_labels = sort_by_regex([os.path.join(LABELS_DIR, filename) for filename in
    os.listdir(LABELS_DIR) if re.findall(r"labels-", filename)])

for idx in tqdm(range(len(all_labels))):
    label_path = all_labels[idx]
    label_name = re.findall(r"labels-\d+", label_path)[0]  # pathlib.Path(label_path).stem
    dest = os.path.join(DEST_DIR, label_name, f"step_size_{STEP_SIZE}")

    seg = np.round(nib.load(label_path).get_fdata()) == CUR_IDX
    precut_volume = seg.sum()
    if precut_volume == 0:
        continue

    path_precut_volume = os.path.join(DEST_DIR, label_name, 'precut_volume_scalar.npy')
    # if os.path.exists(os.path.join(DEST_DIR, label_name, "precut_volume.npy")):
    #     os.rename(os.path.join(DEST_DIR, label_name, "precut_volume.npy"), os.path.join(DEST_DIR, label_name, "precut_volume_scalar.npy"))

    if not (only_new and os.path.exists(path_precut_volume)):
        pathlib.Path(dest).mkdir(exist_ok=True, parents=True)
        np.save(path_precut_volume, precut_volume)

    if only_new and os.path.exists(dest):
        print(f'{label_name}-{STEP_SIZE} Already computed. Pass.')
        continue


    reg = (get_most_important_regions(seg) > 0).astype(int)

    verts, faces, normals, values = meas.marching_cubes(reg, step_size=STEP_SIZE)

    pathlib.Path(dest).mkdir(exist_ok=True, parents=True)
    np.save(os.path.join(dest, "vertexes.npy"), verts)
    np.save(os.path.join(dest, "faces.npy"), faces)
    np.save(os.path.join(dest, "normals.npy"), normals)
    np.save(os.path.join(dest, "values.npy"), values)
    # if os.path.exists(os.path.join(dest, "volume.npy")):
    #     reg = np.load(os.path.join(dest, "volume.npy"))
    #     os.remove(os.path.join(dest, "volume.npy"))
    np.save(os.path.join(dest, "volume_scalar.npy"), reg.sum())
