import os
import re

import pandas as pd
import numpy as np

import ssm.utils as su


MESH_DIR = "/hdd/datasets/CT-ORG/meshes/"
DEST = "/hdd/datasets/CT-ORG/meshes/labels_description.csv"

all_dfs = []

organs = [dir for dir in os.listdir(MESH_DIR) if os.path.isdir(os.path.join(MESH_DIR, dir))]
for organ in organs:
    ORGAN_DIR = os.path.join(MESH_DIR, organ, 'labels')
    for path_label in su.sort_by_regex(os.listdir(ORGAN_DIR)):
        for step_size in sorted(os.listdir(os.path.join(ORGAN_DIR, path_label))):
            full_path = os.path.join(ORGAN_DIR, path_label, step_size)
            if not(os.path.isdir(full_path)):
                continue

            dic_df = {
                'id': [path_label],
                'step_size': [int(re.findall(r"step_size_(\d+)", step_size)[0])],
                'n_vertexes': [len(np.load(os.path.join(full_path, 'vertexes.npy')))],
                'n_faces': [len(np.load(os.path.join(full_path, 'faces.npy')))],
                'full_path': [full_path],
                'organ': [organ],
            }


            volume_path = os.path.join(full_path, 'volume_scalar.npy')
            if os.path.exists(volume_path):
                volume = np.load(volume_path)
                precut_volume = np.load(os.path.join(ORGAN_DIR, path_label, 'precut_volume_scalar.npy'))
                dic_df['volume'] = [volume]
                dic_df['cut_ratio'] = [volume / precut_volume]

            all_dfs.append(pd.DataFrame(dic_df))
all_dfs = pd.concat(all_dfs)

all_dfs.to_csv(DEST, index=False)
