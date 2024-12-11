import os
from os.path import join
import re

import pandas as pd
import numpy as np

import ssm.utils as su


MESH_DIR = "/hdd/aouadt/these/projets/3d_segm/data/CT-SCANS/meshes/iliac"
DEST = "/hdd/aouadt/these/projets/3d_segm/data/CT-SCANS/meshes/iliac/labels_descriptions.csv"

all_dfs = []

organ = "iliac"


def get_sorted_dirs_by_regex(path_parent, regex):
    """
    Returns a list of directories that match the regex.

    Args:
        path_parent: write your description
        regex: write your description
    """
    all_dirs = [f for f in os.listdir(path_parent) if os.path.isdir(join(path_parent, f))]
    return su.sort_by_regex(all_dirs, regex=regex)


# for path_label in su.sort_by_regex(os.listdir(MESH_DIR), regex=r'patient_(\d+)'):
for path_label in get_sorted_dirs_by_regex(MESH_DIR, regex=r'patient_(\d+)'):
    for side in os.listdir(join(MESH_DIR, path_label)):
        for type_mesh in sorted(os.listdir(join(MESH_DIR, path_label, side))):
            full_path = join(MESH_DIR, path_label, side, type_mesh)
            if not(os.path.isdir(full_path)):
                continue

            dic_df = {
                'id': [path_label],
                'organ': [organ],
                'side': [side],
                'step_size': [int(re.findall(r"step_size_(\d+)", type_mesh)[0]) if "step_size" in type_mesh else np.nan],
                'n_vertexes': [len(np.load(join(full_path, 'vertexes.npy')))],
                'n_faces': [len(np.load(join(full_path, 'faces.npy')))],
                'full_path': [full_path],
            }


            volume_path = join(full_path, 'volume_scalar.npy')
            if os.path.exists(volume_path):
                volume = np.load(volume_path)
                # precut_volume = np.load(join(ORGAN_DIR, path_label, 'precut_volume_scalar.npy'))
                dic_df['volume'] = [volume]
                # dic_df['cut_ratio'] = [volume / precut_volume]

            all_dfs.append(pd.DataFrame(dic_df))

all_dfs = pd.concat(all_dfs)
all_dfs.to_csv(DEST, index=False)
