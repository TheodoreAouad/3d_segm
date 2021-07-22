import os
import re
from os.path import join
import pathlib

import numpy as np
from tqdm import tqdm

from ssm.utils import sort_by_regex
from ssm.shape import Shape

N_SAMPLES = [1000, 2000]
only_new = True


LABELS_DIR = "/hdd/aouadt/these/projets/3d_segm/data/CT-SCANS/dicoms"
DEST_DIR = "/hdd/aouadt/these/projets/3d_segm/data/CT-SCANS/meshes/iliac"
patient_regex = r"^patient_\d+$"


all_patients = [f for f in os.listdir(LABELS_DIR) if re.match(patient_regex, f)]

for n_sample in N_SAMPLES:
    print(f"Computing n_sample={n_sample}")
    for patient in tqdm(sort_by_regex(all_patients, regex=r'patient_(\d+)')):
        for side in ["right", "left"]:
            path_iliac = join(LABELS_DIR, patient, f"Segmentation-iliac-{side}.obj")
            if os.path.exists(path_iliac):
                dest = join(DEST_DIR, patient, side, f"downsample_{n_sample}")

                if only_new and os.path.exists(dest):
                    print(f'{patient}-{side}-downsample_{n_sample} Already computed. Pass.')
                    continue

                pathlib.Path(dest).mkdir(exist_ok=True, parents=True)
                cur_shape = Shape.load_obj_from_path(path_iliac)
                cur_shape.keep_largest_component()

                cur_shape.uniform_downsample(n_sample)




                pathlib.Path(dest).mkdir(exist_ok=True, parents=True)
                np.save(os.path.join(dest, "vertexes.npy"), cur_shape.vertexes)
                np.save(os.path.join(dest, "faces.npy"), cur_shape.faces)
