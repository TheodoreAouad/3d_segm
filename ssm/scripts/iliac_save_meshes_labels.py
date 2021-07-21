import os
import pathlib

import numpy as np
from tqdm import tqdm
import skimage.measure as meas

from ssm.read_iliac import get_iliac_segmentations

CUR_IDX = 1
STEP_SIZE = 1
only_new = True


LABELS_DIR = "/hdd/aouadt/these/projets/3d_segm/data/CT-SCANS/dicoms"
DEST_DIR = "/hdd/aouadt/these/projets/3d_segm/data/CT-SCANS/meshes/iliac"
patient_regex = r"^patient_\d+$"

all_segms = get_iliac_segmentations(
    path_folder=LABELS_DIR,
    regex_filename=patient_regex,
    right_name="Segmentation-iliac-right.nii",
    left_name="Segmentation-iliac-left.nii",
)

for patient_id, segms in tqdm(all_segms.items()):

    # if patient_id != 'patient_8':
    #     continue

    for side, nib_seg in segms.items():
        dest = os.path.join(DEST_DIR, patient_id, side, f"step_size_{STEP_SIZE}")


        path_volume = os.path.join(DEST_DIR, patient_id, side, 'volume_scalar.npy')

        write_volume = False
        if not (only_new and os.path.exists(path_volume)):
            pathlib.Path(path_volume).parent.mkdir(exist_ok=True, parents=True)
            # np.save(path_volume, volume)
            write_volume = True

        if only_new and os.path.exists(dest):
            print(f'{patient_id}-{side}-{STEP_SIZE} Already computed. Pass.')
            continue

        seg = np.round(nib_seg.get_fdata()) == CUR_IDX

        if write_volume:
            volume = seg.sum()
            if volume == 0:
                continue
            np.save(path_volume, volume)

        verts, faces, normals, values = meas.marching_cubes(seg, step_size=STEP_SIZE)

        pathlib.Path(dest).mkdir(exist_ok=True, parents=True)
        np.save(os.path.join(dest, "vertexes.npy"), verts)
        np.save(os.path.join(dest, "faces.npy"), faces)
        np.save(os.path.join(dest, "normals.npy"), normals)
        np.save(os.path.join(dest, "values.npy"), values)
