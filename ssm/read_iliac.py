import re
from typing import Dict
import os
from os.path import join

import nibabel as nib


def get_iliac_segmentations(
    path_folder: str,
    regex_filename: str = r"^patient_\d+$",
    left_name: str = "Segmentation-iliac-left.nii",
    right_name: str = "Segmentation-iliac-right.nii",
) -> Dict:

    all_patients = [f for f in os.listdir(path_folder) if re.match(regex_filename, f)]
    all_segm = {}

    for patient in all_patients:
        patient_path = join(path_folder, patient)
        all_segm[patient] = {}

        if os.path.exists(join(patient_path, left_name)):
            all_segm[patient]['left'] = nib.load(join(patient_path, left_name))
        if os.path.exists(join(patient_path, right_name)):
            all_segm[patient]['right'] = nib.load(join(patient_path, right_name))


    return all_segm
