from typing import Optional
import pathlib

import numpy as np
import nibabel as nib


def convert_to_nii(ar: np.ndarray, affine: np.ndarray):
    return nib.Nifti1Image(ar, affine)


def save_as_nii(path: str, ar: np.ndarray, affine: np.ndarray, dtype: Optional[type] = None):
    if dtype is not None:
        ar = ar.astype(dtype)
    nib_ar = convert_to_nii(ar, affine)
    pathlib.Path(path).parent.mkdir(exist_ok=True, parents=True)
    nib.save(nib_ar, path)
    return path
