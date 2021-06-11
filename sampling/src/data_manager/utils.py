from os.path import join

import numpy as np
import nibabel as nib

from all_paths import all_paths


def get_order_from_name(filename: str):
    """
    Extracts the order of one plane of MRI

     against the others.
    Args:
        filename (str): name of the file.
    Returns:
        int: order of the file. Note: the order of 'VERSION' is unknwon.
    """
    letters_to_replace = ['I', 'E', 'B', 'A', 'C', 'D']
    if filename[-1] == '1':
        return 10
    # elif filename == 'VERSION':
    #     return -1
    else:
        res = filename.replace('0', '')
        for letter in letters_to_replace:
            res = res.replace(letter, '')
        try:
            return int(res)
        except ValueError:
            return -1


def get_volume_from_mri(mri, axis_to_stack=0):
    """
    Stacks all the pixel_array of the lines of mri to form a 3D cube.

    Args:
        mri (pd.DataFrame): dataframe containing the pixel arrays
        axis_to_stack (int, optional): Axis to stack to. Defaults to 0.

    Returns:
        ndarray: 3D cube of the slices put together.
    """
    volume = []
    for img in mri.pixel_array:
        volume.append(img)
    return np.stack(volume, axis_to_stack)



def get_nib_atlas(parent=all_paths["atlas_nibs"], patient="hoang"):
    """
    Load a NIB atlas file.

    Args:
        parent: write your description
        all_paths: write your description
        patient: write your description
    """
    return nib.load(join(parent, patient, f"3D{patient}-sacrum.nii"))


def get_segm_atlas(parent=all_paths["atlas_nibs"], patient="hoang"):
    """
    Get the segm_atlas file for a patient

    Args:
        parent: write your description
        all_paths: write your description
        patient: write your description
    """
    sacrum = nib.load(join(parent, patient, f"3D{patient}-sacrum.nii")).get_fdata()
    iliaque = nib.load(join(parent, patient, f"3D{patient}-iliaque.nii")).get_fdata()

    sacrum[iliaque.astype(bool)] = 2

    return sacrum


#     return  nib.load(join(parent, patient, f"3D{patient}-sacrum.nii"))

def get_vol_atlas(parent=all_paths["atlas_nibs"], patient="hoang"):
    """
    Loads the vol_atlas file for a patient

    Args:
        parent: write your description
        all_paths: write your description
        patient: write your description
    """
    return nib.load(join(parent, patient, f"3D{patient}.nii")).get_fdata()
