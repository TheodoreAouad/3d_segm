import glob
import re
import pathlib
from os.path import join

import pydicom
import pandas as pd
import numpy as np
from pydicom.tag import Tag

from .utils import get_order_from_name
from all_paths import all_paths


def get_path_patient(
    patient: str = 'hoang',
    parent_path: str = all_paths['atlas_dicoms'],
    st: str = 'images/pat00000/st000000',
    seq: str = 'se000003',
    *args,
    **kwargs
):
    """
    Return the path of a patient to a parent path.

    Args:
        patient: write your description
        parent_path: write your description
        all_paths: write your description
        st: write your description
        seq: write your description
    """
    return join(parent_path, 'irm_{}'.format(patient), st, seq)


def get_mri_atlas(*args, **kwargs):
    """
    Get the MRI of the Stanford IMT s associated atlas.

    Args:
    """
    mri_path = get_path_patient(*args, **kwargs)
    all_paths_imgs = glob.glob(join(mri_path, 'mr0*'), recursive=False)

    return get_mri(all_paths_imgs, kwargs.get('get_dicom', False), file_regex="mr")


def get_mri(
    all_paths_imgs: str,
    get_dicom: bool = False,
    sort: bool = True,
    file_regex: str = r"\d{2}-\d{2}",
):
    """
    Base function to read a sequence of dicoms.

    Args:
        all_paths_imgs (str): Path to the folder containing the dicom files.
        get_dicom (bool, optional): Returns the dicom in the dataframe. Costy. Defaults to False.

    Returns:
        pandas.core.frame.DataFrame: Dataframe where each row is a slice.
    """
    mri = []
    for path_img in all_paths_imgs:
        try:
            current_mri = pydicom.dcmread(path_img)
        except pydicom.errors.InvalidDicomError:
            # print('Error while opening dicom for {}'.format(path_img))
            continue

        mri.append(pd.DataFrame({
            'patient': [re.findall(file_regex, path_img)[0] if re.search(file_regex, path_img) else ''],
            'path': [path_img],
            'filename': [pathlib.Path(path_img).stem],
            'dicom': [current_mri if get_dicom else None],
            'order': [get_order_from_name(pathlib.Path(path_img).stem)],
            'pixel_array': [current_mri.pixel_array.astype(np.int16)],
            'cosdirs': [[float(k) for k in current_mri[Tag(0x00200037)].value] if current_mri.get(Tag(0x00200037)) is not None else []],
            'ImagePosition': [[float(k) for k in current_mri.get(Tag(0x00200032)).value] if current_mri.get(Tag(0x00200032)) is not None else None],
            'y_pos': [float(current_mri.get(Tag(0x00200032)).value[1]) if current_mri.get(Tag(0x00200032)) is not None else None],
            'PixelSpacing': [[float(k) for k in current_mri.get(Tag(0x00280030)).value] if current_mri.get(Tag(0x00280030)) is not None else None],
            'InstanceNumber': [int(current_mri.get(Tag(0x00200013)).value) if current_mri.get(Tag(0x00200013)) is not None else None],
            'SliceLocation': [
                float(current_mri.get(Tag(0x00201041)).value)
                if current_mri.get(Tag(0x00201041)) is not None else None],
            'SliceThickness': [
                float(current_mri.get(Tag(0x00180050)).value)
                if current_mri.get(Tag(0x00201041)) is not None else None],
            'RepetitionTime': [float(current_mri[Tag(0x00180080)].value) if current_mri.get(Tag(0x00180080)) is not None else None],
        }))
    mri = pd.concat(mri, sort=True)
    if sort:
        mri = mri.sort_values('y_pos')
        # if mri.SliceLocation.iloc[0] is not None:
        #     mri = mri.sort_values('SliceLocation')
        # else:
        #     mri = mri.sort_values('order')
    return mri.reset_index(drop=True)
