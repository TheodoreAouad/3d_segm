from enum import Enum

import torch
import numpy as np

from .generator_dataset import GeneratorDataset
from .spalike_generator import SpaLike



class SpalikeSegmEnum(Enum):
    BonesSeparated = 0
    BonesOverlapped = 1
    Roi = 2
    NoSegm = 3



class SpalikeDatasetBase(GeneratorDataset):

    def __init__(
        self,
        image_size: tuple = (256, 256),
        proba_lesion: float = 0.2,
        proba_lesion_locations: dict = {
            "sacrum": 0.2,
            "iliac": 0.2,
        },
        grid_spacing: tuple = (24, 24),
        min_ellipse_axes: int = 13,
        max_ellipse_axes: int = 35,
        period: tuple = (3, 10),
        offset: tuple = (1, 2),
        min_output_ellipse: int = 0,
        max_n_blob_sane: int = 5,
        segm_mode: SpalikeSegmEnum = SpalikeSegmEnum.BonesSeparated,
        merge_input_segm: bool = False,
        normalize: bool = True,
        iliac_dil_coef: float = 2,
        sacrum_dil_coef: float = 2,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.image_size = image_size
        self.proba_lesion = proba_lesion
        self.proba_lesion_locations = proba_lesion_locations
        self.grid_spacing = grid_spacing
        self.min_ellipse_axes = min_ellipse_axes
        self.max_ellipse_axes = max_ellipse_axes
        self.period = period
        self.offset = offset
        self.min_output_ellipse = min_output_ellipse
        self.max_n_blob_sane = max_n_blob_sane
        self.segm_mode = segm_mode
        self.merge_input_segm = merge_input_segm
        self.normalize = normalize
        self.iliac_dil_coef = iliac_dil_coef
        self.sacrum_dil_coef = sacrum_dil_coef

        if self.merge_input_segm and self.segm_mode == SpalikeSegmEnum.BonesSeparated:
            self.segm_mode = SpalikeSegmEnum.BonesOverlapped


    def get_spalike(self):
        return SpaLike(
            image_size=self.image_size,
            proba_lesion=self.proba_lesion,
            proba_lesion_locations=self.proba_lesion_locations,
            grid_spacing=self.grid_spacing,
            min_ellipse_axes=self.min_ellipse_axes,
            max_ellipse_axes=self.max_ellipse_axes,
            period=self.period,
            offset=self.offset,
            min_output_ellipse=self.min_output_ellipse,
            max_n_blob_sane=self.max_n_blob_sane,
            iliac_dil_coef=self.iliac_dil_coef,
            sacrum_dil_coef=self.sacrum_dil_coef,
        )


    def generate_input_target(self):
        self.spalike = self.get_spalike()

        input_, target = self.spalike.generate_image()

        if self.segm_mode == SpalikeSegmEnum.BonesSeparated:
            segm = torch.tensor(np.stack([self.spalike.iliac_segmentation, self.spalike.sacrum_segmentation])).float()
        elif self.segm_mode == SpalikeSegmEnum.BonesOverlapped:
            segm = torch.tensor(self.spalike.iliac_segmentation | self.spalike.sacrum_segmentation).unsqueeze(0).float()
        elif self.segm_mode == SpalikeSegmEnum.Roi:
            segm = torch.tensor(self.spalike.roi).unsqueeze(0).float()
        elif self.segm_mode == SpalikeSegmEnum.NoSegm:
            segm = torch.tensor(np.ones_like(input_)).float()

        input_ = torch.tensor(input_).float().unsqueeze(0)
        target = torch.tensor(target).float()

        if self.normalize:
            input_ = (input_ - 120) / 68  # mean and std of the dataset. std computed by mean of stds for batch of size 100.

        if self.merge_input_segm:
            return input_ * segm, target

        return (input_, segm), target


class SpalikeDataset(SpalikeDatasetBase):
    def __init__(self, *args, **kwargs):
        kwargs["segm_mode"] = SpalikeSegmEnum.BonesSeparated
        kwargs["merge_input_segm"] = False
        super().__init__(*args, **kwargs)


class SpalikeDatasetMerged(SpalikeDatasetBase):
    def __init__(self, *args, **kwargs):
        kwargs["merge_input_segm"] = True
        super().__init__(*args, **kwargs)
