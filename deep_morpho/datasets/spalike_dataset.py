import torch
import numpy as np

from .generator_dataset import GeneratorDataset
from .spalike_generator import SpaLike


class SpalikeDataset(GeneratorDataset):

    def __init__(
        self,
        image_size,
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
        max_n_blob: int = 5,
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
        self.max_n_blob = max_n_blob


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
            max_n_blob=self.max_n_blob,
        )


    def generate_input_target(self):
        self.spalike = self.get_spalike()

        input_, segm, target = self.spalike.generate_image()

        value_iliac = self.spalike.bone_generator._iliac_segm_value
        value_sacrum = self.spalike.bone_generator._sacrum_segm_value
        segm = torch.tensor(np.stack([segm == value_iliac, segm == value_sacrum])).float()

        input_ = torch.tensor(input_).float()
        target = torch.tensor(target).float()

        return input_, segm, target
