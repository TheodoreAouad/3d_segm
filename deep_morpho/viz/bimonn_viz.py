import numpy as np

from .skeleton_morp_viz import SkeletonMorpViz
from .elt_generator_bimonn import (
    EltGeneratorBise, EltGeneratorLui, EltGeneratorConnectLuiBise, EltGeneratorBiseBinary,
    EltGeneratorConnectLuiBiseClosest
)
from .elt_generator_init import EltGeneratorInitCircle


class BimonnVizualiser(SkeletonMorpViz):

    def __init__(self, model, mode: str = "weights", **kwargs):
        self.model = model
        assert mode in ["weights", "learned", "closest"]

        if mode == "weights":
            kwargs.update({
                "elt_generator_bise": EltGeneratorBise(model),
                "elt_generator_lui": EltGeneratorLui(model),
                "elt_generator_connections": EltGeneratorConnectLuiBise(model=model, binary_mode=False),
            })

        elif mode == "learned":
            kwargs.update({
                "elt_generator_bise": EltGeneratorBiseBinary(model, learned=True),
                "elt_generator_lui": EltGeneratorLui(model),
                "elt_generator_connections": EltGeneratorConnectLuiBise(model=model, binary_mode=True),
            })

        elif mode == "closest":
            kwargs.update({
                "elt_generator_bise": EltGeneratorBiseBinary(model, learned=False),
                "elt_generator_lui": EltGeneratorLui(model, learned=False),
                "elt_generator_connections": EltGeneratorConnectLuiBiseClosest(model=model, ),
            })

        super().__init__(in_channels=model.in_channels, out_channels=model.out_channels, **kwargs)
        self.elt_generator_init = EltGeneratorInitCircle(
            radius=self.box_height / (2 * model.in_channels[0]),
        )

    @property
    def max_selem_shape(self):
        return np.array(self.model.kernel_size).max(1)
