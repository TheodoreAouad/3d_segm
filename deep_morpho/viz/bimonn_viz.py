import numpy as np

from general.nn.viz import Canva, ElementArrow, ElementImage, ElementGrouper, ElementCircle
from deep_morpho.models import BiMoNN
from general.nn.viz import (
    ElementSymbolDilation, ElementSymbolErosion, ElementSymbolIntersection, ElementSymbolUnion
)
from .element_lui import ElementLui


LUI_HORIZONTAL_FACTOR = 3
LUI_RADIUS_FACTOR = 1
INPUT_RADIUS_FACTOR = .1

NEXT_LAYER_HORIZONTAL_FACTOR = 1
FIRST_LAYER_HORIZONTAL_FACTOR = 1 / INPUT_RADIUS_FACTOR * .5


class BimonnVizualiser:
    operation_elements = {
        'union': ElementSymbolUnion,
        'intersection': ElementSymbolIntersection,
        'erosion': ElementSymbolErosion,
        'dilation': ElementSymbolDilation,
    }

    def __init__(self, model: BiMoNN, ):
        self.model = model
        self.canva = None
        self.layer_groups = []

        self.box_height = self._infer_box_height()

    def _infer_box_height(self):
        return 2 * (
            np.array(self.model.in_channels) *
            np.array(self.model.out_channels) *
            self.max_selem_shape
        ).max()

    @property
    def max_selem_shape(self):
        return np.array(self.model.kernel_size).max(1)

    def draw(self, **kwargs):
        self.canva = Canva(**kwargs)
        cursor = 0
        prev_elements = []
        for layer_idx in range(-1, len(self)):
            group, cur_elements = self.get_layer_group(layer_idx)
            group.translate(np.array([cursor, 0]))

            self.canva.add_element(group, key=f'layer_{layer_idx}')

            for chin, elt in enumerate(prev_elements):
                for chout in range(self.model.out_channels[layer_idx]):
                    self.canva.add_element(ElementArrow.link_elements(
                        elt, group.elements[f"group_layer_{layer_idx}_chout_{chout}"].elements[f"selem_layer_{layer_idx}_chout_{chout}_chin_{chin}"]
                    ))

            if layer_idx == -1:
                cursor += group.shape[0] * (1 + FIRST_LAYER_HORIZONTAL_FACTOR)
            else:
                cursor += group.shape[0] * (1 + NEXT_LAYER_HORIZONTAL_FACTOR)

            prev_elements = cur_elements

        return self.canva


    def get_input_layer(self):
        layer_group = ElementGrouper()
        n_elts = self.model.in_channels[0]
        coords = np.linspace(0, self.box_height, 2*n_elts + 1)[1::2]

        for elt_idx, coord in enumerate(coords):
            layer_group.add_element(ElementCircle(
                xy_coords_mean=np.array([0, coord]),
                radius=INPUT_RADIUS_FACTOR * self.box_height / (2 * n_elts),
                imshow_kwargs={"fill": True},
            ), key=f"input_chan_{elt_idx}")

        return layer_group, [layer_group.elements[f"input_chan_{elt_idx}"] for elt_idx in range(n_elts)]


    def get_layer_group(self, layer_idx):
        if layer_idx == -1:
            return self.get_input_layer()

        layer_group = ElementGrouper()

        n_groups = self.model.out_channels[layer_idx]
        n_per_group = self.model.in_channels[layer_idx]

        coords_group = np.linspace(0, self.box_height, 2*n_groups + 1)[1::2]

        if n_groups > 1:
            height_group = (coords_group[1] - coords_group[0])*.7
        else:
            height_group = self.box_height

        for chout, coord_group in enumerate(coords_group):

            if n_per_group == 1:
                coords_selem = np.zeros(1)
            else:
                coords_selem = np.linspace(0, height_group, 2*n_per_group + 1)[1::2]
            subgroup = ElementGrouper()

            input_lui_elements = []

            # add bises
            for coord_selem_idx, chin in enumerate(range(n_per_group)):
                coord_selem = coords_selem[coord_selem_idx]
                selem = self.model.layers[layer_idx].normalized_weights[chout, chin].detach().cpu().numpy()
                key_selem = f"selem_layer_{layer_idx}_chout_{chout}_chin_{chin}"

                cur_elt = ElementImage(
                    selem,
                    imshow_kwargs={"cmap": "gray", "interpolation": "nearest"},
                    xy_coords_mean=(0, coord_selem)
                )

                subgroup.add_element(cur_elt, key=key_selem)
                input_lui_elements.append(cur_elt)

            # add lui layers
            shape = LUI_RADIUS_FACTOR * np.array([self.max_selem_shape[layer_idx], self.max_selem_shape[layer_idx]])
            subgroup.add_element(ElementLui(
                model=self.model.layers[layer_idx].luis[chout],
                input_elements=input_lui_elements,
                xy_coords_mean=(LUI_HORIZONTAL_FACTOR * self.max_selem_shape[layer_idx], (coords_selem).mean()),
                shape=shape,
                imshow_kwargs={"color": "k"}
            ), key=f"lui_layer_{layer_idx}_chout_{chout}")

            subgroup.set_xy_coords_mean(np.array([0, coord_group]))
            layer_group.add_element(
                subgroup,
                key=f"group_layer_{layer_idx}_chout_{chout}"
            )

        return layer_group, [
            layer_group.elements[f"group_layer_{layer_idx}_chout_{chout}"].elements[f"lui_layer_{layer_idx}_chout_{chout}"]
            for chout in range(n_groups)]

    def __len__(self):
        return len(self.model)
