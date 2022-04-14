from typing import List
import pathlib

import numpy as np

from .element_generator import EltGenerator
from general.nn.viz import Canva, ElementArrow, Element, ElementGrouper


class SkeletonMorpViz:

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        elt_generator_init: EltGenerator = None,
        elt_generator_bise: EltGenerator = None,
        elt_generator_lui: EltGenerator = None,
        elt_generator_connections: EltGenerator = None,
        max_selem_shape=10,
        lui_radius_factor=1,
        lui_horizontal_factor=3,
        first_layer_horizontal_factor=3,
        next_layer_horizontal_factor=1,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.elt_generator_init = elt_generator_init
        self.elt_generator_bise = elt_generator_bise
        self.elt_generator_lui = elt_generator_lui
        self.elt_generator_connections = elt_generator_connections

        self._max_elem_shape = max_selem_shape * np.ones(len(self))
        self.lui_radius_factor = lui_radius_factor
        self.lui_horizontal_factor = lui_horizontal_factor
        self.first_layer_horizontal_factor = first_layer_horizontal_factor
        self.next_layer_horizontal_factor = next_layer_horizontal_factor

        self.box_height = self._infer_box_height()

    def _infer_box_height(self):
        return 2 * (
            np.array(self.in_channels) *
            np.array(self.out_channels) *
            np.maximum(self.max_selem_shape, 5)
        ).max()

    def __len__(self):
        return len(self.in_channels)

    def draw(self, **kwargs):
        self.canva = Canva(**kwargs)
        cursor = 0
        prev_elements = []
        for layer_idx in range(-1, len(self)):
            group, cur_elements = self.get_layer_group(layer_idx)
            group.set_xy_coords_midleft(np.array([cursor, 0]))

            self.canva.add_element(group, key=f'layer_{layer_idx}')

            for chin, elt in enumerate(prev_elements):
                for chout in range(self.model.out_channels[layer_idx]):
                    self.canva.add_element(ElementArrow.link_elements(
                        elt, group[f"group_layer_{layer_idx}_chout_{chout}"][f"bise_layer_{layer_idx}_chout_{chout}_chin_{chin}"]
                    ))

            if layer_idx == -1:
                cursor += group.shape[0] * (1 + self.first_layer_horizontal_factor)
            else:
                cursor += group.shape[0] * (1 + self.next_layer_horizontal_factor)

            prev_elements = cur_elements

        return self.canva

    @property
    def max_selem_shape(self):
        return self._max_elem_shape

    def get_input_layer(self):
        layer_group = ElementGrouper()
        n_elts = self.in_channels[0]
        coords = self._get_coords_selem(self.box_height, n_elts)

        for chan, coord in enumerate(coords):
            self.add_init_chin_to_group(group=layer_group, chan=chan, coord=coord,)

        return layer_group, [layer_group.elements[f"input_chan_{elt_idx}"] for elt_idx in range(n_elts)]

    def get_layer_group(self, layer_idx):
        if layer_idx == -1:
            return self.get_input_layer()

        layer_group = ElementGrouper()

        out_channels = self.model.out_channels[layer_idx]
        in_channels = self.model.in_channels[layer_idx]

        # coords_group = np.linspace(0, self.box_height, 2*out_channels + 1)[1::2]
        coords_group = self._get_coords_selem(self.box_height, out_channels)
        height_group = self._get_height_group(coords_group, out_channels)

        for chout, coord_group in enumerate(coords_group):

            coords_selem = self._get_coords_selem(height_group, in_channels)
            subgroup = ElementGrouper()

            # add bises
            for chin, coord_selem in enumerate(coords_selem):
                self.add_bise_to_group(subgroup, layer_idx, chout, chin, coord_selem)

            # add lui layers
            self.add_lui_to_group(subgroup, layer_idx, chout, coords_selem.mean())

            # add connections between lui and bises
            for chin in range(in_channels):
                self.add_connections_to_group(subgroup, layer_idx, chin, chout)

            subgroup.set_xy_coords_mean(np.array([0, coord_group]))
            layer_group.add_element(
                subgroup,
                key=f"group_layer_{layer_idx}_chout_{chout}"
            )

        return layer_group, [
            layer_group.elements[f"group_layer_{layer_idx}_chout_{chout}"].elements[f"lui_layer_{layer_idx}_chout_{chout}"]
            for chout in range(out_channels)]

    def add_init_chin_to_group(self, group, chan, coord):
        elt = self.elt_generator_init.generate(
            chan=chan, xy_coords_mean=np.array([0, coord])
        )

        if elt is None:
            return

        key_init = f"input_chan_{chan}"
        group.add_element(elt, key=key_init)

        return group

    def add_bise_to_group(self, group, layer_idx, chout, chin, coord_selem) -> Element:
        """ Add bise to the group. Returns the element to link to the LUI.
        """
        elt = self.elt_generator_bise.generate(
            layer_idx=layer_idx, chin=chin, chout=chout, xy_coords_mean=np.array([0, coord_selem])
        )

        if elt is None:
            return

        key_selem = f"bise_layer_{layer_idx}_chout_{chout}_chin_{chin}"
        group.add_element(elt, key=key_selem)

        return group

    def add_lui_to_group(self, group, layer_idx, chout, coord_lui) -> Element:
        # Move the part of the shape with the LUI element?
        shape = self.lui_radius_factor * np.array([self.max_selem_shape[layer_idx], self.max_selem_shape[layer_idx]])
        elt = self.elt_generator_lui.generate(
            layer_idx=layer_idx,
            chout=chout,
            xy_coords_mean=(self.lui_horizontal_factor * self.max_selem_shape[layer_idx], coord_lui),
            shape=shape,
        )

        if elt is None:
            return

        key_lui = f"lui_layer_{layer_idx}_chout_{chout}"
        group.add_element(elt, key=key_lui)

        return group

    def add_connections_to_group(self, group, layer_idx, chin, chout) -> Element:
        elt = self.elt_generator_connections.generate(
            group=group,
            layer_idx=layer_idx,
            chout=chout,
            chin=chin,
        )

        if elt is None:
            return

        key_connections = f"connections_layer_{layer_idx}_chout_{chout}_chin_{chin}"
        group.add_element(elt, key=key_connections)

        return group

    def _get_height_group(self, coords_group, n_groups):
        if n_groups > 1:
            return (coords_group[1] - coords_group[0])*.7
        return self.box_height

    def _get_coords_selem(self, height_group, n_per_group):
        if n_per_group == 1:
            return np.zeros(1)
        return np.linspace(0, height_group, 2*n_per_group + 1)[1::2]

    def save_fig(self, savepath: str, **kwargs):
        pathlib.Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        self.draw(**kwargs)
        self.canva.fig.savefig(savepath)

    def get_fig(self, **kwargs):
        self.draw(**kwargs)
        return self.canva.fig
