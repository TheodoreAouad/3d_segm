import numpy as np

from .element import Element
from .plot_symbols import plot_union_on_ax, plot_intersection_on_ax, plot_erosion_on_ax, plot_dilation_on_ax


class ElementSymbolUnion(Element):

    def __init__(self, xy_coords_mean, shape=(1, 1), prop_arc=.3, imshow_kwargs={}, **kwargs):
        super().__init__(np.array(shape), xy_coords_mean=xy_coords_mean, **kwargs)
        self.prop_arc = prop_arc
        self.imshow_kwargs = imshow_kwargs

    def add_to_canva(self, canva: "Canva"):
        return plot_union_on_ax(
            canva.ax, self.xy_coords_mean, width=self.shape[0], height=self.shape[1], prop_arc=self.prop_arc, **self.imshow_kwargs
        )


class ElementSymbolIntersection(Element):

    def __init__(self, xy_coords_mean, shape=(1, 1), prop_arc=.3, imshow_kwargs={}, **kwargs):
        super().__init__(np.array(shape), xy_coords_mean=xy_coords_mean, **kwargs)
        self.prop_arc = prop_arc
        self.imshow_kwargs = imshow_kwargs

    def add_to_canva(self, canva: "Canva"):
        return plot_intersection_on_ax(
            canva.ax, self.xy_coords_mean, width=self.shape[0], height=self.shape[1], prop_arc=self.prop_arc, **self.imshow_kwargs
        )


class ElementSymbolDilation(Element):

    def __init__(self, xy_coords_mean, radius=1, imshow_kwargs={}, **kwargs):
        super().__init__((radius, radius), xy_coords_mean=xy_coords_mean, **kwargs)
        self.radius = radius
        self.imshow_kwargs = imshow_kwargs

    def add_to_canva(self, canva: "Canva"):
        return plot_dilation_on_ax(
            canva.ax, self.xy_coords_mean, radius=self.radius, **self.imshow_kwargs
        )


class ElementSymbolErosion(Element):

    def __init__(self, xy_coords_mean, radius=1, imshow_kwargs={}, **kwargs):
        super().__init__((radius, radius), xy_coords_mean=xy_coords_mean, **kwargs)
        self.radius = radius
        self.imshow_kwargs = imshow_kwargs

    def add_to_canva(self, canva: "Canva"):
        return plot_erosion_on_ax(
            canva.ax, self.xy_coords_mean, radius=self.radius, **self.imshow_kwargs
        )
