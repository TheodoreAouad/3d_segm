import numpy as np
from matplotlib.patches import Polygon

from general.nn.viz import Element, ElementGrouper, ElementSymbolIntersection, ElementSymbolUnion
from deep_morpho.models import LUI


OPERATION_FACTOR = .3

LUI_INVERT_CODE = {v: k for (k, v) in LUI.operation_code.items()}


class ElementLuiCoefs(Element):

    def __init__(self, model, imshow_kwargs={}, fill=True, fill_color='w', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.imshow_kwargs = imshow_kwargs
        self.imshow_kwargs['color'] = self.imshow_kwargs.get('color', 'k')
        self.imshow_kwargs['fill'] = False  # handle fill independantly
        self.fill = fill
        self.fill_color = fill_color

    def add_to_canva(self, canva: "Canva"):
        if self.fill:
            fill_kwargs = self.imshow_kwargs.copy()
            fill_kwargs['fill'] = True
            fill_kwargs['color'] = self.fill_color
            canva.ax.add_patch(Polygon(np.stack([
                self.xy_coords_botleft, self.xy_coords_topleft, self.xy_coords_midright
            ]), closed=True, **fill_kwargs))
        canva.ax.add_patch(Polygon(np.stack([
            self.xy_coords_botleft, self.xy_coords_topleft, self.xy_coords_midright
        ]), closed=True, **self.imshow_kwargs))





class ElementLui(ElementGrouper):
    operation_element_dicts = {'intersection': ElementSymbolIntersection, 'union': ElementSymbolUnion}

    def __init__(self, model, shape, imshow_kwargs={}, v1=None, v2=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.v1 = v1
        self.v2 = v2
        self.imshow_kwargs = imshow_kwargs
        self.imshow_kwargs['color'] = self.imshow_kwargs.get('color', 'k')

        self.element_lui_operation = None

        self.element_lui_coefs = ElementLuiCoefs(model, imshow_kwargs, shape=shape, *args, **kwargs)
        self.add_element(self.element_lui_coefs, key="coefs")


        if self.model.is_activated[0]:
            operation = LUI_INVERT_CODE[self.model.learned_operation[0]]
            shape = self.element_lui_coefs.shape * OPERATION_FACTOR
            self.element_lui_operation = self.operation_element_dicts[operation](
                width=shape[0], height=shape[1],
                xy_coords_mean=self.element_lui_coefs.xy_coords_mean + np.array([0, self.element_lui_coefs.shape[-1] / 2 + 2])
            )
            self.add_element(self.element_lui_operation, key="operation")


class ElementLuiClosest(ElementGrouper):
    operation_element_dicts = {'intersection': ElementSymbolIntersection, 'union': ElementSymbolUnion}

    def __init__(self, model, shape, imshow_kwargs={}, v1=None, v2=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.v1 = v1
        self.v2 = v2
        self.imshow_kwargs = imshow_kwargs
        self.imshow_kwargs['color'] = self.imshow_kwargs.get('color', 'k')

        self.element_lui_operation = None

        self.element_lui_coefs = ElementLuiCoefs(model, imshow_kwargs, shape=shape, *args, **kwargs)
        self.add_element(self.element_lui_coefs, key="coefs")

        operation = LUI_INVERT_CODE[self.model.closest_operation[0]]
        shape = self.element_lui_coefs.shape * OPERATION_FACTOR
        self.element_lui_operation = self.operation_element_dicts[operation](
            width=shape[0], height=shape[1],
            xy_coords_mean=self.element_lui_coefs.xy_coords_mean + np.array([0, self.element_lui_coefs.shape[-1] / 2 + 2])
        )
        self.add_element(self.element_lui_operation, key="operation")
