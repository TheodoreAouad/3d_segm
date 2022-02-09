import numpy as np
from matplotlib.patches import Polygon

from general.nn.viz import Element, ElementArrow


MAX_WIDTH_COEF = 1

class ElementLui(Element):

    def __init__(self, model, input_elements, imshow_kwargs={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.input_elements = input_elements
        self.imshow_kwargs = imshow_kwargs

        self.imshow_kwargs['color'] = self.imshow_kwargs.get('color', 'k')

    def add_to_canva(self, canva: "Canva"):
        canva.ax.add_patch(Polygon(np.stack([
            self.xy_coords_botleft, self.xy_coords_topleft, self.xy_coords_midright
        ]), closed=True, fill=False, **self.imshow_kwargs))
        self.link_input_lui(canva)


    def link_input_lui(self, canva, max_width_coef=MAX_WIDTH_COEF):
        coefs = self.model.positive_weight[0].detach().cpu().numpy()
        coefs = coefs / coefs.max()

        for elt_idx, elt in enumerate(self.input_elements):
            canva.add_element(ElementArrow.link_elements(
                elt, self, width=coefs[elt_idx] * max_width_coef,
            ))
