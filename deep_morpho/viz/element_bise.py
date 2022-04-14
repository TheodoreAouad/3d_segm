import numpy as np

from general.nn.viz import ElementGrouper, ElementImage, ElementSymbolDilation, ElementSymbolErosion, ElementCircle
from deep_morpho.models import BiSE


MAX_WIDTH_COEF = 1
BISE_INVERT_CODE = {v:k for (k, v) in BiSE.operation_code.items()}

class ElementBiseWeightsChan(ElementImage):

    def __init__(self, model, chout=0, *args, **kwargs):
        self.model = model
        self.chout = chout
        super().__init__(image=None, *args, **kwargs)

        self.imshow_kwargs['vmin'] = self.imshow_kwargs.get('vmin', 0)
        self.imshow_kwargs['vmax'] = self.imshow_kwargs.get('vmax', 1)
        self.imshow_kwargs['cmap'] = self.imshow_kwargs.get('cmap', 'gray')

    @property
    def image(self):
        return self.model._normalized_weight[self.chout, 0].detach().cpu().numpy()


class ElementBiseSelemChan(ElementGrouper):
    operation_elements_dict = {'dilation': ElementSymbolDilation, "erosion": ElementSymbolErosion}

    def __init__(self, model, chout=0, learned=True, v1=0, v2=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.chout = chout
        self.learned = learned
        self.v1 = v1
        self.v2 = v2
        self.kernel_shape = self.model.weight.shape[-2:]

        if self.learned and not model.is_activated[chout]:
            self.selem_element = ElementCircle(radius=self.kernel_shape[-1] / 2, **kwargs)
            self.operation_element = None
        else:
            selem = model.closest_selem[..., chout]
            operation = BISE_INVERT_CODE[model.closest_operation[chout]]

            radius_operation = max(2, self.kernel_shape[-1] / 4)

            self.selem_element = ElementImage(selem, imshow_kwargs={"interpolation": "nearest", "vmin": 0, "vmax": 1}, **kwargs)
            self.operation_element = self.operation_elements_dict[operation](
                radius=radius_operation, xy_coords_mean=self.selem_element.xy_coords_mean + np.array([0, self.kernel_shape[-1] / 2 + radius_operation / 2])
            )
            self.add_element(self.operation_element, key="operation")

        self.add_element(self.selem_element, key="selem")
