from .element_bise import ElementBiseSelemChan, ElementBiseWeightsChan

from general.nn.viz import ElementArrow
from .element_arrow_no import ElementArrowNo
from .element_generator import EltGenerator
from .element_lui import ElementLui, ElementLuiClosest



class EltGeneratorBise(EltGenerator):

    def __init__(self, bimonn_model):
        super().__init__()
        self.bimonn_model = bimonn_model

    def generate(self, layer_idx, chin, chout, xy_coords_mean, **kwargs):
        bise_layer = self.bimonn_model.layers[layer_idx].bises[chin]
        return ElementBiseWeightsChan(model=bise_layer, chout=chout, xy_coords_mean=xy_coords_mean, **kwargs)


class EltGeneratorBiseBinary(EltGenerator):
    """ Generator for BiSE element binary. Choose either learned = True for learned or learned = False
    for closest selem.
    """
    def __init__(self, bimonn_model, learned: bool = True):
        super().__init__()
        self.bimonn_model = bimonn_model
        self.learned = learned

    def generate(self, layer_idx, chin, chout, xy_coords_mean, **kwargs):
        bise_layer = self.bimonn_model.layers[layer_idx].bises[chin]
        bise_layer.update_learned_selems()

        return ElementBiseSelemChan(model=bise_layer, learned=self.learned, chout=chout, xy_coords_mean=xy_coords_mean, **kwargs)



class EltGeneratorLui(EltGenerator):

    def __init__(self, bimonn_model, learned: bool = True, imshow_kwargs={"color": "k"}):
        super().__init__()
        self.bimonn_model = bimonn_model
        self.imshow_kwargs = imshow_kwargs
        self.learned = learned

    def generate(self, layer_idx, chout, xy_coords_mean, shape):
        lui_layer = self.bimonn_model.layers[layer_idx].luis[chout]
        lui_layer.update_learned_sets()

        if self.learned:
            constructor = ElementLui
        else:
            constructor = ElementLuiClosest

        return constructor(
            lui_layer,
            xy_coords_mean=xy_coords_mean,
            shape=shape,
            imshow_kwargs=self.imshow_kwargs,
        )


class EltGeneratorConnectLuiBiseBase(EltGenerator):

    def __init__(self, model, max_width_coef=1):
        super().__init__()
        self.model = model
        self.max_width_coef = max_width_coef

    def generate(self, group, layer_idx, chout, chin):
        bise_elt = group[f"bise_layer_{layer_idx}_chout_{chout}_chin_{chin}"]
        lui_elt = group[f"lui_layer_{layer_idx}_chout_{chout}"]["coefs"]

        width = self.infer_width(lui_elt, chin)

        activation_P = bise_elt.model.activation_P[chout]
        if activation_P > 0 or width == 0:
            return ElementArrow.link_elements(bise_elt, lui_elt, width=width)
        return ElementArrowNo.link_elements(bise_elt, lui_elt, height_circle=max(self.model.kernel_size[layer_idx])*0.7, width=width)

    def infer_width(self, lui_elt, chin):
        raise NotImplementedError


class EltGeneratorConnectLuiBise(EltGeneratorConnectLuiBiseBase):

    def __init__(self, binary_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_mode = binary_mode

    def infer_width(self, lui_elt, chin):
        model= lui_elt.model

        if self.binary_mode and model.is_activated[0]:
            return float(model.learned_set[0, chin])


        coefs = model.positive_weight[0].detach().cpu().numpy()
        coefs = coefs / coefs.max() * self.max_width_coef
        return coefs[chin]


class EltGeneratorConnectLuiBiseClosest(EltGeneratorConnectLuiBiseBase):

    def infer_width(self, lui_elt, chin):
        model= lui_elt.model
        return float(model.closest_set[0, chin])
