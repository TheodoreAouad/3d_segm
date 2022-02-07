import numpy as np

from .element import Element


class ElementArrow(Element):

    def __init__(
        self,
        x, y, dx, dy, arrow_kwargs=dict(),
        **kwargs
    ):
        super().__init__(
            shape=np.array([np.abs(dx), np.abs(dy)]),
            xy_coords_mean=np.array([x, y]) + .5 * np.array([dx, dy]),
            **kwargs
        )
        self.x, self.y, self.dx, self.dy = x, y, dx, dy
        self.arrow_kwargs = arrow_kwargs

    def add_to_canva(self, canva: "Canva", ):
        return canva.ax.arrow(self.x, self.y, self.dx, self.dy, **self.arrow_kwargs)

    @staticmethod
    def link_elements(
        elt1: Element,
        elt2: Element,
        link1="adapt",
        link2="adapt",
        length_includes_head=True, width=.1, **kwargs
    ):
        if link1 == "adapt" or link2 == "adapt":
            link1, link2 = ElementArrow.adapt_link(elt1, elt2, link1, link2)

        if isinstance(link1, str):
            x1, y1 = getattr(elt1, f"xy_coords_{link1}")
        elif isinstance(link1, (tuple, np.ndarray)):
            x1, y1 = link1
        else:
            raise ValueError('link1 must be string or tuple or numpy array.')

        if isinstance(link2, str):
            x2, y2 = getattr(elt2, f"xy_coords_{link2}")
        elif isinstance(link2, (tuple, np.ndarray)):
            x2, y2 = link2
        else:
            raise ValueError('link2 must be string or tuple or numpy array.')

        kwargs.update({
            "length_includes_head": length_includes_head,
            "width": width
        })

        return ElementArrow(x1, y1, x2-x1, y2-y1, arrow_kwargs=kwargs)

    @staticmethod
    def adapt_link(elt1, elt2, link1, link2):
        dx, dy = elt2.barycentre - elt1.barycentre

        if link1 == "adapt":
            if dx > 0:
                link1 = "midright"
            elif dx < 0:
                link1 = "midleft"
            elif dy > 0:
                link1 = "midtop"
            else:
                link1 = "midbot"

        if link2 == "adapt":
            if dx < 0:
                link2 = "midright"
            elif dx > 0:
                link2 = "midleft"
            elif dy < 0:
                link2 = "midtop"
            else:
                link2 = "midbot"

        return link1, link2
