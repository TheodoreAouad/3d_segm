from typing import Tuple, Union

import numpy as np
import cv2

from .element import Element


class ElementImage(Element):

    def __init__(self, image: np.ndarray, new_shape=None, imshow_kwargs={}, *args, **kwargs):
        super().__init__(shape=np.array(image.shape), *args, **kwargs)
        self.image = image
        self.imshow_kwargs = imshow_kwargs

        if new_shape is not None:
            self.resize(new_shape)


    @property
    def img(self):
        return self.image

    def add_to_canva(self, canva: "Canva", new_shape=None, coords=None, coords_type="barycentre", imshow_kwargs=None):
        if new_shape is not None:
            self.resize(new_shape)

        if imshow_kwargs is None:
            imshow_kwargs = self.imshow_kwargs

        if coords is not None:
            if coords_type == "barycentre":
                self.set_xy_coords_mean(coords)
            elif coords_type == "botleft":
                self.set_xy_coords_botleft(coords)

        canva.ax.imshow(self.image, extent=(
            self.xy_coords_botleft[0], self.xy_coords_botleft[0] + self.shape[0],
            self.xy_coords_botleft[1], self.xy_coords_botleft[1] + self.shape[1],
        ), **imshow_kwargs)


    def resize(self, new_shape: Union[Union[float, int], Tuple[int]], interpolation=cv2.INTER_AREA):
        if isinstance(new_shape, (float, int)):
            new_shape = [int(new_shape), int(new_shape)]

        self.image = cv2.resize(self.image, (new_shape[1], new_shape[0]), interpolation=interpolation)
        self.shape = np.array(new_shape)
        return self.image
