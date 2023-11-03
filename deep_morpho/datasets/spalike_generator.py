from typing import Tuple
from time import time
import random
from abc import ABC, abstractmethod

import numpy as np

from scipy.ndimage import convolve
import skimage.morphology as morp
import cv2


def dilate_cv(image, selem):
    origin_type = image.dtype
    return cv2.dilate(image.astype(np.uint8), selem[::-1, ::-1].astype(np.uint8)).astype(origin_type)


def erode_cv(image, selem):
    origin_type = image.dtype
    return cv2.erode(image.astype(np.uint8), selem[::-1, ::-1].astype(np.uint8)).astype(origin_type)


binary_erosion = (
    # morp.binary_erosion
    erode_cv
)
binary_dilation = (
    # morp.binary_dilation
    dilate_cv
)


class Texture(ABC):
    @abstractmethod
    def draw(self, img):
        pass


class GridEllipse(Texture):

    def __init__(
        self,
        grid_spacing: tuple = (24, 24),
        min_ellipse_axes: int = 13,
        max_ellipse_axes: int = 35,
        period: tuple = (3, 10),
        offset: tuple = (1, 2),
        min_output_ellipse: int = 0,
    ):
        self.grid_spacing = self.tuplify(grid_spacing)

        self.min_ellipse_axes = min_ellipse_axes
        self.max_ellipse_axes = max_ellipse_axes

        self.period = self.tuplify(period)
        self.offset = self.tuplify(offset)

        self.sin_grid_min = 3
        self.min_output_ellipse = min_output_ellipse

    @property
    def sin_grid_max(self):
        return self.sin_grid_min + 4

    @staticmethod
    def tuplify(x):
        if isinstance(x, int):
            return (x, x)
        else:
            return x

    def draw(self, img: np.ndarray) -> np.ndarray:
        # Randomly generate ellipses with different shapes and colors in a grid
        img = img.copy()
        # img = img + 0
        Xs = np.linspace(0, img.shape[0], num=int(img.shape[0] / self.grid_spacing[0]) + 1, dtype=np.int32)
        Ys = np.linspace(0, img.shape[1], num=int(img.shape[1] / self.grid_spacing[1]) + 1, dtype=np.int32)

        for idx_x, center_x in enumerate(Xs):
            for idx_y, center_y in enumerate(Ys):
                ellipse_color = self.normalize(self.sin_grid(idx_x + 1, idx_y + 1, ))
                img = self.draw_ellipse(img, center_x, center_y, self.min_ellipse_axes, self.max_ellipse_axes, ellipse_color)
                # ellipse_axes = (random.randint(self.min_ellipse_axes, self.max_ellipse_axes), random.randint(self.min_ellipse_axes, self.max_ellipse_axes))
                # angle = random.uniform(0, 360)  # Random rotation angle
                # # Draw the random ellipse
                # cv2.ellipse(img, (center_x, center_y), ellipse_axes, angle, 0, 360, ellipse_color, -1)

        return img

    @staticmethod
    def draw_ellipse(img: np.ndarray, center_x, center_y, min_ellipse_axes: int, max_ellipse_axes: int, color: float) -> np.ndarray:
        # img = img + 0
        img = img.copy()
        ellipse_axes = (random.randrange(min_ellipse_axes, max_ellipse_axes+1), random.randrange(min_ellipse_axes, max_ellipse_axes+1))
        angle = random.uniform(0, 360)
        cv2.ellipse(img, (center_x, center_y), ellipse_axes, angle, 0, 360, color, -1)
        return img

    # sinusoidal grid
    def sin_grid(self, x: float, y: float, ):
        return (
            (self.sin_period(x - self.offset[0], self.period[0]) + 1) * (self.sin_period(y - self.offset[1], self.period[1]) + 1)
        ) + self.sin_grid_min

    def normalize(self, x, ):
        return (x - self.sin_grid_min) / (self.sin_grid_max - self.sin_grid_min) * (255 - self.min_output_ellipse) + self.min_output_ellipse

    @staticmethod
    def sin_period(x, period):
        return (np.sin(x * 2 * np.pi / period) + 1) / 2




class BonesLike:

    def __init__(
        self,
    ):
        self.img = None

        self.iliac_coordinates = None
        self.sacrum_coordinates = None
        self.iliac_triangle1 = None
        self.iliac_triangle2 = None

        self._iliac_segm_value = 1
        self._sacrum_segm_value = 2

        self.segmentation = None

    @property
    def iliac_segmentation(self) -> np.ndarray:
        return self.segmentation == self._iliac_segm_value

    @property
    def sacrum_segmentation(self) -> np.ndarray:
        return self.segmentation == self._sacrum_segm_value

    @property
    def iliac_bbox1(self) -> np.ndarray:
        """Return bounding box for first part of the iliac bone."""
        return self.iliac_triangle1.min(axis=0), self.iliac_triangle1.max(axis=0)

    def get_sacrum_value(self):
        return np.random.randint(130, 170)

    def get_iliac_value(self):
        return np.random.randint(130, 170)

    def draw_bones(self, img: np.ndarray) -> np.ndarray:
        img = img + 0
        W_, L_ = img.shape
        size_sacrum = (int(.3 * W_), int(.3 * L_))
        w_, l_ = size_sacrum


        self.segmentation = np.zeros(img.shape).astype(np.uint8)
        self.segmentation[
            W_//2 - w_//2:W_//2 + w_//2,
            -l_:
        ] = self._sacrum_segm_value

        self.sacrum_coordinates = [
            np.array([(W_ - w_) // 2, L_ - l_]), np.array([(W_ + w_) // 2, L_ - 1])
        ]

        img[self.sacrum_segmentation] = self.get_sacrum_value()

        self.iliac_triangle1, self.iliac_triangle2 = self.get_triangle(W_, w_, L_, l_)

        cv2.fillPoly(self.segmentation, [self.iliac_triangle1, self.iliac_triangle2], self._iliac_segm_value)

        img[self.iliac_segmentation] = self.get_iliac_value()

        self.iliac_coordinates = [self.iliac_triangle1, self.iliac_triangle2]

        return img

    def get_triangle(self, W_: int, w_: int, L_: int, l_: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            W_: image width
            w_: sacrum width
            L_: image height
            l_: sacrum height
        """
        p1 = [
            np.random.randint((W_ - w_) / 2 - .5*w_, (W_ - w_) / 2 - .1*w_),
            L_ - l_
        ]
        p2 = [
            np.random.randint((W_ - w_) / 2 + w_, (W_ - w_) / 2 + 1.1*w_),
            L_ - l_
        ]
        p3 = [
            np.random.randint(p1[0], self.sacrum_coordinates[0][0]),
            np.random.randint(p1[1] - .5 * w_, p1[1] - .3*w_)
        ]

        triangle1 = np.array([
            p1[::-1], p2[::-1], p3[::-1]
        ], np.int32)

        p4 = [
            np.random.randint(.6 * p1[0], .9 * p1[0]),
            np.random.randint(p3[1] - .7 * (p2[0] - p1[0]), p3[1] - .3 * (p2[0] - p1[0]))
        ]
        triangle2 = np.array([
            p1[::-1], p4[::-1], p3[::-1]
        ], np.int32)

        return triangle1, triangle2


    def draw_lesion(self, img):
        pass

    def draw(self, img):
        img = self.draw_bones(img)
        # img = self.draw_lesion(img)
        return img



class Blob:

    def __init__(self, big_ax, small_ax, angle, value):
        self.big_ax = max(big_ax, small_ax)
        self.small_ax = min(big_ax, small_ax)
        self.angle = angle
        self.value = value

        self.segmentation = None
        self.frame = None

        self.get_blob_frame()

    def get_empty_frame(self):
        # L_ = max(self.big_ax, self.small_ax)
        return np.zeros((
            2*self.big_ax+3, 2*self.big_ax+3
        ))

    def draw(self, img, center_x, center_y):
        img = img.copy()
        # img = img + 0
        segm = np.zeros(img.shape, dtype=np.uint8)
        cv2.ellipse(
            segm,
            (center_x, center_y),
            (self.big_ax, self.small_ax),
            self.angle,
            0,
            360,
            1,
            -1
        )
        self.segmentation = segm.astype(bool)
        img[self.segmentation] = self.value
        return img

    def get_blob_frame(self):
        self.frame = self.get_empty_frame()
        self.frame = self.draw(self.frame, self.frame.shape[1] // 2, self.frame.shape[0] // 2)

        return self.frame



class SpaLike:

    def __init__(
        self,
        image_size: tuple = (256, 256),
        proba_lesion: float = 0.5,
        proba_lesion_locations: dict = {
            "sacrum": 0.2,
            "iliac": 0.2,
        },
        grid_spacing: tuple = (16, 16),
        min_ellipse_axes: int = 13,
        max_ellipse_axes: int = 18,
        period: tuple = (3, 10),
        offset: tuple = (1, 2),
        min_output_ellipse: int = 0,
        max_n_blob_sane: int = 5,
        iliac_dil_coef: float = 2,
        sacrum_dil_coef: float = 2,
    ):
        self.image_size = image_size
        self.bone_generator = BonesLike()
        self.texture_generator = GridEllipse(
            grid_spacing=grid_spacing,
            min_ellipse_axes=min_ellipse_axes,
            max_ellipse_axes=max_ellipse_axes,
            period=period,
            offset=offset,
            min_output_ellipse=min_output_ellipse,
        )

        self.roi = None
        self.label = False
        self.proba_lesion = proba_lesion
        self.proba_lesion_locations = proba_lesion_locations
        self.max_n_blob_sane = max_n_blob_sane
        self.iliac_dil_coef = iliac_dil_coef
        self.sacrum_dil_coef = sacrum_dil_coef

        self.blobs = []

    @property
    def bone_segmentation(self):
        return self.bone_generator.segmentation

    @property
    def sacrum_segmentation(self):
        return self.bone_generator.sacrum_segmentation

    @property
    def iliac_segmentation(self):
        return self.bone_generator.iliac_segmentation

    def get_value_blob(self):
        return np.random.randint(200, 240)

    def generate_image(self):
        img = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.uint8)
        img = self.texture_generator.draw(img)
        img = self.bone_generator.draw(img)
        img = convolve(img, np.ones((3, 3)))
        # fimg = convolve(img.astype(float), np.ones((3, 3)))
        # segm = self.bone_generator.segmentation

        self.roi = self.get_roi(self.iliac_segmentation, self.sacrum_segmentation)

        self.segmentation_lesion = np.zeros(img.shape, dtype=bool)

        n_blob = np.random.randint(1, self.max_n_blob_sane + 1)
        if np.random.rand() < self.proba_lesion:
            self.label = True
            # n_blob -= 1
            img, blob = self.draw_blob(img=img, value=self.get_value_blob(), zone_str="roi")
            self.blobs.append(blob)
            self.segmentation_lesion[blob.segmentation] = True

        for _ in range(n_blob):
            rand_value = np.random.rand()
            if rand_value < self.proba_lesion_locations["sacrum"]:
                zone = "sacrum"
            elif rand_value < self.proba_lesion_locations["sacrum"] + self.proba_lesion_locations["iliac"]:
                zone = "iliac"
            else:
                zone = "other"
            img, blob = self.draw_blob(img=img, value=self.get_value_blob(), zone_str=zone)
            self.blobs.append(blob)
            self.segmentation_lesion[blob.segmentation] = True

        return img, self.label


    def draw_blob(self, img: np.ndarray, value: int, zone_str: str, size: Tuple[int, int] = (10, 10), n_try: int = 1):
        blob = Blob(
            big_ax=random.randint(2, size[0]),
            small_ax=random.randint(2, size[1]),
            angle=random.randint(0, 360),
            value=value,
        )

        if zone_str == "roi":
            zone = self.roi
        elif zone_str == "sacrum":
            zone = self.sacrum_segmentation & ~self.roi
        elif zone_str == "iliac":
            zone = self.iliac_segmentation & ~self.roi
        else:
            zone = self.background_segmentation

        zone = binary_erosion(zone, blob.segmentation)
        if zone.sum() == 0:
            return self.draw_blob(img=img, size=(int(size[0] // 1.5), int(size[1] // 1.5)), value=value, zone_str=zone_str, n_try=n_try + 1)

        def get_coordinates():
            Xs, Ys = np.where(zone)
            idx = np.random.randint(0, len(Xs))
            center_x, center_y = Xs[idx], Ys[idx]
            return center_x, center_y

        center_y, center_x = get_coordinates()

        img = blob.draw(img, center_x, center_y)
        # if n_try > 1:
        #     print(f"n_try: {n_try}")
        return img, blob

    @property
    def background_segmentation(self):
        return ~self.roi & ~self.sacrum_segmentation & ~self.iliac_segmentation



    def get_roi(self, segm_ili: np.ndarray, segm_sac: np.ndarray):
        (xmin, ymin), (xmax, ymax) = self.bone_generator.iliac_bbox1

        selem_sac = np.zeros(
            (1, int(self.sacrum_dil_coef*(xmax - xmin)) + 1)
        )
        selem_sac[0, :selem_sac.shape[1] // 2] = 1

        selem_ili = np.zeros(
            (1, int(self.iliac_dil_coef*(xmax - xmin)) + 1)
        )
        selem_ili[0, selem_ili.shape[1] // 2:] = 1

        dil_sac = binary_dilation(segm_sac, selem_sac)
        dil_ili = binary_dilation(segm_ili, selem_ili)

        ili_roi = dil_sac & segm_ili
        ili_roi = binary_dilation(ili_roi, morp.disk(5)) & segm_ili

        sac_roi = dil_ili & segm_sac

        final_roi = ili_roi | sac_roi
        return final_roi
