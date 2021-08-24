from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw


def get_rect(x, y, width, height, angle):
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    theta = (np.pi / 180.0) * angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect


def draw_poly(draw, poly):
    draw.polygon([tuple(p) for p in poly], fill=1)


def draw_ellipse(draw, center, radius):
    bbox = (center[0] - radius[0], center[1] - radius[1], center[0] + radius[0], center[1] + radius[1])
    draw.ellipse(bbox, fill=1)


def get_random_rotated_diskorect(
    size: Tuple, n_shapes: int = 30, max_shape: Tuple[int] = (15, 15), p_invert: float = 0.5, **kwargs
):
    diskorect = np.zeros(size)
    img = Image.fromarray(diskorect)
    draw = ImageDraw.Draw(img)

    for _ in range(n_shapes):

        x = np.random.randint(0, size[0] - 2)
        y = np.random.randint(0, size[0] - 2)

        if np.random.rand() < .5:
            W = np.random.randint(1, max_shape[0])
            L = np.random.randint(1, max_shape[1])

            angle = np.random.rand() * 45
            draw_poly(draw, get_rect(x, y, W, L, angle))

        else:
            rx = np.random.randint(1, max_shape[0]//2)
            ry = np.random.randint(1, max_shape[1]//2)
            draw_ellipse(draw, np.array([x, y]), (rx, ry))

    diskorect = np.asarray(img)
    if np.random.rand() < p_invert:
        diskorect = 1 - diskorect

    return diskorect
