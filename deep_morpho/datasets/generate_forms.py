import random
from typing import Tuple

import numpy as np


def random_rectangle(size: Tuple, width_props: Tuple[float] = (.1, .9), length_props: Tuple[float] = (.1, .9)) -> np.ndarray:
    rect = np.zeros(size)

    min_W, max_W = int(size[0] * width_props[0]), int(size[1] * width_props[1])
    W = int(random.uniform(min_W, max_W))

    min_L, max_L = int(size[0] * length_props[0]), int(size[1] * length_props[1])
    L = int(random.uniform(min_L, max_L))

    top_left_x = random.choice(range(size[0] - W))
    top_left_y = random.choice(range(size[1] - L))

    rect[top_left_x:top_left_x + W, top_left_y:top_left_y + L] = 1
    return rect
