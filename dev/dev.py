from time import time

from deep_morpho.datasets.generate_forms3 import get_random_rotated_diskorect

t1 = time()
for _ in range(1):
    get_random_rotated_diskorect(
        (50, 50), n_shapes = 30, max_shape = (15, 15), p_invert = 0.5,
            border=(4, 4), n_holes = 15, max_shape_holes = (5, 5), noise_proba=0.05
    )
print("first one:", time() - t1)

t1 = time()
for _ in range(20000):
    get_random_rotated_diskorect(
        (50, 50), n_shapes = 30, max_shape = (15, 15), p_invert = 0.5,
            border=(4, 4), n_holes = 15, max_shape_holes = (5, 5), noise_proba=0.05
    )
print(time() - t1)