import numpy as np


A = np.arange(3 * 4 * 5).reshape(3, 4, 5)
B = np.ones((3, 4))

B @ A