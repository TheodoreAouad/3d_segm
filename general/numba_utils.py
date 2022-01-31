import random

import numpy as np
from numba import njit

@njit
def numba_randint(a: int, b: int):
    """
    Return a random integer between a and b.

    Args:
        a: write your description
        b: write your description
    """
    return random.randint(a, b)

@njit
def numba_rand():
    """
    Return a random number from the numba library.

    Args:
    """
    return random.random()

@njit
def numba_rand_shape_2d(x, y):
    """
    Random shape in 2d numpy array.

    Args:
        x: write your description
        y: write your description
    """
    return np.random.rand(x, y)

@njit
def numba_array(ar):
    """
    Convert a numpy array to a Numba array.

    Args:
        ar: write your description
    """
    return np.array(ar)

@njit
def numba_compare(ar1, ar2):
    """
    Compares two numbers using the numba library.

    Args:
        ar1: write your description
        ar2: write your description
    """
    return ar1 < ar2