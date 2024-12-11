from importlib import reload
from time import time
import random

from numba import njit
import numpy as np

import general.numba_utils as nu

for modl in [nu]:
    reload(modl)


def time_decorator(n_iter=int(1e6)):
    """
    Decorator to time a function.

    Args:
        n_iter: write your description
        int: write your description
    """
    def wrapper_fn(fn):
        """
        Wrap a function to print the time it took to execute it and print the return value.

        Args:
            fn: write your description
        """
        def wrapper(*args, **kwargs):
            """
            Wrapper function to print time and n_iter times

            Args:
            """
            t1 = time()
            fn(*args, **kwargs)
            print('first:', time() - t1)

            t1 = time()
            for _ in range(n_iter):
                res = fn(*args, **kwargs)
            print(time() - t1)
            return res
        return wrapper
    return wrapper_fn


@njit
def numba_sum(ar1, ar2):
    """
    Sums the two numbers using the numba algorithm.

    Args:
        ar1: write your description
        ar2: write your description
    """
    return ar1 + ar2


ar = np.zeros((50, 50))

@time_decorator(n_iter=int(1e6))
def f1(x):
    """
    First order function

    Args:
        x: write your description
    """
    numba_sum(ar, ar)

@time_decorator(n_iter=int(1e6))
def f2(x):
    """
    Exponential random variates.

    Args:
        x: write your description
    """
    return ar + ar


f1(1)
f2(1)
