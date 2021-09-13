from importlib import reload
from time import time
import random

from numba import njit
import numpy as np

import general.numba_utils as nu

for modl in [nu]:
    reload(modl)


def time_decorator(n_iter=int(1e6)):
    def wrapper_fn(fn):
        def wrapper(*args, **kwargs):
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
    return ar1 + ar2


ar = np.zeros((50, 50))

@time_decorator(n_iter=int(1e6))
def f1(x):
    numba_sum(ar, ar)

@time_decorator(n_iter=int(1e6))
def f2(x):
    return ar + ar


f1(1)
f2(1)
