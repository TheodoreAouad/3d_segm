## TEST REGISTRATION

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes

import ssm.utils as su

rec1 = np.concatenate((
    np.hstack(
        (np.ones((50, 1)), np.linspace(1, 2, 50)[:, np.newaxis]),
    ),
    np.hstack(
        (np.linspace(1, 2, 50)[:, np.newaxis], np.ones((50, 1))),
    ),
    np.hstack(
        (1 + np.ones((50, 1)), np.linspace(1, 2, 50)[:, np.newaxis]),
    ),
    np.hstack(
        (np.linspace(1, 2, 50)[:, np.newaxis], 1 + np.ones((50, 1))),
    ),
))
rec1 -= rec1.mean(0)

def rot_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

T = np.eye(3)
R = rot_matrix(np.pi / 4)
scale_ini = 3
T[:2, :2] = R / scale_ini
# T[:-1, -1] = np.array([3, 3])

rec2 = su.transform_cloud(T, rec1)

U, S, Vt = np.linalg.svd(rec1.T @ rec2)
Ropt = Vt.T @ U.T
scale = S.sum() / (np.linalg.norm(rec1) ** 2)

Ropt_scipy, scale_scipy = orthogonal_procrustes(rec1, rec2)

print(scale_scipy)

fig = plt.figure(figsize=(21, 7))
ax = fig.add_subplot(131)
ax.scatter(*rec1.T)
ax.scatter(*rec2.T)
ax.axis('equal')

ax = fig.add_subplot(132)
ax.scatter(*((Ropt * scale) @ rec1.T))
ax.scatter(*rec2.T)
ax.axis('equal')

ax = fig.add_subplot(133)
ax.scatter(*((Ropt_scipy * scale_scipy) @ rec1.T))
ax.scatter(*rec2.T)
ax.axis('equal')

fig.show()
