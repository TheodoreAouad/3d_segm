import time

import vpython as visual
import numpy as np
import skimage.morphology as morp

from src.geometry.utils import get_barycenter_nonzero_3d

segm = np.load('./true_seg.npy') == 2
# segm = np.load('./test_segm.npy')
W, L, H = segm.shape


segm = (
    morp.binary_dilation(segm, morp.ball(1)).astype(int) -
    morp.binary_erosion(segm, morp.ball(1)).astype(int)
)

bary = get_barycenter_nonzero_3d(segm)

# for h in range(H):
#     segm[..., h] = (
#         morp.binary_dilation(segm[..., h], morp.disk(1)).astype(int) - 
#         morp.binary_erosion(segm[..., h], morp.disk(1)).astype(int)
#     )


start = time.time()

for i in range(W):
    for j in range(L):
        for k in range(H):
            if segm[i, j, k]:
                # ball=visual.box(pos=visual.vector(i,j,k) - visual.vector(*bary),radius=1,color=visual.color.green)
                ball=visual.sphere(pos=visual.vector(i,j,k),radius=1,color=visual.color.green)

print('Duration:', time.time() - start)

# for i in range(100):
#     ball=visual.sphere(pos=visual.vector(i,7,3),radius=2,color=visual.color.green)

