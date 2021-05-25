# %% Morp Registration
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.morphology as morp
from scipy.ndimage import rotate

import general.utils as u
import general.array_morphology as am

def reload_modules():
    for modl in [u, am]:
        reload(modl)

reload_modules()


# %%
plt.close()
ar1 = np.zeros((100, 100))
ar1[40: 60, 40:60] = 1
ar1 = ar1.astype(int)

ar2 = np.zeros((100, 100))
ar2[44: 69, 44: 69] = 1
ar2 = rotate(ar2, 30) > .4
ar2 = ar2[30: 130, 30: 130]

# %%
fig, axs = plt.subplots(1, 2)
axs[0].imshow(ar1)
axs[1].imshow(ar2)
fig.show()

# %%

print(ar1.sum(), ar2.sum())

# %%
reload_modules()
plt.close()

selem = morp.disk(1)
bord1 = am.array_dilation(ar1, selem) - ar1
bord2 = am.array_dilation(ar2, selem) - ar2

print(bord1.sum(), bord2.sum())

# %%
fig, axs = plt.subplots(1, 2)
axs[0].imshow(bord1, interpolation='nearest')
axs[1].imshow(bord2, interpolation='nearest')
fig.show()