
import matplotlib.pyplot as plt

from general.structuring_elements import *


# for name, selem in zip(["hstick", "dcross", "bcomplex", "bsquare", "scross", "bdiamond"], [hstick, dcross, bcomplex, bsquare, scross, bdiamond]):
#     plt.imsave(f"todelete/{name}.png", selem(size=7))

plt.imsave("todelete/disk.png", disk(3))