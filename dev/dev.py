
import random
from os.path import join
import os

import matplotlib.pyplot as plt
import cv2
import numpy as np

paths = "/hdd/aouadt/projets/nft/layers/pose1"
parts = sorted(os.listdir(paths))

all_paths = []
for part in parts:
    all_paths.append(
        join(paths, part, random.choice(os.listdir(join(paths, part))))
    )


all_imgs = [plt.imread(t) for t in all_paths if "png" in t]


final_img = np.zeros((300, 300, 3))

for img in all_imgs:
    img = cv2.resize(img, (300, 300))
    mask = img[..., -1] == 1
    final_img[mask, ...] = img[mask, :-1]
