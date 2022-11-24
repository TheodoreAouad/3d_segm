from importlib import reload
import random
from functools import reduce, partial
from os.path import join
from time import time
import os

import torch
import torch.nn as nn
# from deep_morpho.datasets.mnist_dataset import MnistClassifDataset
# from deep_morpho.models import LightningBiMoNNClassifier, BiSE, COBiSE, BiSEC, COBiSEC
import deep_morpho.binarization.bise_closest_selem as bcs
import deep_morpho.viz.bimonn_viz as bimonn_viz




import deep_morpho.models as models

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(device)

# tb_path = ("deep_morpho/results/DGMM_2022/sandbox/1/axspa_roi/version_7")
# tb_path = "/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/results_tensorboards/Bimonn_exp_56/sandbox/2/softplus/diskorect/opening/disk/version_1"
tb_path = "deep_morpho/results/results_tensorboards/Bimonn_exp_70/sandbox/3/bisel/softplus/mnist_gray/erosion_gray/disk/version_12"
# tb_path = "/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/results_tensorboards/test_refactor_lui/bisel2/softplus/diskorect/dilation/disk/version_0"

tb_path = join(tb_path, 'checkpoints', os.listdir(join(tb_path, 'checkpoints'))[0])

model = models.LightningBiMoNN.load_from_checkpoint(tb_path)
model.to(device)

loss = nn.BCELoss()

# model.model.layers[0].bises[0].closest_selem_handler = bcs.BiseClosestMinDistOnCst(bise_module=model.model.layers[0].bises[0])
model.model.layers[0].bises[0].closest_selem_handler = bcs.BiseClosestMinDistBounds(bise_module=model.model.layers[0].bises[0])

fig_weights = bimonn_viz.BimonnVizualiser(model.model, mode="closest").get_fig(figsize=(10, 10), axis='on')
fig_weights.axes[0].set_xticks([])
fig_weights.axes[0].set_yticks([])
