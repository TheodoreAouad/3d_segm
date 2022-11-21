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