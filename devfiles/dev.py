from importlib import reload
import random
from functools import reduce, partial
from os.path import join
from time import time
import os
import copy

from tqdm.notebook import tqdm
import pandas as pd
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
import skimage.morphology as morp
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from PIL import Image, ImageDraw
import torchviz
# from deep_morpho.datasets.mnist_dataset import MnistClassifDataset
# from deep_morpho.models import LightningBiMoNNClassifier, BiSE, COBiSE, BiSEC, COBiSEC


import deep_morpho.models.softplus as sfp
import general.array_morphology as arm
import general.structuring_elements as selm
from general.nn.loss import DiceLoss
import general.nn.pytorch_lightning_module.obs_lightning_module as olm

from deep_morpho.save_results_template.display_results import DisplayResults
import deep_morpho.binarization.bise_closest_selem as bcs
import deep_morpho.binarization.projection_constant_set as pcs
import deep_morpho.binarization.projection_activated as pa
import deep_morpho.binarization as binarization

import deep_morpho.initializer.bise_initializer as bise_init
import deep_morpho.initializer.bisel_initializer as bisel_init
import deep_morpho.initializer.bimonn_initializer as bimonn_init
import deep_morpho.initializer as inits

import deep_morpho.loss as dmloss
import deep_morpho.threshold_fn as threshold_fn
import deep_morpho.morp_operations as morpop
import deep_morpho.models.threshold_layer as threshold_layer
import deep_morpho.models.weights_layer as weights_layer
import deep_morpho.models.complementation_layer as complementation_layer
import deep_morpho.models.pconv as pconv
import deep_morpho.models.lightning_pconv as lpconv
import deep_morpho.models.bise_base as bise_base
import deep_morpho.models.bise as bise
import deep_morpho.models.lui as lui
import deep_morpho.models.bisel as bisel
import deep_morpho.models.dilation_sum_layer as dis
import deep_morpho.models.bimonn as bimonn
import deep_morpho.models as models
import deep_morpho.models.lightning_bise as lbise
import deep_morpho.models.lightning_bimonn as lbimonn
import deep_morpho.datasets.generate_forms1 as gfo
import deep_morpho.datasets.generate_forms2 as gfo2
import deep_morpho.datasets.generate_forms3 as gfo3
import deep_morpho.datasets.axspa_roi_dataset as axd
import deep_morpho.datasets.diskorect_dataset as mrda
import deep_morpho.datasets.mnist_dataset as mnist_dataset
import deep_morpho.observables.plot_parameters as obs_weights
import deep_morpho.observables.plot_pred as obs_pred
import deep_morpho.observables.weight_histogram as weight_histogram
import deep_morpho.observables as obs
import general.nn.viz.plot_histogram as phist
import general.nn.viz.element_image as eltimage
import general.nn.viz.element_histogram as elthistogram
import deep_morpho.viz.element_bise as eltbise
import deep_morpho.viz.morp_operations_viz as mov
import deep_morpho.viz.elt_generator_bimonn as eltgenbi
import deep_morpho.viz.elt_generator_bimonn_forward_save as eltgenbifor
import deep_morpho.viz.elt_generator_bimonn_histogram as eltgenbihist
import deep_morpho.viz.bimonn_viz as bimonn_viz
import deep_morpho.loss.regularization_dist_cst as rdc
import deep_morpho.loss.regularization_dist_activated as rda


# def reload_modules():
#     for modl in [olm, bcs, binarization, bise_init, bisel_init, bimonn_init, inits, sfp, arm, dmloss, selm, threshold_fn, morpop,
#                  threshold_layer, weights_layer, complementation_layer, pconv, lpconv, bise_base, bise, lui, bisel,
#                  dis, bimonn, models, lbise, lbimonn, gfo, gfo2, gfo3, axd,
#                  mrda, mnist_dataset, obs_weights, obs_pred, obs,
#                 weight_histogram, phist, eltimage, elthistogram, eltbise, mov, eltgenbi, eltgenbifor, eltgenbihist, bimonn_viz]:
#         reload(modl)
        
# reload_modules()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(device)

# reload_modules()
# tb_path = ("deep_morpho/results/DGMM_2022/sandbox/1/axspa_roi/version_7")
# tb_path = "/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/results_tensorboards/Bimonn_exp_56/sandbox/2/softplus/diskorect/opening/disk/version_1"
# tb_path = "deep_morpho/results/results_tensorboards/Bimonn_exp_81/sandbox/0_/2/diskorect/bimonn/closing/hstick/version_0"
# tb_path = "/hdd/aouadt/these/projets/Bimonn_LBQNN2023/results/results_tensorboards/debug_noisti/10/noisti/bimonn/version_0"

tb_path = "deep_morpho/results/results_tensorboards/Bimonn_exp_80/sandbox/positive_weights/0_/5/mnistclassifchannel/BimonnDenseNotBinary/version_77"
# tb_path = "deep_morpho/results/results_tensorboards/Bimonn_exp_80/sandbox/positive_weights/0_/5/mnistclassifchannel/BimonnDenseNotBinary/version_37"
# tb_path = "deep_morpho/results/results_tensorboards/Bimonn_exp_80/ruche/positive_weights/0_/3/mnistclassifchannel/BimonnDenseNotBinary/version_5"
# tb_path = "deep_morpho/results/results_tensorboards/Bimonn_exp_80/sandbox/positive_weights/0_/0/mnistclassifchannel/BimonnDenseNotBinary/version_8"

# tb_path = "deep_morpho/results/results_tensorboards/test/5/mnistclassifchannel/bimonndense/version_0"

# tb_path = "deep_morpho/results/results_tensorboards/test/bisel/softplus/classif_mnist/version_8"
# tb_path = "deep_morpho/results/results_tensorboards/Bimonn_exp_75/sandbox/0/dual_bisel/softplus/sticks_noised/sticks_noised/version_19"
# tb_path = "deep_morpho/results/results_tensorboards/Bimonn_exp_71/sandbox/0/bisel/softplus/diskorect/dilation/disk/version_4"

# tb_path = "/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/results_tensorboards/test_refactor_lui/bisel2/softplus/diskorect/dilation/disk/version_0"


# df, global_args, changing_args = DisplayResults().get_df_from_tb_paths([tb_path])


# tb_path = join(tb_path, 'checkpoints', os.listdir(join(tb_path, 'checkpoints'))[0])
if os.path.exists(join(tb_path, 'best_weights')):
    tb_path = join(tb_path, 'best_weights', os.listdir(join(tb_path, 'best_weights'))[0])
elif os.path.exists(join(tb_path, 'checkpoints')):
    tb_path = join(tb_path, 'checkpoints', os.listdir(join(tb_path, 'checkpoints'))[0])

# bise_base.BiseBiasOptimEnum[df["bias_optim_mode"].iloc[0]]

loss = nn.BCELoss()
model = models.LightningBiMoNN.load_from_checkpoint(tb_path, loss=loss)
# model = models.LightningBiMoNNClassifierLastLinearNotBinary.load_from_checkpoint(tb_path, model_args=model_args, loss=None, learning_rate=None, optimizer=None,)
# model.to(device)
model.model.binary(True);
model.model.binary(False);

model.model.to(device)
img = torch.rand(100, 1, 28, 28)
img = img.to(device)
img = img.view(img.shape[0], -1)

lui1 = model.model.layers[0]

otp = lui1.forward_partial_binary(img)
