import numpy as np
from skimage.morphology import disk
import torch.nn as nn
import torch.optim as optim

from deep_morpho.datasets.generate_forms3 import get_random_rotated_diskorect
from general.utils import dict_cross
from .args_morp_ops import morp_operations

args = {}

args['experiment_name'] = ['Bimonn_exp_20']


# DATA ARGS
args['morp_operation'] = morp_operations
args['random_gen_fn'] = [get_random_rotated_diskorect]
args['random_gen_args'] = [
    {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02}
]


# TRAINING ARGS
args['learning_rate'] = [
    # 1e-2,
    2.5e-3,
]
args['loss'] = [
    nn.BCELoss(),
    # nn.BCEWithLogitsLoss(),
]
args['optimizer'] = [optim.Adam]
args['batch_size'] = [32]
args['n_inputs'] = [
    500_000,
    # 1_000_000,
]


# MODEL ARGS
args['n_atoms'] = [
    'adapt',
]
args['atomic_element'] = [
    # 'bise',
    # 'bisec',
    'cobise',
    # 'cobisec',
]
args['kernel_size'] = [
    # 7
    "adapt",
]
args['init_weight_identity'] = [True]
args['activation_P'] = [1]
args['threshold_mode'] = [
    # 'arctan',
    # 'sigmoid',
    'tanh',
    # 'erf',
    # {"activation": "tanh", "weight": "tanh", "complementation": "clamp"}
]
args["alpha_init"] = [0]

args['share_weights'] = [False]
args['do_thresh_penalization'] = [False]
args['args_thresh_penalization'] = [{
    'coef': .005,
    'degree': 4,
    'detach_weights': True,
}]
args['first_batch_pen'] = [1]

all_args = dict_cross(args)
#
for idx, args in enumerate(all_args):
    if args["kernel_size"] == "adapt":
        args["kernel_size"] = args["morp_operation"].selems[0].shape[0]
    args["random_gen_args"]["border"] = (args["kernel_size"]//2 + 1, args["kernel_size"]//2 + 1)

    if args["n_atoms"] == 'adapt':
        args['n_atoms'] = len(args['morp_operation'])
        if args['atomic_element'] in ['cobise', 'cobisec']:
            args['n_atoms'] = args['n_atoms'] // 2

#     if "dilation" in args['morp_operation'].name:
#         all_args[idx]['n_inputs'] = 200_000
