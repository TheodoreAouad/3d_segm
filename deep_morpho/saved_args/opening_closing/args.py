"""
This argument file is to test the bimonn, with and without logical, on the erosion and dilation on different
structuring elements. You can change the size and the threshold used.
"""
import torch.nn as nn
import torch.optim as optim
import torch

from general.structuring_elements import *
from general.array_morphology import SequentialMorpOperations
from deep_morpho.datasets.generate_forms3 import get_random_rotated_diskorect
from general.utils import dict_cross

############ ARGS ############
SE_sizes = [7]
threshold_modes = ['tanh']
##############################

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

args = {}

args['experiment_name'] = ['Bimonn_exp_19']

# DATA ARGS
args['morp_operation'] = []
for SE_size in SE_sizes:
    for se in [
        disk(SE_size//2), hstick(SE_size), vstick(SE_size),
        diagonal_cross(SE_size), straight_cross(SE_size), square(SE_size)
    ]:
        args['morp_operation'].append(SequentialMorpOperations(
            name= f"opening_size_{SE_size}x{SE_size}",
            selems=[se, se],
            operations=['erosion', 'dilation'],
            return_numpy_array=False,
            device=device
        ))
        args['morp_operation'].append(SequentialMorpOperations(
            name=f"closing_size_{SE_size}x{SE_size}",
            selems=[se, se],
            operations=['dilation', 'erosion'],
            return_numpy_array=False,
            device=device
        ))

args['random_gen_fn'] = [get_random_rotated_diskorect]
args['random_gen_args'] = [
    {'size': (50, 50), 'n_shapes': 30, 'max_shape': (15, 15), 'p_invert': 0.5, 'n_holes': 15, 'max_shape_holes': (7, 7)}
]

# TRAINING ARGS
args['learning_rate'] = [1e-2]
args['loss'] = [nn.BCELoss()]
args['optimizer'] = [optim.Adam]
args['batch_size'] = [32]
args['n_inputs'] = [3_000_000]

# MODEL ARGS
args['logical_not'] = [False]
args['kernel_size'] = ["adapt"]
args['init_weight_identity'] = [True]
args['activation_P'] = [1]
args['threshold_mode'] = threshold_modes
args["alpha_init"] = [0]

args['do_thresh_penalization'] = [False]
args['args_thresh_penalization'] = [{
    'coef': .005,
    'degree': 4,
    'detach_weights': True,
}]
args['first_batch_pen'] = [1]


all_args = dict_cross(args)

for idx, args in enumerate(all_args):
    if args["kernel_size"] == "adapt":
        args["kernel_size"] = args["morp_operation"].selems[0].shape[0]
    args["random_gen_args"]["border"] = (args["kernel_size"] // 2 + 1, args["kernel_size"] // 2 + 1)
