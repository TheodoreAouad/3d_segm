import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from deep_morpho.datasets.generate_forms3 import get_random_rotated_diskorect, get_random_diskorect_channels
from general.utils import dict_cross
from general.nn.loss import DiceLoss
from .args_morp_ops import morp_operations

all_args = {}

all_args['experiment_name'] = ['Bimonn_exp_37_bis']


# DATA ARGS
all_args['morp_operation'] = morp_operations
all_args['dataset_type'] = [
    'diskorect',
    # 'axspa_roi',
]
all_args['preprocessing'] = [  # for axspa roi
    transforms.ToTensor(),
]
all_args['dataset_path'] = [
    # 'data/deep_morpho/dataset_0',
    'generate',
]
all_args['in_ram'] = [
    # False,
    True,
]
all_args['random_gen_fn'] = [
    # get_random_rotated_diskorect,
    get_random_diskorect_channels
]
all_args['random_gen_args'] = [
    {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02}
    # {'size': (50, 50), 'n_shapes': 30, 'max_shape': (15, 15), 'p_invert': 0.5, 'n_holes': 15, 'max_shape_holes': (7, 7)}

]
all_args['n_inputs'] = [
    1_000_000,
    # 100_000,
]


# TRAINING ARGS
all_args['learning_rate'] = [
    1e-2,
    # 1,
]

# if max_plus, then the loss is MSELoss
all_args['loss'] = [
    nn.BCELoss(),
    # nn.BCEWithLogitsLoss(),
    # nn.MSELoss(),
    # DiceLoss(),
]
all_args['optimizer'] = [
    optim.Adam,
    # optim.SGD
]
all_args['batch_size'] = [256]
all_args['num_workers'] = [
    20,
    # 0,
]
all_args['freq_imgs'] = [300]
all_args['n_epochs'] = [10]


# MODEL ARGS
all_args['n_atoms'] = [
    # 'adapt',
    4,
]
all_args['atomic_element'] = [
    # 'conv',
    # 'bise',
    "bisel",
    # 'bisec',
    # 'cobise',
    # 'cobisec',
    # "max_plus",
]
all_args['kernel_size'] = [
    7,
    # "adapt",
]
all_args['channels'] = [
    'adapt'
]
all_args['init_weight_identity'] = [True]
all_args['activation_P'] = [1]
all_args['constant_activation_P'] = [False]
all_args['constant_weight_P'] = [True]
all_args['threshold_mode'] = [
    # 'arctan',
    # 'sigmoid',
    'tanh',
    # 'erf',
    # "identity",
    # {"activation": "sigmoid", "weight": "identity", "complementation": "clamp"}
]
all_args["alpha_init"] = [0]

all_args['share_weights'] = [False]
all_args['do_thresh_penalization'] = [False]
all_args['args_thresh_penalization'] = [{
    'coef': .005,
    'degree': 4,
    'detach_weights': True,
}]
all_args['first_batch_pen'] = [1]

all_args = dict_cross(all_args)
#
for idx, args in enumerate(all_args):

    if args['dataset_type'] == "axspa_roi":
        args['dataset_path'] = "data/deep_morpho/axspa_roi/axspa_roi.csv"
        args['morp_operation'] = []
        args['experiment_subname'] = 'axspa_roi'
        args['freq_imgs'] = 10

    elif args['dataset_type'] == "diskorect":
        args['kernel_size'] = 'adapt'
        args['n_atoms'] = 'adapt'
        args['n_epochs'] = 1


        if args["kernel_size"] == "adapt":
            args["kernel_size"] = args["morp_operation"].selems[0][0][0].shape[0]

        args["random_gen_args"] = args["random_gen_args"].copy()
        args["random_gen_args"]["border"] = (args["kernel_size"]//2 + 1, args["kernel_size"]//2 + 1)
        args['random_gen_args']['size'] = args['random_gen_args']['size'] + (args["morp_operation"].in_channels[0],)
        args['experiment_subname'] = args['morp_operation'].name

        if args['channels'] == 'adapt':
            args['channels'] = args['morp_operation'].in_channels + [args['morp_operation'].out_channels[-1]]

        if args["n_atoms"] == 'adapt':
            args['n_atoms'] = len(args['morp_operation'])
            if args['atomic_element'] in ['cobise', 'cobisec']:
                args['n_atoms'] = max(args['n_atoms'] // 2, 1)

    if args['atomic_element'] == "conv":
        args['threshold_mode'] = {"activation": "sigmoid", "weight": "identity"}

    if args['atomic_element'] == "max_plus":
        args['loss'] = nn.MSELoss()
