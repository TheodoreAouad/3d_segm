import numpy as np
import torch.optim as optim
import torch.nn as nn

from deep_morpho.datasets.generate_forms3 import get_random_diskorect_channels
from deep_morpho.loss import MaskedMSELoss, MaskedDiceLoss, MaskedBCELoss, QuadraticBoundRegularization, LinearBoundRegularization
from general.utils import dict_cross
from deep_morpho.models import InitBiseEnum

loss_dict = {
    "MSELoss": nn.MSELoss,
    "BCELoss": nn.BCELoss,
}

all_args = {}

all_args['batch_seed'] = [None]

all_args['n_try'] = [0]
# all_args['n_try'] = range(1, 11)

all_args['experiment_name'] = [
    "Bimonn_exp_55/classif/sandbox/0"
    # "Bimonn_exp_55/classif/baseline"
]


# DATA ARGS
all_args['dataset_type'] = [
    "mnist",
]
all_args['n_inputs'] = [70_000]
all_args['preprocessing'] = [  # for axspa roi
    None,
]
all_args['mnist_args'] = [
    {"threshold": 30, "invert_input_proba": 0},
]
all_args['train_test_split'] = [(0.8, 0.2, 0)]


# TRAINING ARGS
all_args['learning_rate'] = [
    1e-2,
    # 1,
]

# if max_plus, then the loss is MSELoss
all_args['loss_data_str'] = [
    # nn.BCELoss(),
    "BCELoss",
    # "MaskedMSELoss",
]
all_args['loss_regu'] = [
    # ("quadratic", {"lower_bound": 0, "upper_bound": np.infty, "lambda_": 0.01})
    # "linear",
    "None",
]
all_args['optimizer'] = [
    optim.Adam,
    # optim.SGD
]
all_args['batch_size'] = [64]
all_args['num_workers'] = [
    20,
    # 0,
]
all_args['freq_imgs'] = [300]
all_args['n_epochs'] = [10]


# MODEL ARGS
# all_args['n_atoms'] = [
#     # 'adapt',
#     4,
# ]
all_args['atomic_element'] = [
    "bisel",
]
all_args['kernel_size'] = [
    3,
    # "adapt",
]
all_args['channels'] = [
    [1, 3, 3, 3, 1]
]
all_args['init_weight_mode'] = [
    # "identity",
    # "normal_identity",
    # "conv_0.5"
    InitBiseEnum.CUSTOM_HEURISTIC
    # InitBiseEnum.CUSTOM_CONSTANT
]
all_args['activation_P'] = [5]
all_args['force_lui_identity'] = [False]
all_args['constant_activation_P'] = [False]
all_args['constant_P_lui'] = [False]
all_args['constant_weight_P'] = [True]
all_args['init_bias_value_bise'] = [1]
all_args['init_bias_value_lui'] = [1]

all_args['threshold_mode'] = [
    # 'arctan',
    # 'sigmoid',
    {
        "weight": 'softplus',
        "activation": 'tanh',
    },
    # {
    #     # "weight": 'softplus',
    #     "weight": 'identity',
    #     # "weight": 'tanh',
    #     "activation": 'tanh',
    # },
    # 'erf',
    # "identity",
    # {"activation": "sigmoid", "weight": "identity", "complementation": "clamp"}
]
all_args["alpha_init"] = [0]

all_args['share_weights'] = [False]

all_args = dict_cross(all_args)
#
for idx, args in enumerate(all_args):

    args['n_atoms'] = len(args['channels']) - 1

    args['loss_data'] = loss_dict[args['loss_data_str']]()
    args['experiment_subname'] = f"{args['threshold_mode']['weight']}/{args['dataset_type']}"

    if args["init_weight_mode"] == InitBiseEnum.CUSTOM_CONSTANT:
        args['init_bias_value_bise'] = 'auto'
        args['init_bias_value_lui'] = 'auto'

    if args['mnist_args']['invert_input_proba'] == 1:
        args['experiment_subname'] = args['experiment_subname'].replace('mnist', 'inverted_mnist')

    args['loss'] = {"loss_data": args['loss_data']}

    if isinstance(args['threshold_mode'], str) or args['threshold_mode']['weight'] != "identity":
        args['loss_regu'] = "None"


    if args['loss_regu'] != "None":
        args['loss_regu'] = (loss_dict[args['loss_regu'][0]], args['loss_regu'][1])
        args['loss']['loss_regu'] = args['loss_regu']
