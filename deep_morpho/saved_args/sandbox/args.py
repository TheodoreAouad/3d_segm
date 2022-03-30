import numpy as np
import torch.optim as optim

from deep_morpho.datasets.generate_forms3 import get_random_diskorect_channels
from deep_morpho.loss import MaskedMSELoss, MaskedDiceLoss, QuadraticBoundRegularization, LinearBoundRegularization
from general.utils import dict_cross
from .args_morp_ops import morp_operations

loss_dict = {
    "MaskedMSELoss": MaskedMSELoss,
    "MaskedDiceLoss": MaskedDiceLoss,
    "quadratic": QuadraticBoundRegularization,
    "linear": LinearBoundRegularization,
}

all_args = {}

# all_args['n_try'] = [0]
all_args['n_try'] = range(1, 11)

all_args['experiment_name'] = [
    # 'Bimonn_exp_45_sandbox_disk'
    # 'Bimonn_exp_46/multi_closing'
    # 'Bimonn_exp_46_sandbox'
    # 'Bimonn_exp_48',
    # "Bimonn_exp_49/sandbox"
    "Bimonn_exp_51/sandbox/multi/1"
    # "Bimonn_exp_51/sandbox/5"
    # "Bimonn_exp_52/sandbox/0"
]


# DATA ARGS
all_args['morp_operation'] = morp_operations
all_args['dataset_type'] = [
    # 'axspa_roi',
    "mnist",
    # 'diskorect',
]
all_args['preprocessing'] = [  # for axspa roi
    None,
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
    get_random_diskorect_channels
]
all_args['random_gen_args'] = [
    {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02}

]
all_args['mnist_args'] = [
    {"threshold": 30, "size": (50, 50), "invert_input_proba": 0}
]
all_args['n_inputs'] = [
    # 3_000_000,
    200_000,
]
all_args['train_test_split'] = [(0.8, 0.2, 0)]


# TRAINING ARGS
all_args['learning_rate'] = [
    1e-2,
    # 1,
]

# if max_plus, then the loss is MSELoss
all_args['loss_data'] = [
    # nn.BCELoss(),
    # nn.BCEWithLogitsLoss(),
    "MaskedMSELoss",
    # "MaskedDiceLoss",
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
all_args['batch_size'] = [256]
all_args['num_workers'] = [
    20,
    # 0,
]
all_args['freq_imgs'] = [300]
all_args['n_epochs'] = [20]


# MODEL ARGS
# all_args['n_atoms'] = [
#     # 'adapt',
#     4,
# ]
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
    # 41,
    "adapt",
]
all_args['channels'] = [
    'adapt',
    # [
    #     # 1,  # input
    #     2, 1,
    # ]
]
all_args['init_weight_mode'] = [
    # "identity",
    # "normal_identity",
    "conv_0.5"
]
all_args['activation_P'] = [0]
all_args['force_lui_identity'] = [False]
all_args['constant_activation_P'] = [False]
all_args['constant_P_lui'] = [False]
all_args['constant_weight_P'] = [True]
all_args['threshold_mode'] = [
    # 'arctan',
    # 'sigmoid',
    {
        "weight": 'identity',
        "activation": 'tanh',
    }
    # 'erf',
    # "identity",
    # {"activation": "sigmoid", "weight": "identity", "complementation": "clamp"}
]
all_args["alpha_init"] = [0]

all_args['share_weights'] = [False]

all_args = dict_cross(all_args)
#
for idx, args in enumerate(all_args):

    if args['dataset_type'] == "axspa_roi":
        args['dataset_path'] = "data/deep_morpho/axspa_roi/axspa_roi.csv"
        args['morp_operation'] = []
        args['experiment_subname'] = 'axspa_roi'
        args['freq_imgs'] = 10
        args['n_atoms'] = len(args['channels']) - 1
        args['loss_data'] = loss_dict[args['loss_data']](border=(0, 0))

    if args['dataset_type'] == "mnist":
        args['freq_imgs'] = 300


    if args['dataset_type'] in ["diskorect", 'mnist']:
        # args['kernel_size'] = 'adapt'
        args['n_atoms'] = 'adapt'


        if args["kernel_size"] == "adapt":
            # args["kernel_size"] = args["morp_operation"].selems[0][0][0].shape[0]
            args["kernel_size"] = int(max(args['morp_operation'].max_selem_shape))

        args['loss_data'] = loss_dict[args['loss_data']](border=np.array([args['kernel_size'] // 2, args['kernel_size'] // 2]))
        args['experiment_subname'] = f"{args['dataset_type']}/{args['morp_operation'].name}"

        if args['channels'] == 'adapt':
            args['channels'] = args['morp_operation'].in_channels + [args['morp_operation'].out_channels[-1]]

        if args["n_atoms"] == 'adapt':
            args['n_atoms'] = len(args['morp_operation'])
            if args['atomic_element'] in ['cobise', 'cobisec']:
                args['n_atoms'] = max(args['n_atoms'] // 2, 1)


    if args['dataset_type'] == "diskorect":
        args['n_epochs'] = 1
        args["random_gen_args"] = args["random_gen_args"].copy()
        args["random_gen_args"]["border"] = (args["kernel_size"]//2 + 1, args["kernel_size"]//2 + 1)
        args['random_gen_args']['size'] = args['random_gen_args']['size'] + (args["morp_operation"].in_channels[0],)

    if args['dataset_type'] == "mnist":
        args['n_inputs'] = 70_000


    if args['atomic_element'] == "conv":
        args['threshold_mode'] = {"activation": "sigmoid", "weight": "identity"}

    if args['atomic_element'] == "max_plus":
        args['loss'] = MaskedMSELoss()

    args['loss'] = {"loss_data": args['loss_data']}

    if isinstance(args['threshold_mode'], str) or args['threshold_mode']['weight'] != "identity":
        args['loss_regu'] = "None"


    if args['loss_regu'] != "None":
        args['loss_regu'] = (loss_dict[args['loss_regu'][0]], args['loss_regu'][1])
        args['loss']['loss_regu'] = args['loss_regu']
