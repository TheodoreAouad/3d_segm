import warnings
import numpy as np
import torch.optim as optim
import torch.nn as nn

from deep_morpho.datasets.generate_forms3 import get_random_diskorect_channels
from deep_morpho.loss import (
    MaskedMSELoss, MaskedDiceLoss, MaskedBCELoss, QuadraticBoundRegularization, LinearBoundRegularization, 
    MaskedBCENormalizedLoss, MaskedNormalizedDiceLoss
)
from general.utils import dict_cross
from deep_morpho.models.bise_base import ClosestSelemEnum, ClosestSelemDistanceEnum, BiseBiasOptimEnum, BiseWeightsOptimEnum
from deep_morpho.initializer import InitBimonnEnum, InitBiseEnum
from .args_morp_ops import morp_operations

loss_dict = {
    "MaskedMSELoss": MaskedMSELoss,
    "MaskedDiceLoss": MaskedDiceLoss,
    "MaskedBCELoss": MaskedBCELoss,
    "MaskedBCENormalizedLoss": MaskedBCENormalizedLoss,
    "quadratic": QuadraticBoundRegularization,
    "linear": LinearBoundRegularization,
    "MaskedNormalizedDiceLoss": MaskedNormalizedDiceLoss,
    "MSELoss": nn.MSELoss,
}

all_args = {}

# all_args['batch_seed'] = [2249939862]
all_args['batch_seed'] = [None]

all_args['n_try'] = [0, 1, 2, 3, 4]
# all_args['n_try'] = range(1, 20)

all_args['experiment_name'] = [
    # "Bimonn_exp_59/sandbox/1"
    # "Bimonn_exp_60/sandbox/0"
    "Bimonn_exp_63/sandbox",
    # "Bimonn_exp_63/multi"
    # "Bimonn_exp_63/multi/0"
    # "Bimonn_mega_multi_1/sandbox/0"
    # "Bimonn_mega_multi_1/"
    # "test_new_bias"
    # "Bimonn_reprod"
    # "debug"
    # "sandbox"
    # "Bimonn_exp_62/sandbox/0"
    # "test_refactor_bise_old2"
    # "sybisel_debug"
]


# DATA ARGS
all_args['morp_operation'] = morp_operations
all_args['dataset_type'] = [
    # 'axspa_roi',
    # "mnist",
    # "inverted_mnist",
    'diskorect',
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
    # {'size': (50, 50), 'n_shapes': 2, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02}
    {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02, "border": (0, 0)}
    # {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02,}
    # {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02, "border": (0, 0)}
    # {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 1, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02, "border": (0, 0)}
]
all_args['mnist_args'] = [
    {"threshold": 30, "size": (50, 50), "invert_input_proba": 0, },
    # {"threshold": 30, "size": (50, 50), "invert_input_proba": 1},
]
all_args['n_steps'] = [1000]
all_args['nb_batch_indep'] = [0]
# all_args['n_inputs'] = [
#     3_000_000,
#     # 100_000,
# ]
all_args['train_test_split'] = [(0.8, 0.2, 0)]


# TRAINING ARGS
all_args['learning_rate'] = [
    1e-1,
    # 1e-2,
    # 1,
]

# if max_plus, then the loss is MSELoss
all_args['loss_data_str'] = [
    # nn.BCELoss(),
    # "MaskedBCENormalizedLoss",
    # "MaskedNormalizedDiceLoss",
    # "MaskedBCELoss",
    # "MSELoss",
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
all_args['patience_loss'] = [2100]
all_args['patience_reduce_lr'] = [700]


# MODEL ARGS
# all_args['n_atoms'] = [
#     # 'adapt',
#     4,
# ]
all_args['atomic_element'] = [
    # "bisel",
    "sybisel",
]
all_args['n_atoms'] = [
    'adapt',
    # 2
]

all_args['kernel_size'] = [
    # 7,
    "adapt",
    # 7 + 4,
]
all_args['channels'] = [
    'adapt',
    # [1, 1, 1]
    # [
    #     2, 2, 2, 2, 2, 2, 1
    # ]
    # [
    #     # 1,  # input
    #     2, 2, 1,
    # ]
]
# all_args['init_weight_mode'] = [
#     # "identity",
#     # "normal_identity",
#     # "conv_0.5"
#     # InitBiseEnum.KAIMING_UNIFORM,
#     # InitBiseEnum.CUSTOM_HEURISTIC,
#     InitBiseEnum.CUSTOM_CONSTANT
# ]
all_args['closest_selem_method'] = [
    ClosestSelemEnum.MIN_DIST
    # ClosestSelemEnum.MAX_SECOND_DERIVATIVE
]
all_args['closest_selem_distance_fn'] = [
    # ClosestSelemDistanceEnum.DISTANCE_BETWEEN_BOUNDS
    # ClosestSelemDistanceEnum.DISTANCE_TO_AND_BETWEEN_BOUNDS
    ClosestSelemDistanceEnum.DISTANCE_TO_BOUNDS
]
all_args['bias_optim_mode'] = [
    BiseBiasOptimEnum.RAW,
    # BiseBiasOptimEnum.POSITIVE,
    # BiseBiasOptimEnum.POSITIVE_INTERVAL_PROJECTED,
    # BiseBiasOptimEnum.POSITIVE_INTERVAL_REPARAMETRIZED
]
all_args['bias_optim_args'] = [
    {"offset": 0}
]
all_args['weights_optim_mode'] = [
    BiseWeightsOptimEnum.THRESHOLDED,
    # BiseWeightsOptimEnum.NORMALIZED
]
all_args['weights_optim_args'] = [
    # {"constant_P": True}
    {"constant_P": True, "factor": 1}
]

all_args['initializer_method'] = [
    # InitBiseEnum.KAIMING_UNIFORM,
    # InitBiseEnum.CUSTOM_HEURISTIC,
    # InitBiseEnum.CUSTOM_CONSTANT,
    # InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
    InitBimonnEnum.INPUT_MEAN,
]
all_args['initializer_args'] = [
    {
        # "bise_init_method": InitBiseEnum.CUSTOM_CONSTANT,
        "bise_init_method": InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
        "bise_init_args": {"init_bias_value": 0, "mean_weight": "auto", "ub": 0.01}
    }
]

all_args['activation_P'] = [0]
all_args['constant_activation_P'] = [False]
all_args['force_lui_identity'] = [False]
all_args['constant_P_lui'] = [False]

all_args['threshold_mode'] = [
    {
        "weight": 'softplus',
        "activation": 'tanh',
    },
]


all_args = dict_cross(all_args)
#

to_remove = []
for idx, args in enumerate(all_args):

    # Duality training
    # warnings.warn('Warning, duality training.')
    # if "erosion" in args['morp_operation'].name:
    #     args['random_gen_args']['p_invert'] = 1
    # elif "dilation" in args['morp_operation'].name:
    #     args['random_gen_args']['p_invert'] = 0

    # elif "closing" in args['morp_operation'].name:
    #     args['random_gen_args']['p_invert'] = 1
    # elif "opening" in args['morp_operation'].name:
    #     args['random_gen_args']['p_invert'] = 0

    # elif "white_tophat" in args['morp_operation'].name:
    #     args['random_gen_args']['p_invert'] = 1
    # elif "black_tophat" in args['morp_operation'].name:
    #     args['random_gen_args']['p_invert'] = 0

    for key in ['closest_selem_method', 'closest_selem_distance_fn', 'bias_optim_mode']:
        args[f'{key}_str'] = str(args[key])
    
    args['init_bimonn_str'] = str(args["initializer_method"])
    args['init_bise_str'] = str(args["initializer_args"]["bise_init_method"])

    if args['dataset_type'] == "axspa_roi":
        args['dataset_path'] = "data/deep_morpho/axspa_roi/axspa_roi.csv"
        args['morp_operation'] = []
        args['experiment_subname'] = 'axspa_roi'
        args['freq_imgs'] = 10
        args['n_atoms'] = len(args['channels']) - 1
        args['loss_data'] = loss_dict[args['loss_data_str']](border=(0, 0))

    # if args['init_weight_mode'] == InitBiseEnum.CUSTOM_CONSTANT and args['atomic_element'] == "bisel":
    #     args['init_bias_value_bise'] = "auto"
    #     args['init_bias_value_lui'] = "auto"

    if args['dataset_type'] in ["diskorect", 'mnist', 'inverted_mnist']:
        # args['kernel_size'] = 'adapt'


        if args["kernel_size"] == "adapt":
            # args["kernel_size"] = args["morp_operation"].selems[0][0][0].shape[0]
            args["kernel_size"] = int(max(args['morp_operation'].max_selem_shape))

        if "Normalized" in args['loss_data_str'] and args['atomic_element'] == 'sybisel':
            args['loss_data'] = loss_dict[args['loss_data_str']](
                border=np.array([args['kernel_size'] // 2, args['kernel_size'] // 2]),
                vmin=-1, vmax=1,
            )
        else:
            # args['loss_data'] = loss_dict[args['loss_data_str']]()
            args['loss_data'] = loss_dict[args['loss_data_str']](border=np.array([args['kernel_size'] // 2, args['kernel_size'] // 2]))

        args['experiment_subname'] = f"{args['atomic_element']}/{args['threshold_mode']['weight']}/{args['dataset_type']}/{args['morp_operation'].name}"

        if args['channels'] == 'adapt':
            args['channels'] = args['morp_operation'].in_channels + [args['morp_operation'].out_channels[-1]]

        if args["n_atoms"] == 'adapt':
            args['n_atoms'] = len(args['morp_operation'])
            if args['atomic_element'] in ['cobise', 'cobisec']:
                args['n_atoms'] = max(args['n_atoms'] // 2, 1)


    if args['dataset_type'] == "diskorect":
        args['n_epochs'] = 1
        args['n_inputs'] = args['n_steps'] * args['batch_size']
        args["random_gen_args"] = args["random_gen_args"].copy()
        # args["random_gen_args"]["border"] = (args["kernel_size"]//2 + 1, args["kernel_size"]//2 + 1)
        args['random_gen_args']['size'] = args['random_gen_args']['size'] + (args["morp_operation"].in_channels[0],)

    if args['dataset_type'] in ["mnist", "inverted_mnist"]:
        args['freq_imgs'] = 300
        args['n_inputs'] = 70_000

    if args['dataset_type'] == 'inverted_mnist':
        args['mnist_args']['invert_input_proba'] = 1
        # args['experiment_subname'] = args['experiment_subname'].replace('mnist', 'inverted_mnist')


    if args['atomic_element'] == "conv":
        args['threshold_mode'] = {"activation": "sigmoid", "weight": "identity"}

    if args['atomic_element'] == "sybisel":
        # args['threshold_mode']["activation"] += "_symetric"
        args['threshold_mode'] = {'weight': args['threshold_mode']['weight'], 'activation': args['threshold_mode']['activation'] + "_symetric"}
        args['bias_optim_mode'] = BiseBiasOptimEnum.RAW
        if "mnist" in args['dataset_type']:
            args['mnist_args']['do_symetric_output'] = True
        # args['init_bias_value_bise'] = 0
        # args['init_bias_value_lui'] = 0

    args['loss'] = {"loss_data": args['loss_data']}

    if isinstance(args['threshold_mode'], str) or args['threshold_mode']['weight'] != "identity":
        args['loss_regu'] = "None"


    if args['loss_regu'] != "None":
        args['loss_regu'] = (loss_dict[args['loss_regu'][0]], args['loss_regu'][1])
        args['loss']['loss_regu'] = args['loss_regu']

#     already_seen_path = f"deep_morpho/results/results_tensorboards/Bimonn_exp_63/multi/sybisel/softplus/diskorect/seen_args.txt"
#     with open(already_seen_path, "r") as f:
#         already_seen = f.read()
#     if str((
#         args['morp_operation'].name.split('/')[0],
#         args['morp_operation'].selem_names[0][-1][0],
#         # str(args["init_weight_mode"]).split(".")[-1],
#         # str(args["bias_optim_mode"]).split(".")[-1],
#         args["loss_data_str"],
#         str(args["learning_rate"])
#     )) in already_seen:
#         to_remove.append(idx)

# for idx in to_remove[::-1]:
#     del all_args[idx]

# print(len(to_remove))
# assert False

# pass