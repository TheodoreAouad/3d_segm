import warnings
from functools import partial
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import os

from deep_morpho.datasets.generate_forms3 import get_random_diskorect_channels
from deep_morpho.datasets.gray_to_channels_dataset import LevelsetValuesEqualIndex
# from deep_morpho.loss import (
#     MaskedMSELoss, MaskedDiceLoss, MaskedBCELoss, QuadraticBoundRegularization, LinearBoundRegularization,
#     MaskedBCENormalizedLoss, MaskedNormalizedDiceLoss, BCENormalizedLoss, DiceLoss, NormalizedDiceLoss
# )
from deep_morpho.models.bise_base import ClosestSelemEnum, BiseBiasOptimEnum, BiseWeightsOptimEnum
from deep_morpho.models.activations import NormalizedTanh
from deep_morpho.initializer import InitBimonnEnum, InitBiseEnum
from deep_morpho.experiments.parser import MultiParser
from .args_morp_ops import morp_operations
from .args_enforcers import enforcers
from deep_morpho.datasets.cifar_dataset import transform_default


all_args = MultiParser()

# all_args['batch_seed'] = [2249939862]
all_args['batch_seed'] = [None]

all_args['n_try'] = [0]

all_args['experiment_name'] = [
    # "Bimonn_exp_76/sandbox/bisel-dense/0",
    "debug",
    # "Bimonn_exp_79/sandbox/1_/"
    # "Bimonn_exp_78/bnn/"
    # "Bimonn_exp_79/bimonn-equivalent-binaryconnect/",
    # "Bimonn_exp_78/equivalent-params-sota/"
    # "Bimonn_exp_78/raw_weights/"
    # "Bimonn_exp_78/levelset_numbers/"
    # "Bimonn_exp_78/grid_search/"
    # "Bimonn_exp_78/baseline/"
    # "Bimonn_exp_77/iccv_2023/",
    # "Bimonn_exp_77/bimonn_dense/",
    # "Bimonn_exp_77/bimonn_bisel_dense/",
    # "Bimonn_exp_77/bimonn_last_linear/"
]

all_args["model"] = [
    ##### MORPHO ####
    # "BiMoNN",

    ##### CLASSIFIERS #####
    # "BiMoNNClassifierMaxPoolNotBinary",
    # "BiMoNNClassifierMaxPool",
    # "BiMoNNClassifierLastLinear",
    # "BimonnDense",
    "BimonnDenseNotBinary",
    # "BiMoNNClassifierLastLinearNotBinary",
    # "BimonnBiselDenseNotBinary",
    # "ConvNetLastLinear",

    ##### CLASSICAL MODELS ####
    # "MLPBatchNormClassical",
    # "ResNet18",
    # "ResNet34",
    # "ResNet50",

    ###### BIBLIO ######
    # "BNNConv",
    # "MLPBinaryConnectMNIST",
    # "ConvNetBinaryConnectCifar10",
]

all_args['dataset'] = [
    # 'diskorectdataset',
    # 'mnistmorphodataset',
    # "noistidataset",
    # 'mnistgrayscaledataset',
    # 'fashionmnistgrayscaledataset',
    # 'axsparoidataset',
    # 'axsparoisimpledataset',

    # 'mnistclassifdataset',
    'mnistclassifchanneldataset',

    # 'cifar10dataset',
    # 'cifar100dataset',

    # 'mnistclassical',
    # 'cifar10classical',
    # 'cifar100classical',
]


# DATA ARGS
all_args['morp_operation'] = morp_operations

# TODO: put in an args enforcer
all_args['apply_one_hot_target'] = [
    False,  # For cross entropy loss
    # True,
]

all_args['preprocessing'] = [  # for axspa roi
    None,
]
# all_args['transform.train'] = [
#     transforms.Compose([
#         transforms.RandomRotation(degrees=10),
#         transform_default,
#     ])
# ]

all_args['dataset_path'] = [
    "data/deep_morpho/axspa_roi/axspa_roi.csv"
]
all_args['in_ram'] = [
    # False,
    True,
]
all_args['random_gen_fn'] = [
    get_random_diskorect_channels
]
all_args['random_gen_args'] = [
    {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': .5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02, "border": (0, 0)}
]
all_args['mnist_args'] = [
    {"threshold": 30, "size": (50, 50), "invert_input_proba": 0,},
]
all_args['mnist_gray_args'] = [
    {"n_gray_scale_values": 20, "size": (50, 50),}
]

all_args['fashionmnist_gray_args'] = [
    {"n_gray_scale_values": 20, "size": (50, 50), }
]

all_args['channel_classif_args'] = [
    {
        "levelset_handler_mode": LevelsetValuesEqualIndex,
        # "levelset_handler_args": {"n_values": 10},
        "levelset_handler_args": {"n_values": 10},
    }
]


##### Noisti Args #####
all_args['sticks_noised_angles'] = [
    [0, 45, 90]
]
all_args['sticks_noised_args'] = [
    {
        "size": (70, 70),
        "n_shapes": 30,
        "lengths_lim": (12, 15),
        "widths_lim": (0, 0),
        "p_invert": 0,
        "border": (0, 0),
        "noise_proba": 0.1,
    }
]
########################


all_args['n_steps'] = [1000]  # for Diskorect
all_args['nb_batch_indep'] = [0]
all_args["n_inputs_train"] = [50_000]
all_args["n_inputs_val"] = [10_000]
all_args["n_inputs_test"] = [10_000]

# TRAINING ARGS
all_args['learning_rate'] = [
    1e-1,
    # 0.001,
    # 1e-3,
    # 1e-4,
]

all_args['loss_data_str'] = [
    # nn.BCELoss(),
    # "MaskedBCENormalizedLoss",
    # "MaskedMSELoss",
    # "MaskedNormalizedDiceLoss",
    # "MaskedBCELoss",
    # "BCENormalizedLoss",
    "BCELoss",
    # "CrossEntropyLoss",
    # "SquaredHingeLoss",
    # "MSELoss",
    # "DiceLoss",
    # "MaskedDiceLoss",
    # "NormalizedDiceLoss",
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
all_args['optimizer_args'] = [{}]
all_args['batch_size'] = [64]
all_args['num_workers'] = [
    20,
    # 0,
]
all_args['freq_imgs'] = [800]
all_args['freq_hist'] = [800]
all_args["freq_update_binary_batch"] = [
    None
]
all_args["freq_update_binary_epoch"] = [
    1,
    # None,
]
all_args['freq_scalars'] = [50]
# all_args['max_epochs.trainer'] = [1]
all_args['max_epochs.trainer'] = [200]

all_args['patience_loss_batch'] = [2100]
all_args['patience_loss_epoch'] = [15]
all_args['patience_reduce_lr'] = [1/5]
all_args['early_stopping_on'] = [
    # 'batch',
    'epoch'
]


# MODEL ARGS

all_args["do_batchnorm"] = [  # For BNNConv
    True,
    # False,
]

all_args["num_units"] = [  # For MLPBinaryConnectMNIST
    1024
]

all_args["activation_constructor"] = [
    # nn.ReLU,
    # nn.Sigmoid,
    NormalizedTanh,
]
all_args["apply_last_activation"] = [
    # False,
    True,
]
all_args["do_maxpool"] = [
    True,
    # False,
]

all_args['atomic_element'] = [
    "bisel",
    # "dual_bisel",
    # "sybisel",
]
all_args['n_atoms'] = [
    'adapt',
]

all_args['kernel_size'] = [
    3
    # "adapt",
]
all_args['channels'] = [
    # 'adapt',
    # [1, 3, 3, 1],
    # [1, 3, 1],
    # [1, 10, 1],
    # [1, 1],
    # [1, 10, 1],
    # [1000, 1000],
    # [100],
    [200],
    # [200, 200, 200],
    # [300],
    # [415],  # Nb of channels for fc1 of Laydevant et al. 2021, 3.3M params, 10 level sets
    # [587, 2048, 2048,],  # 10 level sets, Nb of channels for Dense Mnist for BinaryConnect, Courbariaux and Bengio 2015, 10M params, 10 level sets
    # [50, 50],
    # [840],  # 5 level sets, 3.3M params
    # [1000, 2048, 2048,],  # 5 level sets, 10M params
    # [210],  # 20 level sets, 3.3M params
    # [330, 2048, 2048,],  # 20 level sets, 10M params
    # [4096],  # 1 level sets, 3.3M params
    # [2048, 2048, 2048],  # 1 level sets, 10M params
    # [1975],  # Baseline for the best dense bimonn on mnist
    # [100, 100],
    # [50, ],
    # [50, 50],
    # [200, 200, ],
    # [100, 100, 100],
    # [100, 100, 100, 100],
    # [50, 50, ],
]
all_args['closest_selem_method'] = [
    # ClosestSelemEnum.MIN_DIST
    # ClosestSelemEnum.MAX_SECOND_DERIVATIVE
    ClosestSelemEnum.MIN_DIST_DIST_TO_CST
]

all_args['bias_optim_mode'] = [
    # BiseBiasOptimEnum.RAW,
    BiseBiasOptimEnum.POSITIVE,
    # BiseBiasOptimEnum.POSITIVE_INTERVAL_PROJECTED,
    # BiseBiasOptimEnum.POSITIVE_INTERVAL_REPARAMETRIZED
]
all_args['bias_optim_args'] = [
    {"offset": 0}
]
all_args['weights_optim_mode'] = [
    BiseWeightsOptimEnum.THRESHOLDED,
    # BiseWeightsOptimEnum.ELLIPSE_ROOT,
    # BiseWeightsOptimEnum.NORMALIZED
]

all_args['threshold_mode'] = [
    {
        # "weight": 'identity',
        "weight": 'softplus',
        "activation": 'tanh',
        # "activation": 'tanh',
        # "activation": 'sigmoid',
    },
]
all_args['weights_optim_args'] = [
    # {"constant_P": True}
    {"constant_P": True, "factor": 1}
]

# all_args["initializer_dense"] = [
#     {
#         "initializer_method": InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
#         "initializer_args": {"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": .5},
#     }
# ]

all_args['initializer_method'] = [
    InitBimonnEnum.INPUT_MEAN,
]
all_args['initializer_args'] = [
    # force operations at init
    {
        # "bise_init_method": InitBiseEnum.KAIMING_UNIFORM,
        # "bise_init_method": InitBiseEnum.CUSTOM_HEURISTIC,

        "bise_init_method": InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
        "lui_init_method": InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS_RANDOM_BIAS,
        "bise_init_args": {"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto"},

        # "bise_init_method": InitBiseEnum.ELLIPSE_ROOT,
        # "bise_init_args": {"init_bias_value": 2},

        # "bise_init_method": InitBiseEnum.CUSTOM_HEURISTIC_RANDOM_BIAS,
        # "bise_init_method": InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
        # "bise_init_args": [{"init_bias_value": -1, "mean_weight": "auto", "ub": 0.01}, {"init_bias_value": 1, "mean_weight": "auto", "ub": 0.01}]
        # "bise_init_args": {"init_bias_value": 1, "mean_weight": "auto", "ub": 0.01}
    },

]

all_args['activation_P'] = [0]
all_args['constant_activation_P'] = [False]
all_args['force_lui_identity'] = [False]
all_args['constant_P_lui'] = [False]



if all_args['dataset'] in [[k] for k in ['axspa_roi', "sticks_noised", "classif_mnist"]]:
    all_args['morp_operation'] = [None]

all_args["args_enforcers"] = enforcers

all_args.parse_args()

# all_args = dict_cross(all_args)
#


# to_remove = []
# for idx, args in enumerate(all_args.multi_args):

#     # if args["model"].lower() == ("BimonnBiselDenseNotBinary").lower():
#     #     args["channels"] = [args["channels"][0], args["channels"][0]]

#     # Duality training
#     # warnings.warn('Warning, duality training.')
#     # if "erosion" in args['morp_operation'].name:
#     #     args['random_gen_args']['p_invert'] = 1
#     # elif "dilation" in args['morp_operation'].name:
#     #     args['random_gen_args']['p_invert'] = 0

#     # elif "closing" in args['morp_operation'].name:
#     #     args['random_gen_args']['p_invert'] = 1
#     # elif "opening" in args['morp_operation'].name:
#     #     args['random_gen_args']['p_invert'] = 0

#     # elif "white_tophat" in args['morp_operation'].name:
#     #     args['random_gen_args']['p_invert'] = 1
#     # elif "black_tophat" in args['morp_operation'].name:
#     #     args['random_gen_args']['p_invert'] = 0

#     args['patience_loss'] = args[f"patience_loss_{args['early_stopping_on']}"]
#     args['patience_reduce_lr'] = max(int(args["patience_loss"] * args['patience_reduce_lr']) - 1, 1)

#     if args['atomic_element'] == "dual_bisel":
#         args['weights_optim_mode'] = BiseWeightsOptimEnum.NORMALIZED

#     if args['weights_optim_mode'] == BiseWeightsOptimEnum.NORMALIZED:
#         args['initializer_args'].update({
#             'bise_init_method': InitBiseEnum.CUSTOM_CONSTANT_DUAL_RANDOM_BIAS,
#             'lui_init_method': InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS_DUAL_RANDOM_BIAS,
#         })

#     args['init_bimonn_str'] = str(args["initializer_method"])
#     if isinstance(args["initializer_args"], dict):
#         args['init_bise_str'] = str(args["initializer_args"]["bise_init_method"])
#     elif isinstance(args["initializer_args"], list):
#         args['init_bise_str'] = [str(ar["bise_init_method"]) for ar in args["initializer_args"]]

#     if args['atomic_element'] == "sybisel":
#         args['threshold_mode'] = {'weight': args['threshold_mode']['weight'], 'activation': args['threshold_mode']['activation'] + "_symetric"}
#         args['bias_optim_mode'] = BiseBiasOptimEnum.RAW
#         if args["loss_data_str"] == "BCELoss":
#             args["loss_data_str"] = "BCENormalizedLoss"

#     args["kwargs_loss"] = {}
#     if "Normalized" in args['loss_data_str'] and args['atomic_element'] == 'sybisel':
#         args["kwargs_loss"].update({"vmin": -1, "vmax": 1})

#     args['loss_data'] = loss_dict[args['loss_data_str']](**args["kwargs_loss"])

#     args['loss'] = {"loss_data": args['loss_data']}

#     # if isinstance(args['threshold_mode'], str) or args['threshold_mode']['weight'] != "identity":
#     #     args['loss_regu'] = "None"


#     # if args['loss_regu'] != "None":
#     #     args['loss_regu'] = (loss_dict[args['loss_regu'][0]], args['loss_regu'][1])
#     #     args['loss']['loss_regu'] = args['loss_regu']

#     for key in ['closest_selem_method', 'bias_optim_mode']:
#         args[f'{key}_str'] = str(args[key])

#     if args['dataset'] in ['mnist_gray', 'fashionmnist']:
#         assert "gray" in args['morp_operation'].name
#     elif args['dataset'] in ["mnist", "diskorectdataset", "inverted_mnist"]:
#         assert "gray" not in args['morp_operation'].name


    # already_seen_path = f"deep_morpho/results/results_tensorboards/{args['experiment_name']}/{args['atomic_element']}/{args['threshold_mode']['weight']}/{args['dataset']}/seen_args.txt"
    # if not os.path.exists(already_seen_path):
    #     continue
    # with open(already_seen_path, "r") as f:
    #     already_seen = f.read()
    # if str((
    #     args['morp_operation'].name.split('/')[0],
    #     args['morp_operation'].selem_names[0][-1][0],
    #     # str(args["init_weight_mode"]).split(".")[-1],
    #     # str(args["bias_optim_mode"]).split(".")[-1],
    #     args["loss_data_str"],
    #     str(args["learning_rate"]),
    #     args["optimizer"].__name__, args["bias_optim_mode"].__str__().split(".")[-1]
    # )) in already_seen:
    #     to_remove.append(idx)

# print(f'Deleting {len(to_remove)} already seen args...')
# for idx in to_remove[::-1]:
#     del all_args[idx]
# assert False

# pass