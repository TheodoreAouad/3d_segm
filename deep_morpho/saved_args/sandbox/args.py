import warnings
import numpy as np
import torch.optim as optim
import torch.nn as nn
import os

from deep_morpho.datasets.generate_forms3 import get_random_diskorect_channels
from deep_morpho.datasets.sticks_noised_dataset import SticksNoisedGeneratorDataset
from deep_morpho.datasets.gray_to_channels_dataset import LevelsetValuesEqualIndex
from deep_morpho.loss import (
    MaskedMSELoss, MaskedDiceLoss, MaskedBCELoss, QuadraticBoundRegularization, LinearBoundRegularization,
    MaskedBCENormalizedLoss, MaskedNormalizedDiceLoss, BCENormalizedLoss, DiceLoss, NormalizedDiceLoss
)
from general.utils import dict_cross
from deep_morpho.models.bise_base import ClosestSelemEnum, ClosestSelemDistanceEnum, BiseBiasOptimEnum, BiseWeightsOptimEnum
from deep_morpho.initializer import InitBimonnEnum, InitBiseEnum
from deep_morpho.env import CLASSIF_DATASETS
from deep_morpho.experiments.parser import MultiParser
from .args_morp_ops import morp_operations


loss_dict = {
    "MaskedMSELoss": MaskedMSELoss,
    "MaskedDiceLoss": MaskedDiceLoss,
    "DiceLoss": DiceLoss,
    "MaskedBCELoss": MaskedBCELoss,
    "MaskedBCENormalizedLoss": MaskedBCENormalizedLoss,
    "quadratic": QuadraticBoundRegularization,
    "linear": LinearBoundRegularization,
    "MaskedNormalizedDiceLoss": MaskedNormalizedDiceLoss,
    "MSELoss": nn.MSELoss,
    "BCENormalizedLoss": BCENormalizedLoss,
    'NormalizedDiceLoss': NormalizedDiceLoss,
    "BCELoss": nn.BCELoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
}

all_args = MultiParser()

# all_args['batch_seed'] = [2249939862]
all_args['batch_seed'] = [None]

all_args['n_try'] = [0]
# all_args['n_try'] = [0, 1, 2, 3, 4]
# all_args['n_try'] = range(1, 20)

all_args['experiment_name'] = [
    # "Bimonn_exp_76/sandbox/bisel-dense/0",
    "tests",
]


# DATA ARGS
all_args['morp_operation.datamodule'] = morp_operations
all_args['dataset'] = [
    # 'diskorectdataset',
    # 'mnistmorphodataset',
    # 'mnistgrayscaledataset',
    # 'fashionmnistgrayscaledataset',
    # 'axsparoidataset',
    # 'axsparoisimpledataset',
    # AxspaROISimpleDataset

    'mnistclassifdataset',
    # 'mnistclassifchanneldataset',

    # 'cifar10dataset',
    # 'cifar100dataset',
]
all_args['preprocessing.datamodule'] = [  # for axspa roi
    None,
]
all_args['dataset_path.datamodule'] = [
    # 'data/deep_morpho/dataset_0',
    "data/deep_morpho/axspa_roi/axspa_roi.csv"
    # 'generate',
]
all_args['in_ram.datamodule'] = [
    # False,
    True,
]
all_args['random_gen_fn.datamodule'] = [
    get_random_diskorect_channels
]
all_args['random_gen_args.datamodule'] = [
    # {'size': (50, 50), 'n_shapes': 2, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02}
    {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': .5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02, "border": (0, 0)}
    # {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02, "border": (0, 0)}
    # {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02,}
    # {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02, "border": (0, 0)}
    # {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 1, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02, "border": (0, 0)}
]
all_args['mnist_args.datamodule'] = [
    {"threshold": 30, "size": (50, 50), "invert_input_proba": 0,},
    # {"threshold": 30, "size": (50, 50), "invert_input_proba": 1},
]
all_args['mnist_gray_args.datamodule'] = [
    {"n_gray_scale_values": 20, "size": (50, 50),}
]

all_args['fashionmnist_gray_args.datamodule'] = [
    {"n_gray_scale_values": 20, "size": (50, 50), }
]

all_args['channel_classif_args.datamodule'] = [
    {
        "levelset_handler_mode": "LevelsetValuesEqualIndex",
        "levelset_handler_args": {"n_values": 10},
    }
]

all_args['sticks_noised_angles.datamodule'] = [
    # [0, 90],
    # [30, 60],
    # [0, 90],
    [0, 45, 90]
    # np.linspace(0, 180, 5),
    # np.linspace(0, 160, 5),
    # [30, 120],
]
all_args['sticks_noised_args.datamodule'] = [
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
all_args['n_steps.datamodule'] = [10000]
all_args['nb_batch_indep.datamodule'] = [0]
all_args["n_inputs_train"] = [50_000]
all_args["n_inputs_val"] = [10_000]
all_args["n_inputs_test"] = [10_000]
# all_args['n_inputs'] = [
#     3_000_000,
#     # 100_000,
# ]
# all_args['train_test_split'] = [(0.1, 0.1, 0.1)]
# all_args['train_test_split.datamodule'] = [(0.9, 0.1, 0.1)]


# TRAINING ARGS
all_args['learning_rate'] = [
    # 1e-2,
    1e-2,
]

# if max_plus, then the loss is MSELoss
all_args['loss_data_str'] = [
    # nn.BCELoss(),
    # "MaskedBCENormalizedLoss",
    # "MaskedMSELoss",
    # "MaskedNormalizedDiceLoss",
    # "MaskedBCELoss",
    # "BCENormalizedLoss",
    "BCELoss",
    # "CrossEntropyLoss",
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
all_args['freq_imgs'] = [500]
all_args['freq_hist'] = [500]
all_args["freq_update_binary_batch"] = [None]
all_args["freq_update_binary_epoch"] = [3]
all_args['freq_scalars'] = [50]
all_args['max_epochs.trainer'] = [20]

all_args['patience_loss_batch'] = [2100]
all_args['patience_loss_epoch'] = [9]
all_args['patience_reduce_lr'] = [1/3]
all_args['early_stopping_on'] = [
    # 'batch',
    'epoch'
]


# MODEL ARGS
# all_args['n_atoms'] = [
#     # 'adapt',
#     4,
# ]
all_args["model"] = [
    # "BiMoNNClassifierMaxPoolNotBinary",
    # "BiMoNNClassifierMaxPool",
    # "BiMoNNClassifierLastLinearNotBinary",
    "BiMoNNClassifierLastLinear",
    # "BiMoNN",
    # "BimonnDense",
    # "BimonnDenseNotBinary",
    # "BimonnBiselDenseNotBinary",
]
all_args['atomic_element.net'] = [
    "bisel",
    # "dual_bisel",
    # "sybisel",
]
all_args['n_atoms.net'] = [
    # 'adapt',
    1
    # 11
]

all_args['kernel_size.net'] = [
    # 3
    7
    # "adapt",
]
all_args['channels.net'] = [
    # 'adapt',
    # [1, 1]
    [50, 50]
    # [1, 100,],
    # [1, 100, ],
    # [50, 100, ],
    # [200, 200, 200]
    # [50, 50]
    # [20, 20, 20],
    # [40, 40, 40],
    # [80, 20, 20],
    # [80, 20,],
    # [1, 50, 50],
    # [1, 75, 75],
    # [1, 100, 100],
    # [1, 200, 200],
]
all_args['closest_selem_method.net'] = [
    # ClosestSelemEnum.MIN_DIST
    # ClosestSelemEnum.MAX_SECOND_DERIVATIVE
    ClosestSelemEnum.MIN_DIST_DIST_TO_CST
]

all_args['bias_optim_mode.net'] = [
    # BiseBiasOptimEnum.RAW,
    BiseBiasOptimEnum.POSITIVE,
    # BiseBiasOptimEnum.POSITIVE_INTERVAL_PROJECTED,
    # BiseBiasOptimEnum.POSITIVE_INTERVAL_REPARAMETRIZED
]
all_args['bias_optim_args.net'] = [
    {"offset": 0}
]
all_args['weights_optim_mode.net'] = [
    BiseWeightsOptimEnum.THRESHOLDED,
    # BiseWeightsOptimEnum.ELLIPSE_ROOT,
    # BiseWeightsOptimEnum.NORMALIZED
]

all_args['threshold_mode.net'] = [
    {
        # "weight": 'identity',
        "weight": 'softplus',
        "activation": 'tanh',
        # "activation": 'sigmoid',
    },
]
all_args['weights_optim_args.net'] = [
    # {"constant_P": True}
    {"constant_P": True, "factor": 1}
]

all_args['initializer_method.net'] = [
    InitBimonnEnum.INPUT_MEAN,
]
all_args['initializer_args.net'] = [
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

all_args['activation_P.net'] = [0]
all_args['constant_activation_P.net'] = [False]
all_args['force_lui_identity.net'] = [False]
all_args['constant_P_lui.net'] = [False]



if all_args['dataset'] in [[k] for k in ['axspa_roi', "sticks_noised", "classif_mnist"]]:
    all_args['morp_operation'] = [None]

all_args.parse_args()

# all_args = dict_cross(all_args)
#

to_remove = []
for idx, args in enumerate(all_args.multi_args):

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

    args['patience_loss'] = args[f"patience_loss_{args['early_stopping_on']}"]
    args['patience_reduce_lr'] = max(int(args["patience_loss"] * args['patience_reduce_lr']) - 1, 1)

    if args['atomic_element'] == "dual_bisel":
        args['weights_optim_mode'] = BiseWeightsOptimEnum.NORMALIZED

    if args['weights_optim_mode'] == BiseWeightsOptimEnum.NORMALIZED:
        args['initializer_args'].update({
            'bise_init_method': InitBiseEnum.CUSTOM_CONSTANT_DUAL_RANDOM_BIAS,
            'lui_init_method': InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS_DUAL_RANDOM_BIAS,
        })

    args['init_bimonn_str'] = str(args["initializer_method"])
    if isinstance(args["initializer_args"], dict):
        args['init_bise_str'] = str(args["initializer_args"]["bise_init_method"])
    elif isinstance(args["initializer_args"], list):
        args['init_bise_str'] = [str(ar["bise_init_method"]) for ar in args["initializer_args"]]

    if args['dataset'] == "axspa_roi":
        args['dataset_path'] = "data/deep_morpho/axspa_roi/axspa_roi.csv"
        args['morp_operation'] = []
        args['freq_imgs'] = 20
        args['freq_scalars'] = 1
        args['batch_size'] = 16
        args['n_atoms'] = len(args['channels']) - 1
        args['experiment_subname'] = f"{args['atomic_element']}/{args['threshold_mode']['weight']}/{args['dataset']}/"

        args['patience_loss'] = 360
        args['patience_reduce_lr'] = 120


        if args["kernel_size"] == "adapt":
            size = np.ceil(41 / args['n_atoms'])
            if size % 2 == 0:
                size += 1
            args['kernel_size'] = int(size)

    if args['dataset'] == "sticks_noised":
        args["sticks_noised_args"] = args["sticks_noised_args"].copy()
        args['sticks_noised_args']['angles'] = args['sticks_noised_angles']
        args['morp_operation'] = SticksNoisedGeneratorDataset.get_default_morp_operation(
            lengths_lim=args['sticks_noised_args']['lengths_lim'],
            angles=args['sticks_noised_args']['angles'],
        )
        args['sticks_noised_args']['size'] = args['sticks_noised_args']['size'] + (args["morp_operation"].in_channels[0],)

    # if args['init_weight_mode'] == InitBiseEnum.CUSTOM_CONSTANT and args['atomic_element'] == "bisel":
    #     args['init_bias_value_bise'] = "auto"
    #     args['init_bias_value_lui'] = "auto"

    if args['dataset'] in [
        # "diskorectdataset", 
        'mnist', 'inverted_mnist', 'sticks_noised', 'mnist_gray', 'fashionmnist']:
        # args['kernel_size'] = 'adapt'


        if args["kernel_size"] == "adapt":
            # args["kernel_size"] = args["morp_operation"].selems[0][0][0].shape[0]
            args["kernel_size"] = int(max(args['morp_operation'].max_selem_shape))

        # kwargs_loss = {}
        # if "Normalized" in args['loss_data_str'] and args['atomic_element'] == 'sybisel':
        #     kwargs_loss.update({"vmin": -1, "vmax": 1})
        #     # args['loss_data'] = loss_dict[args['loss_data_str']](
        #     #     border=np.array([args['kernel_size'] // 2, args['kernel_size'] // 2]),
        #     #     vmin=-1, vmax=1,
        #     # )
        # if "Masked" in args['loss_data_str']:
        #     kwargs_loss.update({"border": np.array([args['kernel_size'] // 2, args['kernel_size'] // 2])})
        #     # args['loss_data'] = loss_dict[args['loss_data_str']]()
        #     # args['loss_data'] = loss_dict[args['loss_data_str']](border=np.array([args['kernel_size'] // 2, args['kernel_size'] // 2]))
        # args['loss_data'] = loss_dict[args['loss_data_str']](**kwargs_loss)

        # args['experiment_subname'] = f"{args['atomic_element']}/{args['threshold_mode']['weight']}/{args['dataset']}/{args['morp_operation'].name}"

        if args['channels'] == 'adapt':
            args['channels'] = args['morp_operation'].in_channels + [args['morp_operation'].out_channels[-1]]

        if args["n_atoms"] == 'adapt':
            args['n_atoms'] = len(args['morp_operation'])

    else:
        args['experiment_subname'] = f"{args['atomic_element']}/{args['threshold_mode']['weight']}/{args['dataset']}"

    if args['dataset'] in CLASSIF_DATASETS:
        if args['n_atoms'] == "adapt":
            args["n_atoms"] = len(args['channels']) - 1
    else:
        args["model_type"] = ["LightningBiMoNN"]

    if args['atomic_element'] == "sybisel":
        # args['threshold_mode']["activation"] += "_symetric"
        args['threshold_mode'] = {'weight': args['threshold_mode']['weight'], 'activation': args['threshold_mode']['activation'] + "_symetric"}
        args['bias_optim_mode'] = BiseBiasOptimEnum.RAW
        if args["loss_data_str"] == "BCELoss":
            args["loss_data_str"] = "BCENormalizedLoss"
        # if "mnist" in args['dataset']:
        #     args['mnist_args']['do_symetric_output'] = True

        # args['init_bias_value_bise'] = 0
        # args['init_bias_value_lui'] = 0

    kwargs_loss = {}
    if "Normalized" in args['loss_data_str'] and args['atomic_element'] == 'sybisel':
        kwargs_loss.update({"vmin": -1, "vmax": 1})
        # args['loss_data'] = loss_dict[args['loss_data_str']](
        #     border=np.array([args['kernel_size'] // 2, args['kernel_size'] // 2]),
        #     vmin=-1, vmax=1,
        # )

    args['loss_data'] = loss_dict[args['loss_data_str']](**kwargs_loss)

    if args['dataset'] in [
        # 'diskorectdataset', 
        'sticks_noised']:
        args['n_epochs'] = 1
        args['n_inputs'] = args['n_steps'] * args['batch_size']

    # if args['dataset'] == "diskorectdataset":
    #     args["random_gen_args"] = args["random_gen_args"].copy()
    #     # args["random_gen_args"]["border"] = (args["kernel_size"]//2 + 1, args["kernel_size"]//2 + 1)
    #     args['random_gen_args']['size'] = args['random_gen_args']['size'] + (args["morp_operation"].in_channels[0],)

    if args['dataset'] in ["mnist", "inverted_mnist", "mnist_gray", "fashionmnist", "classif_mnist", "classif_mnist_channel"]:
        # args['freq_imgs'] = 300
        args['n_inputs'] = 60_000

    if args['dataset'] in ["cifar10", "cifar100"]:
        args["n_inputs"] = 50_000
        args["channels"][0] = args["channel_classif_args"]["levelset_handler_args"]["n_values"] * 3

    if args['dataset'] in ["cifar10", "cifar100"]:
        args["channels"][0] = args["channel_classif_args"]["levelset_handler_args"]["n_values"] * 1


    if args['dataset'] == 'inverted_mnist':
        args['mnist_args']['invert_input_proba'] = 1
        # args['experiment_subname'] = args['experiment_subname'].replace('mnist', 'inverted_mnist')



    args['loss'] = {"loss_data": args['loss_data']}
    # import torch
    # args['loss'] = {"loss_data": lambda *x: torch.ones(1, requires_grad=True)[0]}

    if isinstance(args['threshold_mode'], str) or args['threshold_mode']['weight'] != "identity":
        args['loss_regu'] = "None"


    if args['loss_regu'] != "None":
        args['loss_regu'] = (loss_dict[args['loss_regu'][0]], args['loss_regu'][1])
        args['loss']['loss_regu'] = args['loss_regu']

    for key in ['closest_selem_method', 'bias_optim_mode']:
        args[f'{key}_str'] = str(args[key])

    if args['dataset'] in ['mnist_gray', 'fashionmnist']:
        assert "gray" in args['morp_operation'].name
    elif args['dataset'] in ["mnist", "diskorectdataset", "inverted_mnist"]:
        assert "gray" not in args['morp_operation'].name


    already_seen_path = f"deep_morpho/results/results_tensorboards/{args['experiment_name']}/{args['atomic_element']}/{args['threshold_mode']['weight']}/{args['dataset']}/seen_args.txt"
    if not os.path.exists(already_seen_path):
        continue
    with open(already_seen_path, "r") as f:
        already_seen = f.read()
    if str((
        args['morp_operation'].name.split('/')[0],
        args['morp_operation'].selem_names[0][-1][0],
        # str(args["init_weight_mode"]).split(".")[-1],
        # str(args["bias_optim_mode"]).split(".")[-1],
        args["loss_data_str"],
        str(args["learning_rate"]),
        args["optimizer"].__name__, args["bias_optim_mode"].__str__().split(".")[-1]
    )) in already_seen:
        to_remove.append(idx)

# print(f'Deleting {len(to_remove)} already seen args...')
# for idx in to_remove[::-1]:
#     del all_args[idx]
# assert False

# pass