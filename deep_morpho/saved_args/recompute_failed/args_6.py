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
from deep_morpho.experiments.parser import GridParser
from .args_morp_ops import morp_operations, morp_operations_gray, morp_operations_binary
from .args_enforcers import enforcers
from deep_morpho.datasets.cifar_dataset import transform_default
from deep_morpho.datasets.spalike_dataset import SpalikeSegmEnum


all_args = GridParser()

# all_args['batch_seed'] = [2249939862]
all_args['batch_seed'] = [None]

all_args['n_try'] = [0, 1, 2, 3, 4]

all_args['experiment_name'] = [
    # "Bimonn_exp_76/sandbox/bisel-dense/0",
    # "debug",
    # "test"
    # "Bimonn_exp_81/sandbox/positive_weights/"
    # "Bimonn_exp_82/sandbox/multi/nosegm"
    # "Bimonn_exp_82/sandbox_merged/0_/"
    # "Bimonn_exp_80/sandbox/dilation_proj_activated/"
    # "Bimonn_exp_82/sandbox/3_/"
    # "Bimonn_exp_75/sandbox/4_/"
    # "Bimonn_exp_75/debug/"
    "Bimonn_exp_75/multi/recompute_failed/0"
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
    "BiMoNN",
    # "BimonnIdentity",  # DEBUG

    ##### CLASSIFIERS #####
    # "BiMoNNClassifierMaxPoolNotBinary",
    # "BiMoNNClassifierMaxPool",
    # "BiMoNNClassifierLastLinear",
    # "BimonnDense",
    # "BimonnDenseNotBinary",
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

    ###### AXSPA ######
    # "BimonnAxspaResnet",
    # "BimonnAxspaConv",
    # "ConvSpalikeMerged",
    # "ResnetSpalikeMerged",
]

all_args['dataset'] = [
    ##### MORPHO ####
    'diskorectdataset',
    # 'mnistmorphodataset',
    # 'invertedmnistmorphodataset',
    # "noistidataset",
    # 'mnistgrayscaledataset',
    # 'fashionmnistgrayscaledataset',
    # 'axsparoidataset',
    # 'axsparoisimpledataset',

    ##### CLASSIFICATION #####
    # 'mnistclassifdataset',
    # 'mnistclassifchanneldataset',

    # 'cifar10dataset',
    # 'cifar100dataset',

    # 'mnistclassical',
    # 'cifar10classical',
    # 'cifar100classical',

    ###### AXSPA ######
    # 'spalikedataset',
    # 'spalikedatasetmerged',
]


# DATA ARGS
all_args['morp_operation'] = morp_operations

all_args['preprocessing'] = [  # for axspa roi
    None,
]
# all_args['transform.train'] = [
#     transforms.Compose([
#         transforms.RandomRotation(degrees=10),
#         transform_default,
#     ])
# ]

all_args['channel_classif_args'] = [
    {
        "levelset_handler_mode": LevelsetValuesEqualIndex,
        # "levelset_handler_args": {"n_values": 10},
        "levelset_handler_args": {"n_values": 1},
    }
]

if True:  # args diskorect, axsparoi, mnist
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


if True:  # SpaLike Args
    all_args["spalike_args"] = [
        {
            "image_size": (256, 256),
            "proba_lesion": 0.5,
            "proba_lesion_locations": {
                "sacrum": 0.4,
                "iliac": 0.4,
            },
            "grid_spacing": (24, 24),
            "min_ellipse_axes": 13,
            "max_ellipse_axes": 35,
            "period": (3, 10),
            "offset": (1, 2),
            "min_output_ellipse": 0,
            "max_n_blob_sane": 5,
            "segm_mode": (
                # SpalikeSegmEnum.BonesSeparated
                SpalikeSegmEnum.BonesOverlapped
                # SpalikeSegmEnum.Roi
                # SpalikeSegmEnum.NoSegm
            ),
            "normalize": True,
            "iliac_dil_coef": .7,
            "sacrum_dil_coef": 2.,
        },
    ]


if True:  # Noisti Args
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


if True:  # nb inputs
    all_args['n_steps_train'] = [100]  # for Generation
    all_args['n_steps_val'] = [10]  # for Generation
    all_args['n_steps_test'] = [20]  # for Generation

    all_args['nb_batch_indep'] = [0]

    all_args["n_inputs_train"] = [50_000]  # for MNIST
    all_args["n_inputs_val"] = [10_000]  # for MNIST
    all_args["n_inputs_test"] = [10_000]  # for MNIST

# TRAINING ARGS

if True:  # lr
    all_args['learning_rate'] = [
        # 1e-3,
        1e-1,
        1e-2,
        # 0.0001,
        # 1e-3,
        # 1e-4,
    ]
    all_args['lr_bimonn'] = [
        1e-2 / 2,
    ]
    all_args['lr_classifier'] = [
        1e-4
    ]

if True:  # loss and optimizer
    all_args['apply_one_hot_target'] = [
        # False,  # For cross entropy loss
        True,
    ]
    all_args['loss_data_str'] = [
        # nn.BCELoss(),
        # "MaskedBCENormalizedLoss",
        # "MaskedMSELoss",
        # "MaskedNormalizedDiceLoss",
        # "MaskedBCELoss",
        # "BCENormalizedLoss",
        "BCELoss",
        # "BCEWithLogitsLoss",
        # "CrossEntropyLoss",
        # "SquaredHingeLoss",
        "MSELoss",
        "DiceLoss",
        # "MaskedDiceLoss",
        # "NormalizedDiceLoss",
    ]
    all_args['loss_regu'] = [
        # ("quadratic", {"lower_bound": 0, "upper_bound": np.infty, "lambda_": 0.01})
        # "linear",
        "None",
        # ("RegularizationProjConstant", {"mode": "exact"}),
        # ("RegularizationProjConstant", {"mode": "uniform"}),
        # ("RegularizationProjConstant", {"mode": "normal"}),
        # ("RegularizationProjActivated", {}),
    ]
    all_args["loss_coefs"] = [
        # {"loss_data": 1, "loss_regu": 0},
        {"loss_data": 1, "loss_regu": 1000},
        # {"loss_data": 1, "loss_regu": 0.1},
        # {"loss_data": 1, "loss_regu": 0.01},
        # {"loss_data": 1, "loss_regu": 0.001},
    ]
    all_args["loss_regu_delay"] = [1000]

    all_args['optimizer'] = [
        optim.Adam,
        # optim.SGD
    ]
    all_args['optimizer_args'] = [{}]


if True:  # batch size, epochs, etc
    all_args['batch_size'] = [32]
    all_args['num_workers'] = [
        20,
        # 7
        # 0
    ]
    # all_args['max_epochs.trainer'] = [10]
    all_args['max_epochs.trainer'] = [200]


    all_args['patience_loss_batch'] = [2100]
    all_args['patience_loss_epoch'] = [15]
    # all_args['patience_loss_epoch'] = [1]  # DEBUG
    all_args['patience_reduce_lr'] = [1/5]
    # all_args['patience_reduce_lr'] = [5]  # DEBUG
    all_args['early_stopping_on'] = [
        # 'batch',
        'epoch'
    ]

if True:
    all_args['freq_imgs'] = [
        # 1,
        # int(50000/64) + 1,
        # "epoch"
        2000,
    ]
    all_args['freq_hist'] = [
        # 1,
        # int(50000/64) + 1,
        # "epoch"
        2000,
    ]
    all_args['freq_imgs_val'] = [
        # "one_per_val",
        np.infty,
    ]
    all_args["freq_update_binary_batch"] = [
        # 1
        None
    ]
    all_args["freq_update_binary_epoch"] = [
        1,
        # None,
    ]
    all_args['freq_scalars'] = [10]



# MODEL ARGS

all_args["do_batchnorm"] = [  # For BNNConv, BimonnAxspa
    # True,
    False,
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
all_args["do_maxpool"] = [  # For ConvSpalikeMerged
    True,
    # False,
]

all_args['atomic_element'] = [
    "bisel",
    "dual_bisel",
    # "sybisel",
]
all_args['n_atoms'] = [
    'adapt',
]

all_args["classif_neurons"] = [  # For ConvSpalikeMerged
    [],
    # [512]
]
all_args["classif_kernel_size"] = [  # For ConvSpalikeMerged
    7,
]
all_args["classif_channels"] = [  # For ConvSpalikeMerged
    [128, 256, 512],
]
all_args['kernel_size'] = [
    # 11,
    # [7, 3]
    "adapt",
    # [7, 7, 7]
]
all_args['channels'] = [
    'adapt',
    # [],
    # [2,],
    # [2, 2],
    # [2, 2, 2],
    # [2, 2, 2],
    # [256, 256, ]
    # [128, 256, 512],
    # [4096],
    # [1, 3, 3, 1],
    # [1, 3, 1],
    # [1, 10, 1],
    # [1, 1],
    # [1, 10, 1],
    # [1000, 1000],
    # [100],
    # [200, 200],
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
all_args['activation_P'] = [1]

if True:  # weights and bias handler
    all_args['bias_optim_mode'] = [
        BiseBiasOptimEnum.RAW,
        BiseBiasOptimEnum.POSITIVE,
        BiseBiasOptimEnum.POSITIVE_INTERVAL_PROJECTED,
        BiseBiasOptimEnum.POSITIVE_INTERVAL_REPARAMETRIZED
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

if True:  # init
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
            "bise_init_args": {"ub": 1e-4, "max_output_value": 0.95, "p_for_init": "auto"},

            # "bise_init_method": InitBiseEnum.ELLIPSE_ROOT,
            # "bise_init_args": {"init_bias_value": 2},

            # "bise_init_method": InitBiseEnum.CUSTOM_HEURISTIC_RANDOM_BIAS,
            # "bise_init_method": InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
            # "bise_init_args": [{"init_bias_value": -1, "mean_weight": "auto", "ub": 0.01}, {"init_bias_value": 1, "mean_weight": "auto", "ub": 0.01}]
            # "bise_init_args": {"init_bias_value": 1, "mean_weight": "auto", "ub": 0.01}
        },
    ]

if True:  # binary method
    all_args['closest_selem_method'] = [
        # ClosestSelemEnum.MIN_DIST
        # ClosestSelemEnum.MAX_SECOND_DERIVATIVE
        ClosestSelemEnum.MIN_DIST_DIST_TO_CST,
        # ClosestSelemEnum.MIN_DIST_ACTIVATED_POSITIVE,
    ]



# all_args['constant_activation_P'] = [False]
# all_args['force_lui_identity'] = [False]
# all_args['constant_P_lui'] = [False]

if True:  # args script necessity
    all_args['observables'] = [[]]


    if all_args['dataset'] in [[k] for k in ['axspa_roi', "sticks_noised", "classif_mnist"]]:
        all_args['morp_operation'] = [None]

    all_args["args_enforcers"] = enforcers

    all_args.parse_args()


# if True:  # remove already seen args
#     for args in all_args:
#         already_seen_path = f"deep_morpho/results/results_tensorboards/{args['experiment_name']}/{args['atomic_element']}/{args['threshold_mode']['weight']}/{args['dataset']}/seen_args.txt"
#         if not os.path.exists(already_seen_path):
#             continue
#         with open(already_seen_path, "r") as f:
#             already_seen = f.read()
#         if str((
#             args['morp_operation'].name.split('/')[0],
#             args['morp_operation'].selem_names[0][-1][0],
#             # str(args["init_weight_mode"]).split(".")[-1],
#             # str(args["bias_optim_mode"]).split(".")[-1],
#             args["loss_data_str"],
#             str(args["learning_rate"]),
#             args["optimizer"].__name__, args["bias_optim_mode"].__str__().split(".")[-1]
#         )) in already_seen:
#             to_remove.append(idx)

#         print(f'Deleting {len(to_remove)} already seen args...')
#         for idx in to_remove[::-1]:
#             del all_args[idx]
#         assert False

if True:  # only keep relevent args
    good_args_path = "deep_morpho/saved_args/recompute_failed/args_good_0.txt"
    to_remove = []
    for idx, args in enumerate(all_args.multi_args):
        if not os.path.exists(good_args_path):
            continue
        with open(good_args_path, "r") as f:
            good_args = f.read()

        dataset_type = args['dataset'].replace("dataset", "")
        operation, selem = args['morp_operation'].name.split('/')
        loss = args["loss_data_str"]
        bias_mode = args["bias_optim_mode"].__str__().split(".")[-1]
        lr = str(args["learning_rate"])
        atomic_element = args['atomic_element']

        line = str((
            dataset_type,
            atomic_element,
            bias_mode,
            loss,
            lr,
            operation,
            selem,
        ))
        if line not in good_args:
            to_remove.append(idx)

    print(f'Deleting {len(to_remove)} already seen args...')
    for idx in to_remove[::-1]:
        del all_args.multi_args[idx]
    
    print(len(all_args.multi_args))
    # assert False


