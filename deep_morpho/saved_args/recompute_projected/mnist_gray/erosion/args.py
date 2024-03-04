import warnings
from functools import partial
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import os

from general.structuring_elements import *
from deep_morpho.morp_operations import ParallelMorpOperations
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
from deep_morpho.saved_args.recompute_projected.args_enforcers import enforcers
from deep_morpho.datasets.cifar_dataset import transform_default
from deep_morpho.datasets.spalike_dataset import SpalikeSegmEnum


all_args = GridParser()

all_args['batch_seed'] = [2249939862]

all_args['n_try'] = [0]

all_args['experiment_name'] = [
    # "Bimonn_exp_75/debug/"
    "Bimonn_exp_75/1"
]

all_args["model"] = [
    "BiMoNN",
]

all_args['dataset'] = [
    ##### MORPHO ####
    # 'diskorectdataset',
    # 'mnistmorphodataset',
    # 'invertedmnistmorphodataset',
    'mnistgrayscaledataset',
    # 'fashionmnistgrayscaledataset',
]


# DATA ARGS
morp_operations = []
for op in [
    'disk',
    "hstick",
    "dcross",
    "bsquare",
    "bdiamond",
    "bcomplex",
    "scross"
]:
    size = 7
    morp_operations.append(ParallelMorpOperations.erosion((op, size), name=f"erosion/{op}"))


all_args['morp_operation'] = morp_operations

all_args['preprocessing'] = [  # for axspa roi
    None,
]

if True:  # args diskorect, axsparoi, mnist
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

if True:  # loss and optimizer
    all_args['loss_data_str'] = [
        "BCELoss",
        "MSELoss",
        "DiceLoss",
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
        3,
    ]
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
    all_args['freq_scalars'] = [5]



# MODEL ARGS

all_args["activation_constructor"] = [
    # nn.ReLU,
    # nn.Sigmoid,
    NormalizedTanh,
]
all_args["apply_last_activation"] = [
    # False,
    True,
]

all_args['atomic_element'] = [
    "bisel",
    "dual_bisel",
    # "sybisel",
]
all_args['n_atoms'] = [
    'adapt',
]

all_args['channels'] = [
    'adapt',
]
all_args['kernel_size'] = [
    "adapt",
]

all_args['activation_P'] = [1]

if True:  # weights and bias handler
    all_args['bias_optim_mode'] = [
        # BiseBiasOptimEnum.RAW,
        # BiseBiasOptimEnum.POSITIVE,
        BiseBiasOptimEnum.POSITIVE_INTERVAL_PROJECTED,
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

if True:  # args script necessity
    all_args['observables'] = [[]]


    if all_args['dataset'] in [[k] for k in ['axspa_roi', "sticks_noised", "classif_mnist"]]:
        all_args['morp_operation'] = [None]

    all_args["args_enforcers"] = enforcers

    all_args.parse_args()
