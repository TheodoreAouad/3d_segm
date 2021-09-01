import numpy as np
from skimage.morphology import disk

from deep_morpho.datasets.generate_forms3 import get_random_rotated_diskorect
from general.utils import dict_cross
from .args_morp_ops import morp_operations

args = {}

args['experiment_name'] = ['Bimonn_exp_16']
args['morp_operation'] = morp_operations

args['logical_not'] = [
    True,
    # False,
]

args['batch_size'] = [32]
args['n_inputs'] = [
    3_000_000,
    # 5_000_000,
]
args['learning_rate'] = [1e-3]
args['random_gen_fn'] = [get_random_rotated_diskorect]
args['random_gen_args'] = [{'size': (50, 50), 'n_shapes': 15, 'max_shape': (15, 15), 'p_invert': 0.5}]

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
    # {"activation": "tanh", "weight": "tanh", "logical_not": "clamp"}
]
args["alpha_init"] = [[0, 0]]

args['share_weights'] = [False]
args['do_thresh_penalization'] = [False]
args['args_thresh_penalization'] = [{
    'coef': .005,
    'degree': 4,
    'detach_weights': True,
}]
args['first_batch_pen'] = [1]

args['selem'] = [None]

all_args = dict_cross(args)
#
for idx, args in enumerate(all_args):
    if args["kernel_size"] == "adapt":
        args["kernel_size"] = args["morp_operation"].selems[0].shape[0]
    args["random_gen_args"]["border"] = (args["kernel_size"]//2 + 1, args["kernel_size"]//2 + 1)
#     if "dilation" in args['morp_operation'].name:
#         all_args[idx]['n_inputs'] = 200_000
