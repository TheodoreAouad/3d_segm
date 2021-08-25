import numpy as np
from skimage.morphology import disk

from deep_morpho.datasets.generate_forms3 import get_random_rotated_diskorect
from general.utils import dict_cross
from .args_morp_ops import morp_operations

args = {}

# TODO: have one arg for the fn to learn, and one arg for the neural network OR one arg for both but handle it differently...
args['morp_operation'] = morp_operations

args['logical_not'] = [
    False
]

args['batch_size'] = [32]
args['n_inputs'] = [
    200_000,
    # 2_000_000,
]
args['learning_rate'] = [1e-2/2]
args['random_gen_fn'] = [get_random_rotated_diskorect]
args['random_gen_args'] = [{'size': (50, 50), 'n_shapes': 15, 'max_shape': (15, 15), 'p_invert': 0.5}]

args['kernel_size'] = [7]
args['init_weight_identity'] = [True]
args['activation_P'] = [1]
args['threshold_mode'] = [
    # 'arctan',
    # 'sigmoid',
    'tanh',
    # 'erf',
]
args['share_weights'] = [False]
args['do_thresh_penalization'] = [False]
args['args_thresh_penalization'] = [{
    'coef': .005,
    'degree': 4,
    'detach_weights': True,
}]
args['first_batch_pen'] = [1]

selems = []

# selem = np.zeros((5, 5))
# selem[1:-1, 1:-1] = 1
# selem[0, 2] = 1
# selem[-1, 2] = 1
# selems.append(selem)

# selem = np.zeros((5, 5))
selem = disk(2)
selems.append(selem)

selem = np.zeros((5, 5))
# selem[np.arange(5), 5 - np.arange(1, 6)] = 1
# selem[np.arange(5), np.arange(5)] = 1
# # selem[:, 2] = 1
# # selem[2, :] = 1
selem[np.arange(5), 5 - np.arange(1, 6)] = 1
selem[np.arange(5), np.arange(5)] = 1
# selem[:, 2] = 1
# selem[2, :] = 1
# selems.append(selem)

selem = np.zeros((5, 5))
selem[1:-1, 1:-1] = 1
# selems.append(selem)

selem = np.zeros((5, 5))
selem[:, 2] = 1
# selems.append(selem)

selem = np.zeros((5, 5))
selem[2, :] = 1
# selems.append(selem)
#
# selem = np.zeros((5, 5))
# selem[1:-1, 1:-1] = 1
# selems.append(selem)
#
# selem = np.zeros((5, 5))
# selem[:, 2] = 1
# selems.append(selem)
#
# selem = np.zeros((5, 5))
# selem[2, :] = 1
# selems.append(selem)


# args['selem'] = selems
args['selem'] = [None]

all_args = dict_cross(args)
#
# for idx, args in enumerate(all_args):
#     if args['morp_operation'] in ['Erosion', 'Dilation']:
#         all_args[idx]['n_inputs'] = 200_000
