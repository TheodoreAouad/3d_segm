import numpy as np
from skimage.morphology import disk

from deep_morpho.datasets.generate_forms3 import get_random_rotated_diskorect
from general.utils import dict_cross
from general.array_morphology import sequentiel_morp_operations

args = {}


seq_selems = []

selem = np.zeros((7, 7))
selem[3, :] = 1
seq_selems.append(selem)

selem = np.zeros((7, 7))
selem[:, 3] = 1
seq_selems.append(selem)


# TODO: have one arg for the fn to learn, and one arg for the neural network OR one arg for both but handle it differently...
args['morp_operation'] = [
    # 'Dilation',
    # 'Erosion',
    # 'Opening',
    sequentiel_morp_operations(['dilation', 'dilation'], selems=seq_selems, device="cuda", return_numpy_array=False)
]

args['logical_not'] = [
    False
]

args['batch_size'] = [32]
args['n_inputs'] = [
    # 200_000,
    5_000_000,
]
args['learning_rate'] = [1e-3]
args['random_gen_fn'] = [get_random_rotated_diskorect]
args['random_gen_args'] = [{'size': (50, 50), 'n_shapes': 15, 'max_shape': (15, 15), 'p_invert': 0.5}]

args['activation_P'] = [1]
args['activation_mode'] = [
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


args['selem'] = selems

all_args = dict_cross(args)
#
# for idx, args in enumerate(all_args):
#     if args['morp_operation'] in ['Erosion', 'Dilation']:
#         all_args[idx]['n_inputs'] = 200_000
