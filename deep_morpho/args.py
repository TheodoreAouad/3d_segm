import numpy as np
from skimage.morphology import disk

args = {}

# args['morp_operation'] = ['Dilation']
args['morp_operation'] = [
    'Dilation',
    # 'Erosion',
    # 'Opening',
]
# args['morp_operation'] = ['Opening']

args['batch_size'] = [32]
args['n_inputs'] = [500_000]
args['learning_rate'] = [1e-3]
args['random_gen_args'] = [{'size': (50, 50), 'n_shapes': 15, 'max_shape': (15, 15)}]

args['activation_P'] = [1]
args['activation_mode'] = [
    # 'arctan', 
    # 'sigmoid',
    'tanh',
    'erf',
]
args['share_weights'] = [True]
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

selem = np.zeros((5, 5))
selem = disk(2)
selems.append(selem)
# selem[np.arange(5), 5 - np.arange(1, 6)] = 1
# selem[np.arange(5), np.arange(5)] = 1
# selem[:, 2] = 1
# selem[2, :] = 1
# selem = disk(2)

args['selem'] = selems
