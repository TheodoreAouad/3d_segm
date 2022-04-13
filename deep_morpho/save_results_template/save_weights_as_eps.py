import pathlib
from PIL import Image
from os.path import join
import os

import numpy as np


from general.utils import list_dir_joined
from deep_morpho.models import LightningBiMoNN


# TB_PATHS = (
#     # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/dilation_size_7x7_bise')) +
#     # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/erosion_size_7x7_bise')) + 
#     # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/opening_size_7x7_bise')) +
#     # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/closing_size_7x7_bise')) 
#     # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_46/opening_bisel'))
#     sum([sorted(list_dir_joined(f'deep_morpho/results/ICIP_2022/sandbox/4/diskorect/{op}/bisel')) for op in ['dilation', 'erosion', 'opening', 'closing']], start=[]) +
#     sum([sorted(list_dir_joined(f'deep_morpho/results/ICIP_2022/sandbox/5/inverted_mnist/{op}/bisel')) for op in ['dilation', 'erosion', 'opening', 'closing']], start=[]) +
#     sum([sorted(list_dir_joined(f'deep_morpho/results/ICIP_2022/sandbox/5/mnist/{op}/bisel')) for op in ['dilation', 'erosion', 'opening', 'closing']], start=[])
# )

PATH_OUT = "deep_morpho/weights_eps/dgmm_2022"
pathlib.Path(PATH_OUT).mkdir(exist_ok=True, parents=True)


def save_img(ar, savepath):
    ar = np.uint8(ar  / ar.max() * 255)
    img = Image.fromarray(ar).resize((50, 50), resample=Image.NEAREST)
    img.save(savepath)


selems = ['disk', 'hstick', 'dcross']
tb_path_dict = dict(
    # DISKORECT
    **{str(('diskorect', 'dilation', selem)): f'deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/dilation/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('diskorect', 'erosion', selem)): f'deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/erosion/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('diskorect', 'opening', selem)): f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/opening/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('diskorect', 'closing', 'disk')): 'deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/closing/version_0'},
    **{str(('diskorect', 'closing', 'hstick')): 'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/closing/version_2'},
    **{str(('diskorect', 'closing', 'dcross')): 'deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/closing/version_2'},
    # [[('diskorect', 'black_tophat', selem): f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/black_tophat/version_{nb}' for selem, nb in zip(selems, [0, 5, 10])]},
    # [[('diskorect', 'white_tophat', selem): f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/white_tophat/version_{nb}' for selem, nb in zip(selems, [18, 19, 2])]},
    # MNIST
    **{str(('mnist', 'dilation', selem)): f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/dilation/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('mnist', 'erosion', selem)): f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/erosion/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('mnist', 'opening', selem)): f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/opening/version_{nb}' for selem, nb in zip(selems, [4, 1, 2])},
    **{str(('mnist', 'closing', selem)): f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/closing/version_{nb}' for selem, nb in zip(selems, [0, 9, 6])},
    # [[('mnist', 'black_tophat', selem): f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/black_tophat/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])]},
    # # [[('mnist', 'white_tophat', selem): f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/white_tophat/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])]},
    # INVERTED MNIST
    **{str(('inverted_mnist', 'dilation', selem)): f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/dilation/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('inverted_mnist', 'erosion', selem)): f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/erosion/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    **{str(('inverted_mnist', 'opening', selem)): f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/opening/version_{nb}' for selem, nb in zip(selems, [3, 1, 4])},
    **{str(('inverted_mnist', 'closing', selem)): f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/closing/version_{nb}' for selem, nb in zip(selems, [5, 4, 2])},
    # [[('inverted_mnist', 'black_tophat', selem):, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/black_tophat/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
    # [[('inverted_mnist', 'white_tophat', selem):, f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/white_tophat/version_{nb}' for selem, nb in zip(selems, [0, 1, 2])},
)

# DILATION
for dataset in ['diskorect', 'mnist', 'inverted_mnist']:
    for selem in ['disk', 'hstick', 'dcross']:
        for op in ['dilation', 'erosion']:
        # pathlib.Path(join(PATH_OUT, dataset, op)).mkdir(exist_ok=True, parents=True)

            tb_path = tb_path_dict[str((dataset, op, selem))]
            path_weights = os.listdir(join(tb_path, "checkpoints"))[0]

            model = LightningBiMoNN.load_from_checkpoint(join(tb_path, 'checkpoints', path_weights))

            save_img(
                model.model.layer1.normalized_weight.detach().cpu()[0, 0].numpy(),
                join(PATH_OUT, f"{dataset}_{op}_{selem}.eps")
            )


        for op in ['opening', 'closing']:
        # pathlib.Path(join(PATH_OUT, dataset, op)).mkdir(exist_ok=True, parents=True)

            tb_path = tb_path_dict[str((dataset, op, selem))]
            path_weights = os.listdir(join(tb_path, "checkpoints"))[0]

            model = LightningBiMoNN.load_from_checkpoint(join(tb_path, 'checkpoints', path_weights))

            save_img(
                model.model.layer1.normalized_weight.detach().cpu()[0, 0].numpy(),
                join(PATH_OUT, f"{dataset}_{op}_{selem}1.eps")
            )
            save_img(
                model.model.layer2.normalized_weight.detach().cpu()[0, 0].numpy(),
                join(PATH_OUT, f"{dataset}_{op}_{selem}2.eps")
            )
