print("Import...")
import os
from os.path import join
import pathlib
import re
from deep_morpho.save_results_template.display_results import DisplayResults
import webbrowser

from tqdm import tqdm


def list_dir_joined(folder: str):
    return [os.path.join(folder, k) for k in os.listdir(folder)]


TB_PATHS = []
SAVE_PATH = []

# TB_PATHS.append(
    # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/dilation_size_7x7_bise')) +
    # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/erosion_size_7x7_bise')) +
    # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/opening_size_7x7_bise')) +
    # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/closing_size_7x7_bise'))
    # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_50/opening_bisel'))
    # sum([sorted(list_dir_joined(f'deep_morpho/results/Bimonn_exp_50/sandbox/4/diskorect/{op}/')) \
    #     for op in ['dilation', 'erosion', 'opening', 'closing', "black_tophat", "white_tophat"]], start=[]) +
    # sum([sorted(list_dir_joined(f'deep_morpho/results/Bimonn_exp_51/sandbox/0/diskorect/{op}/')) \
    #     for op in ['dilation', 'erosion', 'opening', 'closing', "black_tophat", "white_tophat"]], start=[]) +
    # sum([sorted(list_dir_joined(f'deep_morpho/results/ICIP_2022/sandbox/5/inverted_mnist/{op}/bisel')) for op in ['dilation', 'erosion', 'opening', 'closing']], start=[]) +
    # sum([sorted(list_dir_joined(f'deep_morpho/results/ICIP_2022/sandbox/5/mnist/{op}/bisel')) for op in ['dilation', 'erosion', 'opening', 'closing']], start=[])
#     []
# )

print('Parsing paths...')

# path_global = "deep_morpho/results/results_tensorboards/Bimonn_mega_multi_1/softplus/diskorect/"
# all_paths = []
# for operation in os.listdir(path_global):
#     if not os.path.isdir(join(path_global, operation)):
#         continue
#     for selem in os.listdir(join(path_global, operation)):
#         all_paths += sorted(list_dir_joined(join(path_global, operation, selem)), key=lambda x: int(
#             re.findall(r'version_(\d+)$', x)[0]
#         ))
# TB_PATHS.append(
#     all_paths
# )
# SAVE_PATH.append('html_pages/diskorect_grid_search.html')

# path_global = "deep_morpho/results/results_tensorboards/Bimonn_mega_multi_1/softplus/mnist/"
# all_paths = []
# for operation in os.listdir(path_global):
#     for selem in os.listdir(join(path_global, operation)):
#         all_paths += sorted(list_dir_joined(join(path_global, operation, selem)), key=lambda x: int(
#             re.findall(r'version_(\d+)$', x)[0]
#         ))
# TB_PATHS.append(
#     all_paths
# )
# SAVE_PATH.append('html_pages/mnist_grid_search.html')

for model in [
    'bisel',
    # 'dual_bisel',
    # 'sybisel',
]:
    for dataset in [
        'diskorect',
        'mnist',
        'inverted_mnist',
        'mnist_gray'
    ]:
        path_global = f"deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/0/{model}/softplus/{dataset}/"
        all_paths = []
        for operation in os.listdir(path_global):
            if not os.path.isdir(join(path_global, operation)):
                continue
            for selem in os.listdir(join(path_global, operation)):
                all_paths += sorted(list_dir_joined(join(path_global, operation, selem)), key=lambda x: int(
                    re.findall(r'version_(\d+)$', x)[0]
                ))
        TB_PATHS.append(
            all_paths
        )
        SAVE_PATH.append(f'html_pages/75_{model}_{dataset}.html')



# path_global = "deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/0/bisel/softplus/diskorect/"
# all_paths = []
# for operation in os.listdir(path_global):
#     if not os.path.isdir(join(path_global, operation)):
#         continue
#     for selem in os.listdir(join(path_global, operation)):
#         all_paths += sorted(list_dir_joined(join(path_global, operation, selem)), key=lambda x: int(
#             re.findall(r'version_(\d+)$', x)[0]
#         ))
# TB_PATHS.append(
#     all_paths
# )
# SAVE_PATH.append('html_pages/75_diskorect.html')


# path_global = "deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/0/bisel/softplus/mnist_gray/"
# all_paths = []
# for operation in os.listdir(path_global):
#     if not os.path.isdir(join(path_global, operation)):
#         continue
#     for selem in os.listdir(join(path_global, operation)):
#         all_paths += sorted(list_dir_joined(join(path_global, operation, selem)), key=lambda x: int(
#             re.findall(r'version_(\d+)$', x)[0]
#         ))
# TB_PATHS.append(
#     all_paths
# )
# SAVE_PATH.append('html_pages/75_mnist_gray.html')


# path_global = "deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/0/bisel/softplus/mnist/"
# all_paths = []
# for operation in os.listdir(path_global):
#     if not os.path.isdir(join(path_global, operation)):
#         continue
#     for selem in os.listdir(join(path_global, operation)):
#         all_paths += sorted(list_dir_joined(join(path_global, operation, selem)), key=lambda x: int(
#             re.findall(r'version_(\d+)$', x)[0]
#         ))
# TB_PATHS.append(
#     all_paths
# )
# SAVE_PATH.append('html_pages/75_mnist.html')

# path_global = "deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/0/bisel/softplus/inverted_mnist/"
# all_paths = []
# for operation in os.listdir(path_global):
#     if not os.path.isdir(join(path_global, operation)):
#         continue
#     for selem in os.listdir(join(path_global, operation)):
#         all_paths += sorted(list_dir_joined(join(path_global, operation, selem)), key=lambda x: int(
#             re.findall(r'version_(\d+)$', x)[0]
#         ))
# TB_PATHS.append(
#     all_paths
# )
# SAVE_PATH.append('html_pages/75_inverted_mnist.html')

# path_global = "/hdd/aouadt/these/projets/3d_segm/deep_morpho/results/results_tensorboards/Bimonn_exp_63/multi/sybisel/softplus/inverted_mnist"
# all_paths = []
# for operation in os.listdir(path_global):
#     if not os.path.isdir(join(path_global, operation)):
#         continue
#     for selem in os.listdir(join(path_global, operation)):
#         all_paths += sorted(list_dir_joined(join(path_global, operation, selem)), key=lambda x: int(
#             re.findall(r'version_(\d+)$', x)[0]
#         ))
# TB_PATHS.append(
#     all_paths
# )
# SAVE_PATH.append('html_pages/63_inverted_mnist.html')

# DGMM 2022 Diskorect
# TB_PATHS.append(
#     # sum([sorted(list_dir_joined(f'deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/{op}')) for op in ['dilation', 'erosion']], start=[]) +
#     # sorted(list_dir_joined(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/opening')) +
#     # [f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/closing/version_2'] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/dilation/version_{nb}') for nb in [0, 1, 2]] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/erosion/version_{nb}') for nb in [0, 1, 2]] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/opening/version_{nb}') for nb in [0, 1, 2]] +
#     [f'deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/closing/version_0'] +
#     [f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/closing/version_2'] +
#     [f'deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/closing/version_2'] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/black_tophat/version_{nb}') for nb in [0, 5, 10]] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/white_tophat/version_{nb}') for nb in [18, 19, 2]] +
#     []
# )
# SAVE_PATH.append('html_pages/dgmm_2022_diskorect.html')

# DGMM 2022 Inverted Mnist
# TB_PATHS.append(
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/dilation/version_{nb}') for nb in [0, 1, 2]] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/erosion/version_{nb}') for nb in [0, 1, 2]] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/opening/version_{nb}') for nb in [3, 1, 4]] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/closing/version_{nb}') for nb in [5, 4, 2]] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/black_tophat/version_{nb}') for nb in [0, 1, 2]] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/inverted_mnist/white_tophat/version_{nb}') for nb in [0, 1, 2]] +
#     []
# )
# SAVE_PATH.append('html_pages/dgmm_2022_inverted_mnist.html')

# DGMM 2022 Mnist
# TB_PATHS.append(
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/dilation/version_{nb}') for nb in [0, 1, 2]] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/erosion/version_{nb}') for nb in [0, 1, 2]] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/opening/version_{nb}') for nb in [4, 1, 2]] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/closing/version_{nb}') for nb in [0, 9, 6]] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/black_tophat/version_{nb}') for nb in [0, 1, 2]] +
#     [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/mnist/white_tophat/version_{nb}') for nb in [0, 1, 2]] +
#     []
# )
# SAVE_PATH.append('html_pages/dgmm_2022_mnist.html')



# Version BiSES
# html = write_html_deep_morpho(TB_PATHS, SAVE_PATH, TITLE)

print('Loading HTML web pages...')

for tb, savepath in tqdm(zip(TB_PATHS, SAVE_PATH)):
    title = str(pathlib.Path(savepath).stem)
    html = DisplayResults().save(tb, savepath, title, show_table=True, show_details=False)
    webbrowser.open(savepath, new=0)

print(SAVE_PATH)
