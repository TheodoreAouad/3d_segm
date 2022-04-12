import os
from deep_morpho.save_results_template.save_html import write_html_deep_morpho
from deep_morpho.save_results_template.display_results import DisplayResults
import webbrowser


def list_dir_joined(folder: str):
    return [os.path.join(folder, k) for k in os.listdir(folder)]



TB_PATHS = (
    # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/dilation_size_7x7_bise')) +
    # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/erosion_size_7x7_bise')) +
    # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/opening_size_7x7_bise')) +
    # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_32/closing_size_7x7_bise'))
    # sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_50/opening_bisel'))
    # sum([sorted(list_dir_joined(f'deep_morpho/results/Bimonn_exp_50/sandbox/4/diskorect/{op}/')) \
    #     for op in ['dilation', 'erosion', 'opening', 'closing', "black_tophat", "white_tophat"]], start=[]) +
    # sum([sorted(list_dir_joined(f'deep_morpho/results/Bimonn_exp_51/sandbox/0/diskorect/{op}/')) \
    #     for op in ['dilation', 'erosion', 'opening', 'closing', "black_tophat", "white_tophat"]], start=[]) +
    sum([sorted(list_dir_joined(f'deep_morpho/results/DGMM_2022/sandbox/0/softplus/diskorect/{op}')) for op in ['dilation', 'erosion']], start=[]) +
    sorted(list_dir_joined(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/opening')) +
    [f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/closing/version_2'] +
    [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/black_tophat/version_{nb}') for nb in [0, 5, 10]] +
    [(f'deep_morpho/results/DGMM_2022/sandbox/1/softplus/diskorect/white_tophat/version_{nb}') for nb in [18, 19, 2]] +
    # sum([sorted(list_dir_joined(f'deep_morpho/results/ICIP_2022/sandbox/5/inverted_mnist/{op}/bisel')) for op in ['dilation', 'erosion', 'opening', 'closing']], start=[]) +
    # sum([sorted(list_dir_joined(f'deep_morpho/results/ICIP_2022/sandbox/5/mnist/{op}/bisel')) for op in ['dilation', 'erosion', 'opening', 'closing']], start=[])
    []
)

SAVE_PATH = 'html_pages/dgmm_2022_diskorect.html'
TITLE = 'test_page'

# Version BiSES
# html = write_html_deep_morpho(TB_PATHS, SAVE_PATH, TITLE)

# Version BiSEls
html = DisplayResults().save(TB_PATHS, SAVE_PATH, TITLE)


webbrowser.open(SAVE_PATH, new=1)
