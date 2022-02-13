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
    sorted(list_dir_joined('deep_morpho/results/Bimonn_exp_46/opening_bisel'))
)

SAVE_PATH = 'html_pages/page0.html'
TITLE = 'test_page'

# Version BiSES
# html = write_html_deep_morpho(TB_PATHS, SAVE_PATH, TITLE)

# Version BiSEls
html = DisplayResults().save(TB_PATHS, SAVE_PATH, TITLE)


webbrowser.open(SAVE_PATH, new=1)
