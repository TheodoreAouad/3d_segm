from os import write
from deep_morpho.save_results_template.save_html import write_html_deep_morpho


TB_PATHS = [
    'deep_morpho/results/Test_save/dilation_bise/version_4',
    'deep_morpho/results/Test_save/dilation_bise/version_4',
    'deep_morpho/results/Test_save/dilation_bise/version_4',
]

SAVE_PATH = 'html_pages/page0.html'
TITLE = 'test_page'

write_html_deep_morpho(TB_PATHS, SAVE_PATH, TITLE)
