import pandas as pd
import numpy as np

PATH_METRICS = "deep_morpho/metric_results/DGMM_2022.csv"
# TEMPLATE_PATH = 'paper_writer/deep_morpho/template_erodila_dgmm.txt'
# TEMPLATE_PATH = 'paper_writer/deep_morpho/template_opeclos_dgmm.txt'
TEMPLATE_PATH = 'paper_writer/deep_morpho/template_tophat_dgmm.txt'
# TEMPLATE_PATH = 'paper_writer/deep_morpho/template_opeclos.txt'
assert 'template' in TEMPLATE_PATH

df = pd.read_csv(PATH_METRICS)

with open(TEMPLATE_PATH, 'r') as f:
    template = f.read()

for idx in range(len(df)):
    dataset = df['dataset'].iloc[idx]
    op = df['op'].iloc[idx]
    selem = df['selem'].iloc[idx]
    dice = df['dice'].iloc[idx]
    dice_binary = df['dice_binary'].iloc[idx]

    for selem_idx in range(2):
        conv_selem = np.isnan(df[f'convergence_selem_{selem_idx}'].iloc[idx])
        # symbol = r"\textcolor{red}{$\times$}" if conv_selem else r"\textcolor{green}{\checkmark}"
        symbol = "ko" if conv_selem else "ok"
        template = template.replace(f'CONVERGED{selem_idx}_{dataset}_{op}_{selem}', symbol)

    template = template.replace(f'ERROR_{dataset}_{op}_{selem}', f'{1 - dice:.3f}')
    template = template.replace(f'ERROR_binary_{dataset}_{op}_{selem}', f'{1 - dice_binary:.3f}')

    template = template.replace(f'DICE_{dataset}_{op}_{selem}', f'{dice:.2f}')
    template = template.replace(f'DICE_binary_{dataset}_{op}_{selem}', f'{dice_binary:.2f}')



with open(TEMPLATE_PATH.replace('template', 'table'), 'w') as f:
    print(template, file=f)
