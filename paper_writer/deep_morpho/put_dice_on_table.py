import pandas as pd
import numpy as np

PATH_METRICS = "deep_morpho/metric_results/results.csv"
# TEMPLATE_PATH = 'paper_writer/deep_morpho/template_erodila.txt'
TEMPLATE_PATH = 'paper_writer/deep_morpho/template_opeclos.txt'
assert 'template' in TEMPLATE_PATH

df = pd.read_csv(PATH_METRICS)

with open(TEMPLATE_PATH, 'r') as f:
    template = f.read()

for idx in range(len(df)):
    dataset = df['dataset'].iloc[idx]
    operation = df['operation'].iloc[idx]
    selem = df['selem'].iloc[idx]
    dice = df['dice'].iloc[idx]

    for selem_idx in range(2):
        conv_selem = np.isnan(df[f'convergence_selem_{selem_idx}'].iloc[idx])
        symbol = r"\textcolor{red}{$\times$}" if conv_selem else r"\textcolor{green}{\checkmark}"
        template = template.replace(f'CONVERGED{selem_idx}_{dataset}_{operation}_{selem}', symbol)

    template = template.replace(f'ERROR_{dataset}_{operation}_{selem}', f'{1 - dice:.3f}')




with open(TEMPLATE_PATH.replace('template', 'table'), 'w') as f:
    print(template, file=f)
