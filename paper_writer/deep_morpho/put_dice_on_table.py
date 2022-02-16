from re import TEMPLATE
import pandas as pd

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
    template = template.replace(f'ERROR_{dataset}_{operation}_{selem}', f'{1 - dice:.3f}')

with open(TEMPLATE_PATH.replace('template', 'table'), 'w') as f:
    print(template, file=f)
