from pathlib import Path
import re


def get_tb_paths_recompute():
    path_global = Path(
        "deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/recompute_projected"
    )
    all_paths = []
    for batch_exp in path_global.iterdir():
        for dataset in batch_exp.iterdir():
            if not dataset.is_dir():
                continue
            if dataset.name == "code":
                continue
            for operation in (dataset / "bimonn").iterdir():
                for selem in operation.iterdir():
                    all_paths += sorted([str(p) for p in selem.iterdir()], key=lambda x: int(
                        re.findall(r'version_(\d+)$', x)[0]
                    ))
    return all_paths


def get_tb_paths_multi(model, dataset):
    path_global = Path(
        f"deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/0/{model}/softplus/{dataset}/"
    )
    all_paths = []
    for operation in path_global.iterdir():
        if not operation.is_dir():
            continue
        for selem in operation.iterdir():
            all_paths += sorted([str(p) for p in selem.iterdir()], key=lambda x: int(
                re.findall(r'version_(\d+)$', x)[0]
            ))
    return all_paths


models_type = [
    'bisel',
    'dual_bisel'
]
datasets = [
    'diskorect',
    'mnist',
    'inverted_mnist',
    'mnist_gray',
    'fashionmnist'
]

all_paths = []
for model in models_type:
    for dataset in datasets:
        all_paths += get_tb_paths_multi(model, dataset)
all_paths += get_tb_paths_recompute()