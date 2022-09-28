import os
from os.path import join
import re

from tqdm import tqdm
from deep_morpho.save_results_template.load_args import load_args


def list_dir_joined(folder: str):
    return [os.path.join(folder, k) for k in os.listdir(folder)]



for dataset in ['diskorect']:
    seen_args = []

    path_global = f"deep_morpho/results/results_tensorboards/Bimonn_exp_63/multi/sybisel/softplus/{dataset}/"
    all_paths = []
    for operation in tqdm(os.listdir(path_global)):
        if not os.path.isdir(join(path_global, operation)):
            continue
        for selem in os.listdir(join(path_global, operation)):
            for version in os.listdir(join(path_global, operation, selem)):
                args = load_args(join(path_global, operation, selem, version, "args.yaml"))
                seen_args.append((operation, selem, args["loss_data_str"], args["learning_rate"]))
                # seen_args.append((operation, selem, args["init_weight_mode"], args["bias_optim_mode"], args["loss_data_str"], args["learning_rate"]))
    print(dataset, len(seen_args))
    with open(f"{path_global}/seen_args.txt", "w") as f:
        print(*seen_args, sep="\n", file=f)