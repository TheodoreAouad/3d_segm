import os
from os.path import join
import re

from deep_morpho.save_results_template.load_args import load_args


def list_dir_joined(folder: str):
    return [os.path.join(folder, k) for k in os.listdir(folder)]


seen_args = []

path_global = "deep_morpho/results/results_tensorboards/Bimonn_mega_multi_1/softplus/diskorect/"
all_paths = []
for operation in os.listdir(path_global):
    for selem in os.listdir(join(path_global, operation)):
        for version in os.listdir(join(path_global, operation, selem)):
            args = load_args(join(path_global, operation, selem, version, "args.yaml"))
            seen_args.append((args["init_weight_mode"], args["bias_optim_mode"], args["loss_data_str"], args["learning_rate"]))

with open("deep_morpho/results/results_tensorboards/Bimonn_mega_multi_1/softplus/diskorect/seen_args.txt", "w") as f:
    print(*seen_args, sep="\n", file=f)
