import os
from os.path import join
import re

from tqdm import tqdm
from deep_morpho_old.save_results_template.load_args import load_args


def list_dir_joined(folder: str):
    return [os.path.join(folder, k) for k in os.listdir(folder)]



for model in ['bisel', 'dual_bisel', 'sybisel']:
    for dataset in [
            'mnist',
            'diskorect',
            'inverted_mnist',
            'mnist_gray',
            'fashionmnist',
        ]:
        seen_args = []

        path_global = f"deep_morpho_old/results/results_tensorboards/Bimonn_exp_75/multi/0/{model}/softplus/{dataset}/"
        all_paths = []
        for operation in tqdm(os.listdir(path_global)):
            if not os.path.isdir(join(path_global, operation)):
                continue
            for selem in os.listdir(join(path_global, operation)):
                for version in os.listdir(join(path_global, operation, selem)):
                    args = load_args(join(path_global, operation, selem, version, "args.yaml"))
                    seen_args.append((operation, selem, args["loss_data_str"], args["learning_rate"],
                                    args["optimizer"], args["bias_optim_mode"],))
                    # seen_args.append((operation, selem, args["init_weight_mode"], args["bias_optim_mode"], args["loss_data_str"], args["learning_rate"]))
        print(model, dataset, len(seen_args))
        with open(f"{path_global}/seen_args.txt", "w") as f:
            print(*seen_args, sep="\n", file=f)
