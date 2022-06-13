import os
from os.path import join
import shutil
import re

from deep_morpho.save_results_template.load_args import load_args


def list_dir_joined(folder: str):
    return [os.path.join(folder, k) for k in os.listdir(folder)]


seen_args = []

path_global = "deep_morpho/results/results_tensorboards/Bimonn_mega_multi_1/softplus/"
all_paths = []

to_delete = []

empty_error = []
too_many_file = []
init_error = []

for dataset in os.listdir(path_global):
    for operation in os.listdir(join(path_global, dataset)):
        if not os.path.isdir(join(path_global, dataset, operation, )):
            continue
        for selem in os.listdir(join(path_global, dataset, operation)):
            for version in os.listdir(join(path_global, dataset, operation, selem)):
                if not os.path.isdir(join(path_global, dataset, operation, selem, version, "observables")):
                    version_path = join(path_global, dataset, operation, selem, version)
                    to_delete.append(version_path)

                    error_path = join(version_path, "error_logs.log")
                    if not os.path.exists(error_path):
                        empty_error.append(version_path)
                    else:
                        with open(error_path, "r") as f:
                            error_content = f.read()
                        if error_content == "":
                            empty_error.append(version_path)
                        elif "Too many open files" in error_content:
                            too_many_file.append(version_path)
                        elif "assert (new_weights >= 0).all()" in error_content:
                            init_error.append(version_path)
                    
                    


print(len(to_delete))

# with open("deep_morpho/results/results_tensorboards/Bimonn_mega_multi_1/softplus/diskorect/seen_args.txt", "w") as f:
#     print(*seen_args, sep="\n", file=f)
