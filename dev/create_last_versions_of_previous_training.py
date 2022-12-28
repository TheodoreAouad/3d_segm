import os
import pathlib
from os.path import join

from general.utils import load_json


big_root = "deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/0/"
# experiment_multi_path: str = "deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/0/bisel/softplus"
last_versions_path = "last_versions.json"

all_versions = load_json(last_versions_path)

for key in all_versions.keys():
    print(key)
    model, dataset, operation, selem = [s.replace("'", "").replace("(", "").replace(")", "") for s in key.split(', ')]

# for dataset in os.listdir(experiment_multi_path):
#     if not os.path.isdir(join(experiment_multi_path, dataset)):
#         continue
#     for operation in os.listdir(join(experiment_multi_path, dataset)):
#         if not os.path.isdir(join(experiment_multi_path, dataset, operation)):
#             continue
#         for selem in os.listdir(join(experiment_multi_path, dataset, operation)):
#             if not os.path.isdir(join(experiment_multi_path, dataset, operation, selem)):
#                 continue

    path_version = join(big_root, model, "softplus", dataset, operation, selem, f"version_{all_versions[key]}")
    pathlib.Path(path_version).mkdir(exist_ok=True, parents=True)
    # os.rmdir(join(big_root, model, dataset, operation, selem, f"version_{all_versions[key]}"))
    print(path_version)
