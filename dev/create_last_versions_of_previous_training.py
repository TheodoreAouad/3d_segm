import os
import pathlib
from os.path import join

from general.utils import load_json


experiment_multi_path: str = "deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/0/bisel/softplus"
last_versions_path = "last_versions.json"

all_versions = load_json(last_versions_path)

for key in all_versions.keys():
    print(key)
    dataset, operation, selem = [s.replace("'", "").replace("(", "").replace(")", "") for s in key.split(', ')]

# for dataset in os.listdir(experiment_multi_path):
#     if not os.path.isdir(join(experiment_multi_path, dataset)):
#         continue
#     for operation in os.listdir(join(experiment_multi_path, dataset)):
#         if not os.path.isdir(join(experiment_multi_path, dataset, operation)):
#             continue
#         for selem in os.listdir(join(experiment_multi_path, dataset, operation)):
#             if not os.path.isdir(join(experiment_multi_path, dataset, operation, selem)):
#                 continue

    pathlib.Path(
        join(experiment_multi_path, dataset, operation, selem, f"version_{all_versions[key]}")
    ).mkdir(exist_ok=True, parents=True)
    print(join(experiment_multi_path, dataset, operation, selem, f"version_{all_versions[key]}"))
