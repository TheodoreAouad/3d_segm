import os
from os.path import join

from general.utils import load_json


experiment_multi_path: str = "deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/0/bisel/softplus"
last_versions_path = "last_versions.json"

all_versions = load_json(last_versions_path)

for dataset in os.listdir(experiment_multi_path):
    if not os.path.isdir(join(experiment_multi_path, dataset)):
        continue
    for operation in os.listdir(join(experiment_multi_path, dataset)):
        if not os.path.isdir(join(experiment_multi_path, dataset, operation)):
            continue
        for selem in os.listdir(join(experiment_multi_path, dataset, operation)):
            if not os.path.isdir(join(experiment_multi_path, dataset, operation, selem)):
                continue

            os.mkdir(join(experiment_multi_path, dataset, operation, selem, all_versions[str((dataset, operation, selem))]))
