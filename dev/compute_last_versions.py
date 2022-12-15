import re
import os
from os.path import join

from general.utils import save_json


experiment_multi_path: str = "deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/0/bisel/softplus"
otp_path = "last_versions.json"

all_versions = {}

for dataset in os.listdir(experiment_multi_path):
    if not os.path.isdir(join(experiment_multi_path, dataset)):
        continue
    for operation in os.listdir(join(experiment_multi_path, dataset)):
        if not os.path.isdir(join(experiment_multi_path, dataset, operation)):
            continue
        for selem in os.listdir(join(experiment_multi_path, dataset, operation)):
            if not os.path.isdir(join(experiment_multi_path, dataset, operation, selem)):
                continue

            max_version = 0
            for version in os.listdir(join(experiment_multi_path, dataset, operation, selem)):
                if not os.path.isdir(join(experiment_multi_path, dataset, operation, selem, version)):
                    continue

                cur_version = int(re.findall(r'version_(\d+)', version)[0])
                if cur_version > max_version:
                    max_version = cur_version
            all_versions[str((dataset, operation, selem))] = max_version

save_json(all_versions, otp_path)
