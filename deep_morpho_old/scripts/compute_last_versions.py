import re
import os
from os.path import join

from general.utils import save_json


big_root = "deep_morpho_old/results/results_tensorboards/Bimonn_exp_75/multi/0/"
# experiment_multi_path: str = bisel/softplus"
otp_path = "last_versions.json"

all_versions = {}

for model in os.listdir(big_root):
    experiment_multi_path = join(big_root, model, "softplus")
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
                all_versions[str((model, dataset, operation, selem))] = max_version

save_json(all_versions, otp_path)
