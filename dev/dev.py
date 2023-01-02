import re
import os
from os.path import join
import pathlib

import tqdm
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

from general.utils import save_json


def extract_last_value_from_tb(path_tb_file: str) -> float:
    ea = event_accumulator.EventAccumulator(
        path_tb_file,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    if len(ea.Tags()['scalars']) == 0:
        return None
    return ea.Scalars(ea.Tags()['scalars'][0])[-1].value


big_root = "deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/0/"
# experiment_multi_path: str = bisel/softplus"
otp_path = "last_versions.json"

all_versions = {}

total_created = 0

bar_new = tqdm.tqdm(desc="added")
bar_exists = tqdm.tqdm(desc="exists")
bar_total = tqdm.tqdm(desc='total')

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
                    tb_path = join(experiment_multi_path, dataset, operation, selem, version)
                    if not os.path.isdir(tb_path):
                        continue

                    bar_total.update(1)
                    obs_path = join(tb_path, "observables", "SaveLoss")
                    if os.path.exists(obs_path):
                        bar_exists.update(1)
                        continue

                    # print(obs_path)
                    # assert False
                    pathlib.Path(obs_path).mkdir(exist_ok=True, parents=True)
                    res = {
                        'loss': extract_last_value_from_tb(join(tb_path, "loss_train_loss")),
                        'loss_data': extract_last_value_from_tb(join(tb_path, "loss_train_loss_data")),
                    }

                    save_json(res, join(obs_path, "last_loss.json"))

                    total_created += 1
                    bar_new.update(1)
                    # cur_version = int(re.findall(r'version_(\d+)', version)[0])
                    # if cur_version > max_version:
                    #     max_version = cur_version
                # all_versions[str((model, dataset, operation, selem))] = max_version

# save_json(all_versions, otp_path)
print(total_created)