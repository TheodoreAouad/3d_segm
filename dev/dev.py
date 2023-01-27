# DO NOT DELETE

import os
from os.path import join
from tqdm import tqdm

from tensorboard.backend.event_processing import event_accumulator


def extract_max_value_from_tb(path_tb_file: str, scalar: str) -> float:
    ea = event_accumulator.EventAccumulator(
        path_tb_file,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    if len(ea.Tags()['scalars']) == 0:
        return None
    return max(ea.Scalars(scalar), key=lambda x: x.value).value


exp_path = "deep_morpho/results/results_tensorboards/Bimonn_exp_76/multi/0/bisel/softplus/classif_mnist/"

tb_paths = sorted(os.listdir(exp_path))

all_maxs = {}

for tb_path in tqdm(tb_paths):
    all_maxs[tb_path] = extract_max_value_from_tb(join(exp_path, tb_path), scalar="binary_mode/metrics_epoch_mean/accuracy_val")

print({k: f"{v:.2f}" for k, v in all_maxs.items()})
print(max(all_maxs.items(), key=lambda x: x[1]))
