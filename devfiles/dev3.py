from general.utils import load_json
import os
import numpy as np

path_exp1 = "deep_morpho/results/results_tensorboards/Bimonn_exp_78/baseline/0/mnistclassical/mlpbatchnormclassical"
all_paths = os.listdir(path_exp1)
all_paths = sorted(all_paths, key=lambda x: int(x.split("_")[-1]))

metrics_3M = np.array(
    [load_json(os.path.join(path_exp1, path, "observables", "CalculateAndLogMetrics", "metrics.json"))["test"]["accuracy"] for path in all_paths[:10]]
).astype(float)

metrics_10M = np.array(
    [load_json(os.path.join(path_exp1, path, "observables", "CalculateAndLogMetrics", "metrics.json"))["test"]["accuracy"] for path in all_paths[10:]]
).astype(float)



path_exp2 = "deep_morpho/results/results_tensorboards/Bimonn_exp_78/baseline/1/mnistclassical/mlpbatchnormclassical"
all_paths = os.listdir(path_exp2)
all_paths = sorted(all_paths, key=lambda x: int(x.split("_")[-1]))

metrics_1M = np.array(
    [load_json(os.path.join(path_exp2, path, "observables", "CalculateAndLogMetrics", "metrics.json"))["test"]["accuracy"] for path in all_paths]
).astype(float)


print(1 - metrics_3M.mean(), metrics_3M.std())
print(1 - metrics_10M.mean(), metrics_10M.std())
print(1 - metrics_1M.mean(), metrics_1M.std())