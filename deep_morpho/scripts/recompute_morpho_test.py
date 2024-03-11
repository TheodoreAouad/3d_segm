import re
import os
from os.path import join
from pathlib import Path
import os
import time

from os.path import join
from tqdm import tqdm
from importlib import import_module
import pytorch_lightning as pl
import deep_morpho.models.lightning_bimonn as lbimonn
from deep_morpho.datasets import DataModule
from deep_morpho.save_results_template.load_args import load_args
from deep_morpho.morp_operations import ParallelMorpOperations, ParallelMorpOperationsGrayTopHats
from deep_morpho.metrics import symetric_dice, dice
from general.utils import save_json
import logging

models_old = import_module("deep_morpho.results.results_tensorboards.Bimonn_exp_75.multi.0.bisel.softplus.diskorect.black_tophat.disk.version_0.code.deep_morpho.models.lightning_bimonn")

t1 = time.time()

def list_dir_joined(folder: str):
    return [os.path.join(folder, k) for k in os.listdir(folder)]


def get_tb_paths_multi(model, dataset):
    path_global = f"deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/0/{model}/softplus/{dataset}/"
    all_paths = []
    for operation in os.listdir(path_global):
        if not os.path.isdir(join(path_global, operation)):
            continue
        for selem in os.listdir(join(path_global, operation)):
            all_paths += sorted(list_dir_joined(join(path_global, operation, selem)), key=lambda x: int(
                re.findall(r'version_(\d+)$', x)[0]
            ))
    return all_paths

def get_tb_paths_recompute():
    path_global = Path(
        "deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/recompute_projected"
    )
    all_paths = []
    for batch_exp in path_global.iterdir():
        for dataset in batch_exp.iterdir():
            if not dataset.is_dir():
                continue
            if dataset.name == "code":
                continue
            for operation in (dataset / "bimonn").iterdir():
                for selem in operation.iterdir():
                    all_paths += sorted([str(p) for p in selem.iterdir()], key=lambda x: int(
                        re.findall(r'version_(\d+)$', x)[0]
                    ))
    return all_paths


models_type = [
    'bisel',
    'dual_bisel'
]
datasets = [
    'diskorect',
    'mnist',
    'inverted_mnist',
    'mnist_gray',
    'fashionmnist'
]

all_paths = []
all_paths += get_tb_paths_recompute()
for model in models_type:
    for dataset in datasets:
        all_paths += get_tb_paths_multi(model, dataset)



def check_is_computed(tb_path):
    return os.path.exists(join(tb_path, "observables", "RecomputeMetrics", "metrics.json"))



def load_model(tb_path):
    try:
        # get the file epoch= from tb_path / best_weights
        path_weights = Path(tb_path) / "best_weights"
        path_weights = path_weights / [f for f in os.listdir(path_weights) if "epoch=" in f][0]
        model = lbimonn.LightningBiMoNN.load_from_checkpoint(path_weights, loss=lambda x: 1)
    except:
        path_weights = join(tb_path, "checkpoints", os.listdir(join(tb_path, "checkpoints"))[0])
        model = models_old.LightningBiMoNN.load_from_checkpoint(path_weights)
    return model

def run_test(tb_path):


    args = load_args(join(tb_path, "args.yaml"))
    operation, selem = args["name"].split("/")
    size = args["kernel_size"] if args["kernel_size"] is not None else args["kernel_size.net"]

    if "tophat_gray" in operation:
        morp_op_class = ParallelMorpOperationsGrayTopHats
    else:
        morp_op_class = ParallelMorpOperations

    morp_op = getattr(morp_op_class, operation)((selem, int(size)), name=args["name"])

    datasets_format = {
        "diskorect": "diskorectdataset",
        "mnist": "mnistmorphodataset",
        "mnistmorpho": "mnistmorphodataset",
        "inverted_mnist": "invertedmnistmorphodataset",
        "invertedmnistmorpho": "invertedmnistmorphodataset",
        "mnist_gray": "mnistgrayscaledataset",
        "mnistgraymorpho": "mnistgrayscaledataset",
        "fashionmnist": "fashionmnistgrayscaledataset",
    }


    dataset = datasets_format[args["dataset_type"]]

    dataloader = DataModule.select(dataset).get_loader(
        batch_size=256,
        n_inputs=50000,
        morp_operation=morp_op,
        shuffle=True
    )

    metrics = {
        "recompute_dice": dice, "symetric_dice": symetric_dice
    }

    results = {}
    model = load_model(tb_path).cuda()
    for name, metric in metrics.items():
        inpt, target = next(iter(dataloader))
        inpt, target = inpt.cuda(), target.cuda()
        model.model.binary(False)
        otp = model(inpt)
        results[name] = metric(target, otp).mean()

        model.model.binary()
        otp_bin = model(inpt)
        results["binary_mode_" + name] = metric(target, otp_bin).mean()

        results["baseline_" + name] = metric(target, inpt).mean()

    for k, v in results.items():
        results[k] = f"{v}"

    savepath = Path(tb_path) / "observables" / "RecomputeMetrics" / "metrics.json"
    savepath.parent.mkdir(exist_ok=True, parents=True)
    save_json(results, savepath)


bugs = []

for path in tqdm(all_paths[::-1]):
    if check_is_computed(path):
        continue
    try:
        run_test(path)
    except:
        logging.exception(f"Error in {path}")
        bugs.append(path)

print("Bugs:")
print(bugs)

print(f"Finished in {time.time() - t1} seconds.")
