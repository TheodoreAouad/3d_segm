import os
from os.path import join
from pathlib import Path
import re

import pandas as pd
import torch.nn as nn
from tqdm import tqdm

import deep_morpho.models as models
import deep_morpho.observables as obs
from deep_morpho.experiments.experiment_classification import ExperimentClassificationChannel
from deep_morpho.experiments.enforcers import ArgsClassifChannel, ArgsMnist
from deep_morpho.experiments.parser import Parser
from deep_morpho.datasets.gray_to_channels_dataset import LevelsetValuesEqualIndex
from deep_morpho.save_results_template.display_results import DisplayResults



def list_dir_joined(folder: str):
    return [os.path.join(folder, k) for k in os.listdir(folder)]


TB_PATHS = []

path_global = Path("deep_morpho/results/results_tensorboards/Bimonn_exp_80/sandbox/positive_weights/0_")
for exp_nb in ([] +
    ["0"] +
    [f"{i}" for i in range(5, 7)] +
[]):
    # for version in list_dir_joined(join(path_global, exp_nb, "mnistclassifchannel", "BimonnDenseNotBinary")):
    for version in (path_global / f"{exp_nb}" / "mnistclassifchannel" / "BimonnDenseNotBinary").iterdir():
        TB_PATHS.append(str(version))

path_global = Path("deep_morpho/results/results_tensorboards/Bimonn_exp_80/ruche/positive_weights/0_")
for exp_nb in range(8):
    # for version in list_dir_joined(join(path_global, str(exp_nb), "mnistclassifchannel", "BimonnDenseNotBinary")):
    for version in (path_global / f"{exp_nb}" / "mnistclassifchannel" / "BimonnDenseNotBinary").iterdir():
        TB_PATHS.append(str(version))


# df, global_args, changing_args = DisplayResults().get_df_from_tb_paths(TB_PATHS)
# df["test_accuracy"] = df["test_accuracy"].astype(float)
# df["binary_test_accuracy"] = df["binary_test_accuracy"].astype(float)
# df["test_error"] = 1 - df["test_accuracy"].astype(float)
# df["binary_test_error"] = 1 - df["binary_test_accuracy"].astype(float)
# df["float_params"] = df["float_params"].astype(float)


# df.loc[pd.isna(df["kwargs_loss_regu"]), "kwargs_loss_regu"] = "None"
# df.loc[df["loss_regu_str"] == "None", "kwargs_loss_regu"] = "None"
# df["loss_coef_regu"] = df["loss_coef_regu"].apply(lambda x: x[1])
# df.loc[df["loss_coef_regu"] == "None", "loss_coef_regu"] = "0"

# df.loc[(df["kwargs_loss_regu"] == "None") & (df["loss_regu_str"] == "RegularizationProjConstant"), "kwargs_loss_regu"] = "exact"

# df["loss_regu_delay"] = df["loss_regu_delay"].astype(int)

# for col in ["val_accuracy", "test_accuracy", "binary_val_accuracy", "binary_test_accuracy"]:
#     df[col] = df[col].astype(float)


# df["float_diff_val_test"] = df["val_accuracy"] - df["test_accuracy"]
# df["binary_diff_val_test"] = df["binary_val_accuracy"] - df["binary_test_accuracy"]


def load_args(tb_path):
    args = Parser()
    # args["batch_seed"] = 2249939862
    seed = get_seed(tb_path)
    args["batch_seed"] = seed
    args["dataset"] = "mnistclassifchanneldataset"
    args["model"] = "bimonndensenotbinary"
    args["batch_size"] = 128
    args["num_workers"] = 5
    args["freq_scalars"] = 50
    args["freq_imgs"] = None
    args["freq_hist"] = None
    args["freq_update_binary_batch"] = None
    args["freq_update_binary_epoch"] = None
    args["patience_reduce_lr"] = None
    args["patience_loss"] = None
    args['channel_classif_args'] = {
        "levelset_handler_mode": LevelsetValuesEqualIndex,
        # "levelset_handler_args": {"n_values": 10},
        "levelset_handler_args": {"n_values": 1},
    }

    args.parse_args(add_argv=False)
    return args


def load_model(tb_path):
    if os.path.exists(join(tb_path, 'best_weights')):
        tb_path = join(tb_path, 'best_weights', os.listdir(join(tb_path, 'best_weights'))[0])
    elif os.path.exists(join(tb_path, 'checkpoints')):
        tb_path = join(tb_path, 'checkpoints', os.listdir(join(tb_path, 'checkpoints'))[0])

    # bise_base.BiseBiasOptimEnum[df["bias_optim_mode"].iloc[0]]

    loss = nn.BCELoss()
    model = models.LightningBiMoNN.load_from_checkpoint(tb_path, loss=loss)
    model.model.binary(True)
    model.model.binary("partial")
    return model


def load_mnist_exp(tb_path):
    args = load_args(tb_path)
    model = load_model(tb_path)

    exp = ExperimentClassificationChannel(
        args=args,
        dest_dir="tmp/",
        verbose=False
    )
    exp.model = model
    exp.args_enforcers = [ArgsClassifChannel(), ArgsMnist()]
    exp.setup()

    obs_binary_partial_metrics = obs.BinaryPartialModeMetricClassifChannel(
        metrics={"accuracy": exp.metric_binary_obs.metrics["accuracy"]},
        freq={"train": 1, "val": 1, "test": 1},
        plot_freq={"train": 1, "val": 1, "test": 100000},
        dataset=exp.trainloader.dataset,
    )
    exp.observables.append(obs_binary_partial_metrics)
    model.observables = exp.observables
    return exp, obs_binary_partial_metrics


def get_seed(tb_path):
    with open(Path(tb_path) / "args.yaml", "r") as f:
        return int(parse_yaml_dict_key_line(f.read(), "seed"))


def regex_find_or_none(regex: str, st: str, *args, group_nb: int = -1, **kwargs,):
    exps = re.findall(regex, st, *args, **kwargs)
    if len(exps) == 0:
        return None
    
    if len(exps) > 1:
        exps = [exps[0]]
    # assert len(exps) == 1, exps

    # for multiple parenthesis, we have to select the group. If there is only one group, -1
    if group_nb == -1:
        return exps[0]
    return exps[0][group_nb]


def parse_yaml_dict_key_line(yaml_str: str, key: str):
    return regex_find_or_none(f"( |^|\n){key}: ([^\n]+)\n", yaml_str, group_nb=1)




def evaluate_test(tb_path: str):
    # model = load_model(tb_path)
    exp, obs_binary_partial_metrics = load_mnist_exp(tb_path)

    exp.test()


    binary_acc = exp.metric_binary_obs.last_value["test"]["accuracy"]
    float_acc = exp.metric_float_obs.last_value["test"]["accuracy"]
    binary_partial_acc = obs_binary_partial_metrics.last_value["test"]["accuracy"]

    return (exp, obs_binary_partial_metrics), (binary_acc, float_acc, binary_partial_acc)


savepath = Path("deep_morpho/results/exp80_mnist_results_df")

for tb_path in tqdm(TB_PATHS):    
    exp_id = tb_path.replace("/", "__")
    if (savepath / exp_id).exists():
        continue
    (savepath / exp_id).mkdir(exist_ok=True, parents=True)
    (exp, obs_binary_partial_metrics), (binary_acc, float_acc, binary_partial_acc) = evaluate_test(tb_path)

    for obs_ in exp.observables:
        obs_.save(savepath / exp_id)





# print("Binary Error:", f"{1 - binary_acc:.4f}")
# print("Binary Partial Error:", f"{1 - binary_partial_acc:.4f}")
# print("Float Error:", f"{1 - float_acc:.4f}")
