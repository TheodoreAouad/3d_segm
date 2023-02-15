# import warnings
# warnings.filterwarnings('error', message=".+leaf Tensor.+")
print('Import native libraries ...')
from functools import partial
from collections import OrderedDict
from time import time
from copy import deepcopy
import os
import math
from os.path import join
import pathlib
from importlib import import_module
import argparse

print('Import libraries...')
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('Import modules...')
from deep_morpho.datasets import (
    MnistMorphoDataset, MnistGrayScaleDataset, MnistClassifDataset, FashionMnistGrayScaleDataset,
    DiskorectDataset, AxspaROISimpleDataset, AxspaROISimpleDataset, CIFAR10Dataset, CIFAR100Dataset,
    MnistClassifChannelDataset, SticksNoisedGeneratorDataset
)
from deep_morpho.datasets.gray_to_channels_dataset import LevelsetValuesEqualIndex
# from deep_morpho.datasets.mnist_dataset import MnistMorphoDataset, MnistGrayScaleDataset, MnistClassifDataset
# from deep_morpho.datasets.fashionmnist_dataset import FashionMnistGrayScaleDataset
from deep_morpho.utils import set_seed
# from deep_morpho.datasets.generate_forms2 import get_random_diskorect
# from deep_morpho.datasets.generate_forms3 import get_random_rotated_diskorect
# from deep_morpho.datasets.diskorect_dataset import DiskorectDataset, MultiRectDataset
# from deep_morpho.datasets.axspa_roi_dataset import AxspaROISimpleDataset
# from deep_morpho.datasets.sticks_noised_dataset import AxspaROISimpleDataset
from deep_morpho.models import (
    LightningBiMoNN, BiSE, BiseWeightsOptimEnum, LightningBiMoNNClassifierMaxPool,
    LightningBiMoNNClassifierMaxPoolNotBinary, LightningBiMoNNClassifierLastLinearNotBinary,
    LightningBiMoNNClassifierLastLinear, LightningBimonnDense, LightningBimonnDenseNotBinary,
    LightningBimonnBiselDenseNotBinary
)  # COBiSE, BiSEC, COBiSEC
import deep_morpho.observables as obs
from general.nn.observables import CalculateAndLogMetrics
from general.utils import format_time, log_console, create_logger, save_yaml, save_pickle, close_handlers
from general.nn.utils import train_val_test_split
from deep_morpho.metrics import masked_dice, accuracy
from deep_morpho.env import CLASSIF_DATASETS, CLASSIF_DATASETS_CHANNEL, SPECIFIC_BIMONNS, SPECIFIC_BIMONNS_DENSE, SPECIFIC_BIMONNS_BISEL
from general.code_saver import CodeSaver


# Import args. Being able to use the CLI to decide the args path.
parser = argparse.ArgumentParser()
parser.add_argument("--args", default="deep_morpho/saved_args/sandbox/args_segm.py")
path_args_module = parser.parse_args().args
path_args_module = path_args_module.replace(".py", "").replace("/", ".")

print(path_args_module)
args_module = import_module(path_args_module)
all_args = args_module.all_args
# from deep_morpho.args_segm import all_args

print('Imports done.')


def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)


def get_dataloader(args):

    # args["channel_classif_args"]["levelset_handler_mode"] = eval(args["channel_classif_args"]["levelset_handler_mode"])
    channel_classif_args = deepcopy(args["channel_classif_args"])
    channel_classif_args["levelset_handler_mode"] = eval(channel_classif_args["levelset_handler_mode"])


    if args['dataset_type'] == 'diskorect':
        # if (args['dataset_path'] is not None) and (args['dataset_path'] != 'generate'):
        #     dataloader = MultiRectDataset.get_loader(
        #         batch_size=args['batch_size'], dataset_path=args['dataset_path'], do_load_in_ram=args['in_ram'],
        #         morp_operation=args['morp_operation'], logger=console_logger, n_inputs=args['n_inputs'],
        #         num_workers=args['num_workers']
        #     )
        # else:
        trainloader = DiskorectDataset.get_loader(
            batch_size=args['batch_size'],
            n_inputs=args['n_inputs'],
            max_generation_nb=args['nb_batch_indep'],
            random_gen_fn=args['random_gen_fn'],
            random_gen_args=args['random_gen_args'],
            morp_operation=args['morp_operation'],
            seed=args['seed'],
            device=device,
            num_workers=args['num_workers'],
            do_symetric_output=args['atomic_element'] == 'sybisel',
            # persistent_workers=True,
            # pin_memory=True,
        )
        valloader = None
        testloader = None

    elif args['dataset_type'] == 'axspa_roi':
        # data = pd.read_csv(args['dataset_path'])
        # prop_train, prop_val, prop_test = args['train_test_split']
        # max_res = data['resolution'].value_counts(sort=True, ascending=False).index[0]
        # data = data[data['resolution'] == max_res]
        # trainloader, valloader, testloader = AxspaROISimpleDataset.get_train_val_test_loader(
        #     *train_val_test_split(
        #         data,
        #         train_size=int(prop_train * len(data)),
        #         val_size=int(prop_val * len(data)),
        #         test_size=int(prop_test * len(data))
        #     ),
        #     batch_size=args['batch_size'],
        #     preprocessing=args['preprocessing'],
        #     shuffle=True,
        #     do_symetric_output=args['atomic_element'] == 'sybisel',
        # )
        trainloader, valloader, testloader = AxspaROISimpleDataset.get_train_val_test_loader(
            n_inputs_train=args['n_inputs_train'],
            n_inputs_val=args['n_inputs_val'],
            n_inputs_test=args['n_inputs_test'],
            batch_size=args['batch_size'],
            preprocessing=args['preprocessing'],
            # shuffle=True,
            do_symetric_output=args['atomic_element'] == 'sybisel',
        )


    elif args['dataset_type'] in ["mnist", "inverted_mnist"]:
        prop_train, prop_val, prop_test = args['train_test_split']
        trainloader, valloader, testloader = MnistMorphoDataset.get_train_val_test_loader(
            n_inputs_train=int(prop_train * args['n_inputs']),
            n_inputs_val=int(prop_val * args['n_inputs']),
            n_inputs_test=int(prop_test * args['n_inputs']),
            batch_size=args['batch_size'],
            morp_operation=args['morp_operation'],
            preprocessing=args['preprocessing'],
            # shuffle=True,
            num_workers=args['num_workers'],
            do_symetric_output=args['atomic_element'] == 'sybisel',
            **args['mnist_args']
        )

    elif args['dataset_type'] == "sticks_noised":
        trainloader = SticksNoisedGeneratorDataset.get_loader(
            batch_size=args['batch_size'],
            n_inputs=args['n_inputs'],
            max_generation_nb=args['nb_batch_indep'],
            seed=args['seed'],
            num_workers=args['num_workers'],
            do_symetric_output=args['atomic_element'] == 'sybisel',
            **args['sticks_noised_args']
        )
        valloader = None
        testloader = None

    elif args['dataset_type'] == "mnist_gray":
        prop_train, prop_val, prop_test = args['train_test_split']
        trainloader, valloader, testloader = MnistGrayScaleDataset.get_train_val_test_loader(
            n_inputs_train=int(prop_train * args['n_inputs']),
            n_inputs_val=int(prop_val * args['n_inputs']),
            n_inputs_test=int(prop_test * args['n_inputs']),
            batch_size=args['batch_size'],
            morp_operation=args['morp_operation'],
            preprocessing=args['preprocessing'],
            # shuffle=True,
            num_workers=args['num_workers'],
            do_symetric_output=args['atomic_element'] == 'sybisel',
            **args['mnist_gray_args']
        )

    elif args['dataset_type'] == "fashionmnist":
        prop_train, prop_val, prop_test = args['train_test_split']
        trainloader, valloader, testloader = FashionMnistGrayScaleDataset.get_train_val_test_loader(
            n_inputs_train=int(prop_train * args['n_inputs']),
            n_inputs_val=int(prop_val * args['n_inputs']),
            n_inputs_test=int(prop_test * args['n_inputs']),
            batch_size=args['batch_size'],
            morp_operation=args['morp_operation'],
            preprocessing=args['preprocessing'],
            # shuffle=True,
            num_workers=args['num_workers'],
            do_symetric_output=args['atomic_element'] == 'sybisel',
            **args['fashionmnist_gray_args']
        )

    elif args['dataset_type'] == "classif_mnist":
        prop_train, prop_val, prop_test = args['train_test_split']
        trainloader, valloader, testloader = MnistClassifDataset.get_train_val_test_loader(
            n_inputs_train=int(prop_train * args['n_inputs']),
            n_inputs_val=int(prop_val * args['n_inputs']),
            n_inputs_test=int(prop_test * args['n_inputs']),
            batch_size=args['batch_size'],
            preprocessing=args['preprocessing'],
            num_workers=args['num_workers'],
            do_symetric_output=args['atomic_element'] == 'sybisel',
            **args['mnist_args']
        )

    elif args['dataset_type'] == "cifar10":
        prop_train, prop_val, prop_test = args['train_test_split']
        trainloader, valloader, testloader = CIFAR10Dataset.get_train_val_test_loader(
            n_inputs_train=int(prop_train * args['n_inputs']),
            n_inputs_val=int(prop_val * args['n_inputs']),
            n_inputs_test=int(prop_test * args['n_inputs']),
            batch_size=args['batch_size'],
            preprocessing=args['preprocessing'],
            num_workers=args['num_workers'],
            do_symetric_output=args['atomic_element'] == 'sybisel',
            **channel_classif_args
        )


    elif args['dataset_type'] == "cifar100":
        prop_train, prop_val, prop_test = args['train_test_split']
        trainloader, valloader, testloader = CIFAR100Dataset.get_train_val_test_loader(
            n_inputs_train=int(prop_train * args['n_inputs']),
            n_inputs_val=int(prop_val * args['n_inputs']),
            n_inputs_test=int(prop_test * args['n_inputs']),
            batch_size=args['batch_size'],
            preprocessing=args['preprocessing'],
            num_workers=args['num_workers'],
            do_symetric_output=args['atomic_element'] == 'sybisel',
            **channel_classif_args
        )


    elif args['dataset_type'] == "classif_mnist_channel":
        prop_train, prop_val, prop_test = args['train_test_split']
        trainloader, valloader, testloader = MnistClassifChannelDataset.get_train_val_test_loader(
            n_inputs_train=int(prop_train * args['n_inputs']),
            n_inputs_val=int(prop_val * args['n_inputs']),
            n_inputs_test=int(prop_test * args['n_inputs']),
            batch_size=args['batch_size'],
            preprocessing=args['preprocessing'],
            num_workers=args['num_workers'],
            do_symetric_output=args['atomic_element'] == 'sybisel',
            **channel_classif_args
        )

    return trainloader, valloader, testloader


def main(args, logger):
    args['seed'] = set_seed(args['batch_seed'])
    with open(join(logger.log_dir, "seed.txt"), "w") as f:
        f.write(f"{args['seed']}")

    trainloader, valloader, testloader = get_dataloader(args)
    if args['dataset_type'] in CLASSIF_DATASETS:
        metrics = {"accuracy": accuracy}
    else:
        metrics = {
            'dice': lambda y_true, y_pred: masked_dice(
                y_true,
                y_pred,
                border=(args['kernel_size'] // 2, args['kernel_size'] // 2),
                threshold=0 if args['atomic_element'] == 'sybisel' else 0.5
            ).mean(),
            # 'mse': lambda y_true, y_pred: ((y_true - y_pred) ** 2).mean()
        }

    if args['dataset_type'] in ['mnist_gray', 'fashionmnist']:
        plot_pred_obs_fn = obs.PlotPredsGrayscale
        binary_mode_fn = obs.BinaryModeMetric
    elif args['dataset_type'] in CLASSIF_DATASETS:
        if args["dataset_type"] in CLASSIF_DATASETS_CHANNEL:
            plot_pred_obs_fn = partial(obs.PlotPredsClassifChannel, dataset=trainloader.dataset)
            binary_mode_fn = partial(obs.BinaryModeMetricClassifChannel, dataset=trainloader.dataset)
        else:
            plot_pred_obs_fn = obs.PlotPredsClassif
            binary_mode_fn = obs.BinaryModeMetricClassif
    else:
        plot_pred_obs_fn = partial(obs.PlotPreds, fig_kwargs={"vmax": 1, "vmin": -1 if args['atomic_element'] == 'sybisel' else 0})
        binary_mode_fn = obs.BinaryModeMetric

    if args['weights_optim_mode'] not in [BiseWeightsOptimEnum.ELLIPSE, BiseWeightsOptimEnum.ELLIPSE_ROOT]:
        plot_grad_obs = obs.PlotGradientBise(freq=args['freq_imgs'])
        plot_weights_fn = obs.PlotWeightsBiSE
    else:
        plot_grad_obs = obs.PlotGradientBiseEllipse(freq=1)
        plot_weights_fn = obs.PlotWeightsBiseEllipse

    if args['early_stopping_on'] == "epoch":
        early_stopping_cls = obs.EpochValEarlyStopping
        reduce_lr_cls = obs.EpochReduceLrOnPlateau
    else:
        early_stopping_cls = obs.BatchEarlyStopping
        reduce_lr_cls = obs.BatchReduceLrOnPlateau

    metric_float_obs = CalculateAndLogMetrics(
        metrics=metrics,
        keep_preds_for_epoch=False,
        freq={'train': args['freq_scalars'], 'val': 1, "test": 1},
    )

    metric_binary_obs = binary_mode_fn(
        metrics=metrics,
        freq={"train": args['freq_scalars'], "val": 1, "test": 1},
        plot_freq={"train": args['freq_imgs'], "val": 0.1 * 60000 // args['batch_size'], "test": args['freq_imgs']},
    )

    observables = [
        # "SetSeed": obs.SetSeed(args['batch_seed']),
        obs.RandomObservable(freq=args['freq_scalars']),
        obs.SaveLoss(freq=1),
        obs.CountInputs(freq=args['freq_scalars']),

        metric_float_obs,
        # "InputAsPredMetric": obs.InputAsPredMetric(metrics, freq=args['freq_scalars']),
        obs.ActivationHistogramBimonn(freq={'train': args['freq_hist'], 'val': 10000 // args['batch_size']}),
        plot_pred_obs_fn(freq={'train': args['freq_imgs'], 'val': 10000 // args['batch_size']}, ),

        # "PlotParametersBiSE": obs.PlotParametersBiSE(freq=args['freq_scalars']),
        # "PlotLUIParametersBiSEL": obs.PlotLUIParametersBiSEL(freq=args['freq_scalars']),
        # "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        obs.PlotParametersBiseEllipse(freq=args['freq_scalars']),
        obs.ActivationPHistogramBimonn(freq={'train': args['freq_hist'], 'val': None}),
        # "PlotWeightsBiSE": plot_weights_fn(freq=args['freq_imgs']),
        # "ExplosiveWeightGradientWatcher": obs.ExplosiveWeightGradientWatcher(freq=1, threshold=0.5),
        # "PlotGradientBise": plot_grad_obs,
        obs.ConvergenceMetrics(metrics, freq=args['freq_scalars']),

        obs.UpdateBinary(freq_batch=args["freq_update_binary_batch"], freq_epoch=args["freq_update_binary_epoch"]),
        # "PlotBimonn": obs.PlotBimonn(freq=args['freq_imgs'], figsize=(10, 5)),
        # "PlotBimonnForward": obs.PlotBimonnForward(freq=args['freq_imgs'], do_plot={"float": True, "binary": True}, dpi=400),
        # "PlotBimonnHistogram": obs.PlotBimonnHistogram(freq=args['freq_imgs'], do_plot={"float": True, "binary": False}, dpi=600),
        obs.ActivationHistogramBinaryBimonn(freq={'train': args['freq_hist'], 'val': 10000 // args['batch_size']}),
        # "CheckMorpOperation": obs.CheckMorpOperation(
        #     selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50
        # ) if args['dataset_type'] == 'diskorect' else obs.Observable(),
        # "ShowSelemAlmostBinary": obs.ShowSelemAlmostBinary(freq=args['freq_imgs']),
        # "ShowSelemBinary": obs.ShowSelemBinary(freq=args['freq_imgs']),
        # "ShowClosestSelemBinary": obs.ShowClosestSelemBinary(freq=args['freq_imgs']),
        # "ShowLUISetBinary": obs.ShowLUISetBinary(freq=args['freq_imgs']),
        metric_binary_obs,
        # "ConvergenceAlmostBinary": obs.ConvergenceAlmostBinary(freq=100),
        # "ConvergenceBinary": obs.ConvergenceBinary(freq=args['freq_imgs']),

        early_stopping_cls(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        # "BatchEarlyStoppingLoss": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss_batch'], mode="min"),
        # "BatchEarlyStoppingBinaryDice": obs.BatchEarlyStopping(name="binary_dice", monitor="binary_mode/dice_train", stopping_threshold=1, patience=np.infty, mode="max"),
        # "BatchActivatedEarlyStopping": obs.BatchActivatedEarlyStopping(patience=0),
        reduce_lr_cls(patience=args['patience_reduce_lr'], on_train=True),
        obs.CheckLearningRate(freq=2 * args['freq_scalars']),
    ]

    # observables_dict = {}

    if args['dataset_type'] in ['mnist_gray', 'fashionmnist']:
        metrics_gray_scale = {'mse': lambda y_true, y_pred: ((y_true - y_pred) ** 2).mean()}
        observables += [
            obs.CalculateAndLogMetricGrayScale(
                metrics=metrics_gray_scale,
                keep_preds_for_epoch=False,
            freq={'train': args['freq_scalars'], 'val': 10, "test": 10},
            ),
            obs.InputAsPredMetricGrayScale(metrics_gray_scale, freq=args['freq_scalars']),
            obs.BinaryModeMetricGrayScale(metrics_gray_scale, freq=args['freq_imgs']),
        ]

    if args['weights_optim_mode'] in [BiseWeightsOptimEnum.ELLIPSE, BiseWeightsOptimEnum.ELLIPSE_ROOT]:
        observables += [
            obs.PlotSigmaBiseEllipse(freq=args['freq_imgs'])
        ]


    # observables = list(observables_dict.values())

    # init_bias_value = next(iter(trainloader))[0].mean()
    inpt = next(iter(trainloader))[0]


    # args["initializer_args"]["input_mean"] = 1/2  # DEBUG

    model_args={
        "channels": args['channels'],
        "threshold_mode": args['threshold_mode'],
        "activation_P": args['activation_P'],
        "constant_activation_P": args['constant_activation_P'],
        "constant_P_lui": args['constant_P_lui'],
        # "init_weight_mode": args["init_weight_mode"],
        # "alpha_init": args["alpha_init"],
        # "init_bias_value_bise": args['init_bias_value_bise'],
        # "init_bias_value_lui": args['init_bias_value_lui'],
        # "input_mean": input_mean,
        "initializer_method": args["initializer_method"],
        "initializer_args": args["initializer_args"],
        "closest_selem_method": args['closest_selem_method'],
        # "closest_selem_distance_fn": args['closest_selem_distance_fn'],
        "bias_optim_mode": args['bias_optim_mode'],
        "bias_optim_args": args['bias_optim_args'],
        "weights_optim_mode": args['weights_optim_mode'],
        "weights_optim_args": args['weights_optim_args'],
        # "constant_weight_P": args['constant_weight_P'],
    }

    if args['model_type'] in SPECIFIC_BIMONNS_BISEL:
        model_args.update({
            "initializer_bise_method": args["initializer_args"]["bise_init_method"],
            "initializer_bise_args": dict(**args["initializer_args"]["bise_init_args"], input_mean=.5),
            "initializer_lui_method": args["initializer_args"]["lui_init_method"],
            "kernel_size": args["kernel_size"],
            "input_mean": inpt.mean().item(),
        })

        if args['dataset_type'] in CLASSIF_DATASETS:
            model_args.update({
                'input_size': inpt.shape[1:],
                'n_classes': trainloader.dataset.n_classes,
            })


    elif args['model_type'] in SPECIFIC_BIMONNS_DENSE:
        model_args.update({
            "initializer_method": args["initializer_args"]["bise_init_method"],
            "initializer_args": dict(**args["initializer_args"]["bise_init_args"], input_mean=.5),
            "input_mean": inpt.mean().item(),
        })

        if args['dataset_type'] in CLASSIF_DATASETS:
            model_args.update({
                'input_size': np.prod(inpt.shape[1:]),
                'n_classes': trainloader.dataset.n_classes,
            })

    else:
        if isinstance(args["initializer_args"], dict):
            args["initializer_args"]["input_mean"] = inpt.mean().item()
        elif isinstance(args["initializer_args"], list):
            args["initializer_args"][0]["input_mean"] = inpt.mean().item()

        model_args.update({
            "kernel_size": [args['kernel_size'] for _ in range(args['n_atoms'])],
            "atomic_element": args["atomic_element"].replace('dual_', ''),
            "lui_kwargs": {"force_identity": args['force_lui_identity']},
        })

        if args['dataset_type'] in CLASSIF_DATASETS:
            model_args.update({
                'input_size': inpt.shape[-2:],
                'n_classes': trainloader.dataset.n_classes,
            })

    lightning_model = eval(args["model_type"])
    model = lightning_model(
        model_args=model_args, learning_rate=args['learning_rate'], loss=args['loss'],
        optimizer=args['optimizer'], observables=observables,
    )

    model.to(device)

    logger.experiment.add_graph(model, inpt[0].unsqueeze(0).to(device))

    hyperparam_str = ""
    for k, v in args.items():
        hyperparam_str += f"**{k}**: {v}  \n"
    hyperparam_str = hyperparam_str[:-2]

    logger.experiment.add_text("hyperparams", hyperparam_str, global_step=0)

    if args['dataset_type'] in ["diskorect", "mnist", "inverted_mnist", "sticks_noised", "mnist_gray"]:
        pathlib.Path(join(logger.log_dir, "target_SE")).mkdir(exist_ok=True, parents=True)
        figs_selems = args['morp_operation'].plot_selem_arrays()
        for (layer_idx, chan_input, chan_output), fig in figs_selems.items():
            fig.savefig(join(logger.log_dir, "target_SE", f"target_SE_l_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
            logger.experiment.add_figure(f"target_SE/target_SE_l_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig)
            plt.close(fig)

        pathlib.Path(join(logger.log_dir, "target_UI")).mkdir(exist_ok=True, parents=True)
        figs_ui = args['morp_operation'].plot_ui_arrays()
        for (layer_idx, chan_output), fig in figs_ui.items():
            fig.savefig(join(logger.log_dir, "target_UI", f"target_UI_l_{layer_idx}_chin_chout_{chan_output}.png"))
            logger.experiment.add_figure(f"target_UI/target_UI_l_{layer_idx}_chin_chout_{chan_output}", fig)
            plt.close(fig)

        pathlib.Path(join(logger.log_dir, "morp_operations")).mkdir(exist_ok=True, parents=True)
        fig_morp_operation = args['morp_operation'].vizualise().fig
        fig_morp_operation.savefig(join(logger.log_dir, "morp_operations", "morp_operations.png"))
        logger.experiment.add_figure("target_operations/morp_operations", fig_morp_operation)
        plt.close(fig_morp_operation)
        # for selem_idx, selem in enumerate(args['morp_operation'].selems):
        #     fig, ax = plt.subplots(); ax.imshow(selem); ax.set_title(args['morp_operation'].operations[selem_idx])
        #     fig.savefig(join(logger.log_dir, "target_SE", f"target_SE_{selem_idx}.png"))
        #     logger.experiment.add_figure(f"target_SE/target_SE_{selem_idx}", fig)
        #     logger.experiment.add_image(f"target_SE/target_SE_{selem_idx}", selem[np.newaxis, :].astype(float))

    callbacks = []
    model_checkpoint_obs = ModelCheckpoint(
        monitor="binary_mode/metrics_epoch_mean/per_batch_step/loss_val",
        dirpath=join(logger.log_dir, "best_weights"),
        save_weights_only=False,
        save_last=True
    )
    callbacks += [model_checkpoint_obs, ]

    trainer = Trainer(
        max_epochs=args['n_epochs'],
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
        # progress_bar_refresh_rate=10,
        callbacks=observables.copy() + callbacks.copy(),
        log_every_n_steps=10,
        deterministic=True,
        num_sanity_val_steps=1,
    )

    log_console(f"Binarizable parameters: {model.model.numel_binary()} / {model.model.numel_float()} = {model.model.numel_binary()/model.model.numel_float():.2f}", logger=console_logger)
    # log_console("Binarizable parameters:", model.model.numel_binary(), logger=console_logger)
    trainer.fit(model, trainloader, valloader)
    model.load_state_dict(torch.load(model_checkpoint_obs.best_model_path)["state_dict"])
    trainer.test(model, testloader)

    metric_dict = {}
    for state in ["train", "val", "test"]:
        for metric_name in metric_float_obs.metrics.keys():
            metric_dict[f"{metric_name}_{state}"] = metric_float_obs.last_value[state][metric_name]
            metric_dict[f"binary_{metric_name}_{state}"] = metric_binary_obs.last_value[state][metric_name]
    logger.log_hyperparams(args, metric_dict)


    for observable in observables:
        observable.save(join(trainer.log_dir, 'observables'))



if __name__ == '__main__':
    start_all = time()

    code_saver = CodeSaver(
        src_path=os.getcwd(),
        temporary_path="deep_morpho/results/results_tensorboards",
        ignore_patterns=("*__pycache__*", "*results*", "data", "*.ipynb", '.git', 'ssm', 'sampling',
                        "tests", "Notebooks", "*geogebra*", ".github", ".idea", ".pytest_cache", ".vscode", "html_pages",
                        "paper_writer", "ruche_logs", ),
    )

    code_saver.save_in_temporary_file()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    bugged = []
    results = []
    console_logger = None

    for args_idx, args in enumerate(all_args):

        name = join(args["experiment_name"], args['experiment_subname'])

        # name += f"_{args['atomic_element']}"

        logger = TensorBoardLogger("deep_morpho/results/results_tensorboards", name=name, default_hp_metric=False)

        if code_saver.nb_final_save == 0:
            code_saver.save_in_final_file(join(logger.log_dir, 'code'))
        else:
            pathlib.Path(logger.log_dir).mkdir(exist_ok=True, parents=True)
            with open(join(logger.log_dir, "code_path.txt"), 'w') as f:
                print(code_saver.last_save_path, file=f)

        save_yaml(args, join(logger.log_dir, 'args.yaml'))
        # save_pickle(args, join(logger.log_dir, 'args.pkl'))

        if console_logger is not None:
            close_handlers(console_logger)

        console_logger = create_logger(
            f'args_{args_idx}', all_logs_path=join(logger.log_dir, 'all_logs.log'), error_path=join(logger.log_dir, 'error_logs.log')
        )

        log_console('Device: {}'.format(device), logger=console_logger)
        log_console('==================', logger=console_logger)
        log_console('==================', logger=console_logger)
        log_console(f'Args number {args_idx + 1} / {len(all_args)}', logger=console_logger)
        log_console('Time since beginning: {} '.format(format_time(time() - start_all)), logger=console_logger)
        log_console(logger.log_dir, logger=console_logger)
        # log_console(args['morp_operation'], logger.log_dir, logger=console_logger)

        results.append(main(args, logger))

        # try:
        #     main(args, logger)
        # except Exception:
        #     console_logger.exception(
        #         f'Args nb {args_idx + 1} / {len(all_args)} failed : ')
        #     bugged.append(args_idx+1)

        log_console(logger.log_dir, logger=console_logger)
        log_console("Done.", logger=console_logger)

    code_saver.delete_temporary_file()

    # results = pd.concat(results)

    log_console(f'{len(bugged)} Args Bugged: ', bugged, logger=console_logger)
    log_console(f'{len(all_args)} args done in {format_time(time() - start_all)} ', logger=console_logger)
