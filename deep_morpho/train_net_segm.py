# import warnings
# warnings.filterwarnings('error', message=".+leaf Tensor.+")

from time import time
import os
from os.path import join
import pathlib

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deep_morpho.datasets.mnist_dataset import MnistMorphoDataset
from deep_morpho.utils import set_seed
# from deep_morpho.datasets.generate_forms2 import get_random_diskorect
# from deep_morpho.datasets.generate_forms3 import get_random_rotated_diskorect
from deep_morpho.datasets.multi_rect_dataset import InputOutputGeneratorDataset, MultiRectDataset
from deep_morpho.datasets.axspa_roi_dataset import AxspaROISimpleDataset
from deep_morpho.models import LightningBiMoNN, BiSE #COBiSE, BiSEC, COBiSEC
import deep_morpho.observables as obs
from general.nn.observables import CalculateAndLogMetrics
from general.utils import format_time, log_console, create_logger, save_yaml, save_pickle, close_handlers
from general.nn.utils import train_val_test_split
from deep_morpho.metrics import masked_dice
from deep_morpho.args_segm import all_args
from general.code_saver import CodeSaver


def get_dataloader(args):

    if args['dataset_type'] == 'diskorect':
        # if (args['dataset_path'] is not None) and (args['dataset_path'] != 'generate'):
        #     dataloader = MultiRectDataset.get_loader(
        #         batch_size=args['batch_size'], dataset_path=args['dataset_path'], do_load_in_ram=args['in_ram'],
        #         morp_operation=args['morp_operation'], logger=console_logger, n_inputs=args['n_inputs'],
        #         num_workers=args['num_workers']
        #     )
        # else:
        trainloader = InputOutputGeneratorDataset.get_loader(
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
        data = pd.read_csv(args['dataset_path'])
        prop_train, prop_val, prop_test = args['train_test_split']
        max_res = data['resolution'].value_counts(sort=True, ascending=False).index[0]
        data = data[data['resolution'] == max_res]
        trainloader, valloader, testloader = AxspaROISimpleDataset.get_train_val_test_loader(
            *train_val_test_split(
                data,
                train_size=int(prop_train * len(data)),
                val_size=int(prop_val * len(data)),
                test_size=int(prop_test * len(data))
            ),
            batch_size=args['batch_size'],
            preprocessing=args['preprocessing'],
            shuffle=True,
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
            **args['mnist_args']
        )

    return trainloader, valloader, testloader


def main(args, logger):
    args['seed'] = set_seed(args['batch_seed'])
    with open(join(logger.log_dir, "seed.txt"), "w") as f:
        f.write(f"{args['seed']}")

    trainloader, valloader, testloader = get_dataloader(args)
    metrics = {'dice': lambda y_true, y_pred: masked_dice(y_true, y_pred, 
                        border=(args['kernel_size'] // 2, args['kernel_size'] // 2), threshold=0 if args['atomic_element'] == 'sybisel' else 0.5).mean()}

    observables_dict = {
        # "SetSeed": obs.SetSeed(args['batch_seed']),
        "RandomObservable": obs.RandomObservable(),
        "SaveLoss": obs.SaveLoss(),
        "CalculateAndLogMetric": CalculateAndLogMetrics(
            metrics=metrics,
            keep_preds_for_epoch=False,
        ),
        "PlotPreds": obs.PlotPreds(freq={'train': args['freq_imgs'], 'val': 39}, fig_kwargs={"vmax": 1, "vmin": -1 if args['atomic_element'] == 'sybisel' else 0}),
        "PlotBimonn": obs.PlotBimonn(freq=args['freq_imgs'], figsize=(10, 5)),
        # "PlotBimonnForward": obs.PlotBimonnForward(freq=args['freq_imgs'], do_plot={"float": True, "binary": True}, dpi=600),
        # "PlotBimonnHistogram": obs.PlotBimonnHistogram(freq=args['freq_imgs'], do_plot={"float": True, "binary": True}, dpi=600),
        "InputAsPredMetric": obs.InputAsPredMetric(metrics),
        "CountInputs": obs.CountInputs(),
        "PlotParametersBiSE": obs.PlotParametersBiSE(freq=1),
        "PlotWeightsBiSE": obs.PlotWeightsBiSE(freq=args['freq_imgs']),
        "PlotLUIParametersBiSEL": obs.PlotLUIParametersBiSEL(),
        "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        # "CheckMorpOperation": obs.CheckMorpOperation(
        #     selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50
        # ) if args['dataset_type'] == 'diskorect' else obs.Observable(),
        "PlotGradientBise": obs.PlotGradientBise(freq=args['freq_imgs']),
        # "ExplosiveWeightGradientWatcher": obs.ExplosiveWeightGradientWatcher(freq=1, threshold=0.5),
        "ConvergenceMetrics": obs.ConvergenceMetrics(metrics),
        # "ShowSelemAlmostBinary": obs.ShowSelemAlmostBinary(freq=args['freq_imgs']),
        "ShowSelemBinary": obs.ShowSelemBinary(freq=args['freq_imgs']),
        "ShowClosestSelemBinary": obs.ShowClosestSelemBinary(freq=args['freq_imgs']),
        "ShowLUISetBinary": obs.ShowLUISetBinary(freq=args['freq_imgs']),
        "BinaryModeMetric": obs.BinaryModeMetric(metrics, freq=args['freq_imgs']),
        # "ConvergenceAlmostBinary": obs.ConvergenceAlmostBinary(freq=100),
        "ConvergenceBinary": obs.ConvergenceBinary(freq=100),
        "BatchEarlyStoppingLoss": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        "BatchEarlyStoppingBinaryDice": obs.BatchEarlyStopping(name="binary_dice", monitor="binary_mode/dice_train", stopping_threshold=1, patience=np.infty, mode="max"),
        # "BatchActivatedEarlyStopping": obs.BatchActivatedEarlyStopping(patience=0),
        "BatchReduceLrOnPlateau": obs.BatchReduceLrOnPlateau(patience=args['patience_reduce_lr'], on_train=True),
        "CheckLearningRate": obs.CheckLearningRate(freq=2),
    }

    observables = list(observables_dict.values())

    xs = torch.tensor(np.linspace(-6, 6, 100)).detach()

    # init_bias_value = next(iter(trainloader))[0].mean()
    input_mean = next(iter(trainloader))[0].mean()
    model = LightningBiMoNN(
        model_args={
            "kernel_size": [args['kernel_size'] for _ in range(args['n_atoms'])],
            "channels": args['channels'],
            "atomic_element": args["atomic_element"] if args["atomic_element"] != "conv" else "bise",
            "threshold_mode": args['threshold_mode'],
            "activation_P": args['activation_P'],
            "constant_activation_P": args['constant_activation_P'],
            "constant_P_lui": args['constant_P_lui'],
            "init_weight_mode": args["init_weight_mode"],
            "alpha_init": args["alpha_init"],
            "lui_kwargs": {"force_identity": args['force_lui_identity']},
            "init_bias_value_bise": args['init_bias_value_bise'],
            "init_bias_value_lui": args['init_bias_value_lui'],
            "input_mean": input_mean,
            "closest_selem_method": args['closest_selem_method'],
            "closest_selem_distance_fn": args['closest_selem_distance_fn'],
            "bias_optim_mode": args['bias_optim_mode'],
        },
        learning_rate=args['learning_rate'],
        loss=args['loss'],
        optimizer=args['optimizer'],
        observables=observables,
    )
    if isinstance(model.model.layers[0], (BiSE)):
        ys = model.model.layers[0].activation_threshold_fn(xs).detach()
        fig, ax = plt.subplots(); ax.plot(xs, ys); ax.set_title(args['threshold_mode'])
        logger.experiment.add_figure("threshold_fn", fig)

    model.to(device)

    logger.experiment.add_graph(model, torch.ones(1, args['channels'][0], 50, 50).to(device))
    hyperparams = dict(
        # **{f'{k}_{layer_idx}': -1 for k in [
        #     f"weights/sum_norm_weights",
        #     f"params/weight_P",
        #     f"params/activation_P",
        #     f"weights/bias",
        # ] for layer_idx in range(len(model.model.layers))},
        **{
            f'{k}/layer_{layer_idx}_chout_{chan_output}_chin_{chan_input}': torch.tensor([np.nan]) for k in [
                "convergence/binary/bisel",
            ] for layer_idx in range(len(model.model.layers))
            for chan_input in range(model.model.layers[layer_idx].in_channels)
            for chan_output in range(model.model.layers[layer_idx].out_channels)
        },
        **{
            f'{k}/layer_{layer_idx}_chout_{chan_output}': torch.tensor([np.nan]) for k in [
                "convergence/binary/lui",
            ] for layer_idx in range(len(model.model.layers))
            for chan_output in range(model.model.layers[layer_idx].out_channels)
        },
        **{"metrics_batch/dice_train": torch.tensor([np.nan])},
        **{"convergence/metric_dice_train": torch.tensor([np.nan])},
    )

    if args['atomic_element'] in ['bisec', 'cobisec']:
        hyperparams.update(dict(
            **{f'{k}_{layer_idx}': -1 for k in ["weights/norm_alpha"] for layer_idx in range(len(model.model.layers))}
        ))

    if args["atomic_element"] == 'cobise':
        hyperparams.update(dict(
            **{f'metrics/bias - lb(op)_{layer_idx}_0': torch.tensor([np.nan]) for layer_idx in range(len(model.model.layers))},
            **{f'metrics/ub(op) - bias_{layer_idx}_0': torch.tensor([np.nan]) for layer_idx in range(len(model.model.layers))},
            **{f'metrics/bias - lb(op)_{layer_idx}_1': torch.tensor([np.nan]) for layer_idx in range(len(model.model.layers))},
            **{f'metrics/ub(op) - bias_{layer_idx}_1': torch.tensor([np.nan]) for layer_idx in range(len(model.model.layers))},
        ))
    elif args["atomic_element"] in ["bise", "conv"]:
        hyperparams.update(dict(
            **{f'metrics/bias - lb(op)_{layer_idx}': torch.tensor([np.nan]) for layer_idx in range(len(model.model.layers))},
            **{f'metrics/ub(op) - bias_{layer_idx}': torch.tensor([np.nan]) for layer_idx in range(len(model.model.layers))},
        ))

    logger.log_hyperparams(args, hyperparams)

    if args['dataset_type'] in ["diskorect", "mnist"]:
        pathlib.Path(join(logger.log_dir, "target_SE")).mkdir(exist_ok=True, parents=True)
        figs_selems = args['morp_operation'].plot_selem_arrays()
        for (layer_idx, chan_input, chan_output), fig in figs_selems.items():
            fig.savefig(join(logger.log_dir, "target_SE", f"target_SE_l_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
            logger.experiment.add_figure(f"target_SE/target_SE_l_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig)

        pathlib.Path(join(logger.log_dir, "target_UI")).mkdir(exist_ok=True, parents=True)
        figs_ui = args['morp_operation'].plot_ui_arrays()
        for (layer_idx, chan_output), fig in figs_ui.items():
            fig.savefig(join(logger.log_dir, "target_UI", f"target_UI_l_{layer_idx}_chin_chout_{chan_output}.png"))
            logger.experiment.add_figure(f"target_UI/target_UI_l_{layer_idx}_chin_chout_{chan_output}", fig)

        pathlib.Path(join(logger.log_dir, "morp_operations")).mkdir(exist_ok=True, parents=True)
        fig_morp_operation = args['morp_operation'].vizualise().fig
        fig_morp_operation.savefig(join(logger.log_dir, "morp_operations", "morp_operations.png"))
        logger.experiment.add_figure("target_operations/morp_operations", fig_morp_operation)
        # for selem_idx, selem in enumerate(args['morp_operation'].selems):
        #     fig, ax = plt.subplots(); ax.imshow(selem); ax.set_title(args['morp_operation'].operations[selem_idx])
        #     fig.savefig(join(logger.log_dir, "target_SE", f"target_SE_{selem_idx}.png"))
        #     logger.experiment.add_figure(f"target_SE/target_SE_{selem_idx}", fig)
        #     logger.experiment.add_image(f"target_SE/target_SE_{selem_idx}", selem[np.newaxis, :].astype(float))


    trainer = Trainer(
        max_epochs=args['n_epochs'],
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
        # progress_bar_refresh_rate=10,
        callbacks=observables.copy(),
        log_every_n_steps=10,
        deterministic=True,
        num_sanity_val_steps=1,
    )

    trainer.fit(model, trainloader, valloader,)

    for observable in observables:
        observable.save(join(trainer.log_dir, 'observables'))



if __name__ == '__main__':
    start_all = time()

    code_saver = CodeSaver(
        src_path=os.getcwd(),
        temporary_path="deep_morpho/results/results_tensorboards",
        ignore_patterns=("*__pycache__*", "*results*", "data", "*.ipynb", '.git', 'ssm'),
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
        code_saver.save_in_final_file(join(logger.log_dir, 'code'))
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
        log_console(args['morp_operation'], logger.log_dir, logger=console_logger)
        results.append(main(args, logger))
        # try:
        #     main(args, logger)
        # except Exception:
        #     console_logger.exception(
        #         f'Args nb {args_idx + 1} / {len(all_args)} failed : ')
        #     bugged.append(args_idx+1)
        log_console("Done.", logger=console_logger)

    code_saver.delete_temporary_file()

    # results = pd.concat(results)

    log_console(f'{len(bugged)} Args Bugged: ', bugged, logger=console_logger)
    log_console(f'{len(all_args)} args done in {format_time(time() - start_all)} ', logger=console_logger)
