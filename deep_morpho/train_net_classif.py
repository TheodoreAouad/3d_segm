print("Import...")
from time import time
import os
from os.path import join
import pathlib

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import matplotlib.pyplot as plt

from deep_morpho.datasets.mnist_dataset import MnistClassifDataset
from deep_morpho.utils import set_seed
from deep_morpho.models import LightningBiMoNNClassifier, BiSE, COBiSE, BiSEC, COBiSEC
from deep_morpho.metrics import accuracy
import deep_morpho.observables as obs
# from deep_morpho.observables.check_morp_op import ShowClosestSelemBinary
from general.nn.observables import CalculateAndLogMetrics
from general.utils import format_time, log_console, create_logger, save_yaml, save_pickle
from deep_morpho.args_classif import all_args
from general.code_saver import CodeSaver
print("Import done.")


def get_dataloader(args):


    prop_train, prop_val, prop_test = args['train_test_split']
    trainloader, valloader, testloader = MnistClassifDataset.get_train_val_test_loader(
        n_inputs_train=int(prop_train * args['n_inputs']),
        n_inputs_val=int(prop_val * args['n_inputs']),
        n_inputs_test=int(prop_test * args['n_inputs']),
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=args['num_workers'],
        **args['mnist_args']
    )

    return trainloader, valloader, testloader


def main(args, logger):
    args['seed'] = set_seed(args['batch_seed'])
    with open(join(logger.log_dir, "seed.txt"), "w") as f:
        f.write(f"{args['seed']}")

    trainloader, valloader, testloader = get_dataloader(args)
    metrics = {"accuracy": accuracy}

    observables_dict = {
        # "SetSeed": obs.SetSeed(args['batch_seed']),
        "RandomObservable": obs.RandomObservable(),
        "SaveLoss": obs.SaveLoss(),
        "CalculateAndLogMetric": CalculateAndLogMetrics(
            metrics=metrics,
            keep_preds_for_epoch=False,
        ),
        "PlotPredsClassif": obs.PlotPredsClassif(freq={'train': 50, 'val': 50}),
        # "InputAsPredMetric": obs.InputAsPredMetric(metrics),
        "CountInputs": obs.CountInputs(),
        "PlotParametersBiSE": obs.PlotParametersBiSE(freq=1),
        # "PlotWeightsBiSE": obs.PlotWeightsBiSE(freq=args['freq_imgs']),
        "PlotLUIParametersBiSEL": obs.PlotLUIParametersBiSEL(),
        "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        # "CheckMorpOperation": obs.CheckMorpOperation(
        #     selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50
        # ) if args['dataset_type'] == 'diskorect' else obs.Observable(),
        "PlotGradientBise": obs.PlotGradientBise(freq=args['freq_imgs'], plot_figure=False),
        # "ExplosiveWeightGradientWatcher": obs.ExplosiveWeightGradientWatcher(freq=1, threshold=0.5),
        "ConvergenceMetrics": obs.ConvergenceMetrics(metrics),
        # "ShowSelemAlmostBinary": obs.ShowSelemAlmostBinary(freq=args['freq_imgs']),
        "ShowSelemBinary": obs.ShowSelemBinary(freq=args['freq_imgs']),
        # "ShowClosestSelemBinary": obs.ShowClosestSelemBinary(freq=args['freq_imgs']),
        # "ShowLUISetBinary": obs.ShowLUISetBinary(freq=args['freq_imgs']),
        # "BinaryModeMetric": obs.BinaryModeMetric(metrics, freq=args['freq_imgs']),
        # "ConvergenceAlmostBinary": obs.ConvergenceAlmostBinary(freq=100),
        "ConvergenceBinary": obs.ConvergenceBinary(freq=100),
        "PlotBimonn": obs.PlotBimonn(freq=args['freq_imgs'], figsize=(10, 5), do_plot={"weights": True, "learned": True, "closest": False,}),
        "PlotBimonnForward": obs.PlotBimonnForward(freq=args['freq_imgs'], do_plot={"float": True, "binary": False}),
        "PlotBimonnHistogram": obs.PlotBimonnHistogram(freq=args['freq_imgs'], do_plot={"float": True, "binary": False})
    }

    observables = list(observables_dict.values())

    xs = torch.tensor(np.linspace(-6, 6, 100)).detach()

    model = LightningBiMoNNClassifier(
        model_args={
            "kernel_size": [args['kernel_size'] for _ in range(args['n_atoms'])],
            "n_classes": 10,
            "input_size": (28, 28),
            "channels": args['channels'],
            "atomic_element": args["atomic_element"] if args["atomic_element"] != "conv" else "bise",
            "threshold_mode": args['threshold_mode'],
            "activation_P": args['activation_P'],
            "constant_activation_P": args['constant_activation_P'],
            "constant_P_lui": args['constant_P_lui'],
            "constant_weight_P": args['constant_weight_P'],
            "init_weight_mode": args["init_weight_mode"],
            "alpha_init": args["alpha_init"],
            "lui_kwargs": {"force_identity": args['force_lui_identity']},
        },
        learning_rate=args['learning_rate'],
        loss=args['loss'],
        optimizer=args['optimizer'],
        observables=observables,
    )
    if isinstance(model.model.layers[0], (BiSE, COBiSE, BiSEC, COBiSEC)):
        ys = model.model.layers[0].activation_threshold_fn(xs).detach()
        fig, ax = plt.subplots(); ax.plot(xs, ys); ax.set_title(args['threshold_mode'])
        logger.experiment.add_figure("threshold_fn", fig)

    model.to(device)

    # logger.experiment.add_graph(model, torch.ones(1, args['channels'][0], 28, 28).to(device))
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
        **{f"metrics_batch/accuracy_train": torch.tensor([np.nan])},
        **{f"convergence/metric_accuracy_train": torch.tensor([np.nan])},
    )


    logger.log_hyperparams(args, hyperparams)

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
        ignore_patterns=("*__pycache__*", "*results*", "data", "*.ipynb", '.git', 'ssm', "*weights*"),
    )

    code_saver.save_in_temporary_file()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    bugged = []
    results = []

    for args_idx, args in enumerate(all_args):

        name = join(args["experiment_name"], args['experiment_subname'])

        # name += f"_{args['atomic_element']}"

        logger = TensorBoardLogger("deep_morpho/results/results_tensorboards", name=name, default_hp_metric=False)
        code_saver.save_in_final_file(join(logger.log_dir, 'code'))
        save_yaml(args, join(logger.log_dir, 'args.yaml'))
        save_pickle(args, join(logger.log_dir, 'args.pkl'))


        console_logger = create_logger(
            f'args_{args_idx}', all_logs_path=join(logger.log_dir, 'all_logs.log'), error_path=join(logger.log_dir, 'error_logs.log')
        )

        log_console('Device: {}'.format(device), logger=console_logger)
        log_console('==================', logger=console_logger)
        log_console('==================', logger=console_logger)
        log_console(f'Args number {args_idx + 1} / {len(all_args)}', logger=console_logger)
        log_console('Time since beginning: {} '.format(format_time(time() - start_all)), logger=console_logger)
        log_console(logger.log_dir, logger=console_logger)
        results.append(main(args, logger))
        # try:
        #     main(args, logger)
        # except Exception:
        #     console_logger.exception(
        #         f'Args nb {args_idx + 1} / {len(all_args)} failed : ')
        #     bugged.append(args_idx+1)

    code_saver.delete_temporary_file()

    # results = pd.concat(results)

    log_console(f'{len(bugged)} Args Bugged: ', bugged, logger=console_logger)
    log_console(f'{len(all_args)} args done in {format_time(time() - start_all)} ', logger=console_logger)
