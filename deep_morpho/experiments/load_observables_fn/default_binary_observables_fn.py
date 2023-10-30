from os.path import join
import os
from typing import Tuple

import numpy as np

from general.nn.observables import CalculateAndLogMetrics
from deep_morpho.metrics import dice, accuracy
import deep_morpho.observables as obs
from pytorch_lightning.callbacks import ModelCheckpoint


def load_observables_morpho_binary(experiment):
    args = experiment.args
    metrics = {
        'dice': lambda y_true, y_pred: dice(
            y_true,
            y_pred,
            threshold=0 if args['atomic_element'] == 'sybisel' else 0.5
        ).mean(),
    }

    metric_float_obs = CalculateAndLogMetrics(
        metrics=metrics, keep_preds_for_epoch=False, freq={'train': args['freq_scalars'], 'val': 1, "test": 1},
    )

    metric_binary_obs = obs.BinaryModeMetricMorpho(
        metrics=metrics,
        freq={"train": args['freq_scalars'], "val": 1, "test": 1},
        plot_freq={"train": args['freq_imgs'], "val": args["n_inputs.val"] // args['batch_size'], "test": args['freq_imgs']},
    )

    observables = [
        obs.RandomObservable(freq=args['freq_scalars']),
        obs.SaveLoss(freq=1),
        obs.CountInputs(freq=args['freq_scalars']),

        metric_float_obs,
        obs.InputAsPredMetric(metrics, freq=args['freq_scalars']),
        # obs.ActivationHistogramBimonn(freq={'train': args['freq_hist'], 'val': args["n_inputs.val"] // args['batch_size']}),
        obs.PlotPreds(
            freq={'train': args['freq_imgs'], 'val': args["n_inputs.val"] // args['batch_size']},
            fig_kwargs={"vmax": 1, "vmin": -1 if (args['atomic_element'] == 'sybisel') or (args["do_symetric_output"]) else 0}
        ),

        # obs.PlotParametersBiSE(freq=args['freq_scalars']),
        # obs.PlotLUIParametersBiSEL(freq=args['freq_scalars']),
        # "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        # obs.PlotParametersBiseEllipse(freq=args['freq_scalars']),
        # obs.ActivationPHistogramBimonn(freq={'train': args['freq_hist'], 'val': None}),
        # obs.PlotWeightsBiSE(freq=args['freq_imgs']),
        # "ExplosiveWeightGradientWatcher": obs.ExplosiveWeightGradientWatcher(freq=1, threshold=0.5),
        # obs.PlotGradientBise(freq=args['freq_imgs']),
        obs.ConvergenceMetrics(metrics, freq=args['freq_scalars']),

        obs.UpdateBinary(freq_batch=args["freq_update_binary_batch"], freq_epoch=args["freq_update_binary_epoch"]),
        # obs.ActivatednessObservable(freq=args["freq_update_binary_epoch"]),
        # obs.ClosestDistObservable(freq=args["freq_update_binary_epoch"]),
        # obs.PlotBimonn(freq=args['freq_imgs'], figsize=(10, 5)),
        # "PlotBimonnForward": obs.PlotBimonnForward(freq=args['freq_imgs'], do_plot={"float": True, "binary": True}, dpi=400),
        # "PlotBimonnHistogram": obs.PlotBimonnHistogram(freq=args['freq_imgs'], do_plot={"float": True, "binary": False}, dpi=600),
        # obs.ActivationHistogramBinaryBimonn(freq={'train': args['freq_hist'], 'val': args["n_inputs.val"] // args['batch_size']}),
        # "CheckMorpOperation": obs.CheckMorpOperation(
        #     selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50
        # ) if args['dataset_type'] == 'diskorect' else obs.Observable(),
        # "ShowSelemAlmostBinary": obs.ShowSelemAlmostBinary(freq=args['freq_imgs']),
        # obs.ShowSelemBinary(freq=args['freq_imgs']),
        # obs.ShowClosestSelemBinary(freq=args['freq_imgs']),
        # "ShowLUISetBinary": obs.ShowLUISetBinary(freq=args['freq_imgs']),
        metric_binary_obs,
        # "ConvergenceAlmostBinary": obs.ConvergenceAlmostBinary(freq=100),
        # "ConvergenceBinary": obs.ConvergenceBinary(freq=args['freq_imgs']),

        obs.EpochReduceLrOnPlateau(patience=args['patience_reduce_lr'], on_train=True),
        obs.CheckLearningRate(freq=2 * args['freq_scalars']),
        # obs.EpochValEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        # "BatchEarlyStoppingLoss": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss_batch'], mode="min"),
        # "BatchActivatedEarlyStopping": obs.BatchActivatedEarlyStopping(patience=0),
        # obs.BatchEarlyStopping(name="binary_dice", monitor="binary_mode/dice_train", stopping_threshold=1, patience=np.infty, mode="max"),
    ]

    if 'early_stopping' in experiment.args:
        observables += experiment.args['early_stopping']
    else:
        observables += [
            obs.EpochValEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        ]

    model_checkpoint_obs = ModelCheckpoint(
        monitor="metrics_epoch_mean/per_batch_step/loss_val",
        dirpath=join(experiment.log_dir, "best_weights"),
        save_weights_only=False,
        save_last=True
    )
    callbacks = [model_checkpoint_obs]

    return observables, callbacks, metric_float_obs, metric_binary_obs, model_checkpoint_obs


def load_observables_ste_morpho_binary(experiment):
    args = experiment.args
    metrics = {
        'dice': lambda y_true, y_pred: dice(
            y_true,
            y_pred,
            threshold=0 if args['atomic_element'] == 'sybisel' else 0.5
        ).mean(),
    }

    metric_float_obs = CalculateAndLogMetrics(
        metrics=metrics, keep_preds_for_epoch=False, freq={'train': args['freq_scalars'], 'val': 1, "test": 1},
    )

    metric_binary_obs = obs.BinaryModeMetricMorpho(
        metrics=metrics,
        freq={"train": args['freq_scalars'], "val": 1, "test": 1},
        plot_freq={"train": args['freq_imgs'], "val": args["n_inputs.val"] // args['batch_size'], "test": args['freq_imgs']},
    )

    observables = [
        obs.RandomObservable(freq=args['freq_scalars']),
        obs.SaveLoss(freq=1),
        obs.CountInputs(freq=args['freq_scalars']),

        metric_float_obs,
        obs.InputAsPredMetric(metrics, freq=args['freq_scalars']),
        # obs.ActivationHistogramBimonn(freq={'train': args['freq_hist'], 'val': args["n_inputs.val"] // args['batch_size']}),
        obs.PlotPreds(
            freq={'train': args['freq_imgs'], 'val': args["n_inputs.val"] // args['batch_size']},
            fig_kwargs={"vmax": 1, "vmin": -1 if (args['atomic_element'] == 'sybisel') or (args["do_symetric_output"]) else 0}
        ),
        obs.PlotSteWeights(freq=args['freq_imgs']),
        obs.PlotSTE(freq=args['freq_imgs']),

        # obs.PlotParametersBiSE(freq=args['freq_scalars']),
        # obs.PlotLUIParametersBiSEL(freq=args['freq_scalars']),
        # "WeightsHistogramBiSE": obs.WeightsHistogramBiSE(freq=args['freq_imgs']),
        # obs.PlotParametersBiseEllipse(freq=args['freq_scalars']),
        # obs.ActivationPHistogramBimonn(freq={'train': args['freq_hist'], 'val': None}),
        # obs.PlotWeightsBiSE(freq=args['freq_imgs']),
        # "ExplosiveWeightGradientWatcher": obs.ExplosiveWeightGradientWatcher(freq=1, threshold=0.5),
        # obs.PlotGradientBise(freq=args['freq_imgs']),
        obs.ConvergenceMetrics(metrics, freq=args['freq_scalars']),

        # obs.UpdateBinary(freq_batch=args["freq_update_binary_batch"], freq_epoch=args["freq_update_binary_epoch"]),
        # obs.ActivatednessObservable(freq=args["freq_update_binary_epoch"]),
        # obs.ClosestDistObservable(freq=args["freq_update_binary_epoch"]),
        # obs.PlotBimonn(freq=args['freq_imgs'], figsize=(10, 5)),
        # "PlotBimonnForward": obs.PlotBimonnForward(freq=args['freq_imgs'], do_plot={"float": True, "binary": True}, dpi=400),
        # "PlotBimonnHistogram": obs.PlotBimonnHistogram(freq=args['freq_imgs'], do_plot={"float": True, "binary": False}, dpi=600),
        # obs.ActivationHistogramBinaryBimonn(freq={'train': args['freq_hist'], 'val': args["n_inputs.val"] // args['batch_size']}),
        # "CheckMorpOperation": obs.CheckMorpOperation(
        #     selems=args['morp_operation'].selems, operations=args['morp_operation'].operations, freq=50
        # ) if args['dataset_type'] == 'diskorect' else obs.Observable(),
        # "ShowSelemAlmostBinary": obs.ShowSelemAlmostBinary(freq=args['freq_imgs']),
        # obs.ShowSelemBinary(freq=args['freq_imgs']),
        # obs.ShowClosestSelemBinary(freq=args['freq_imgs']),
        # "ShowLUISetBinary": obs.ShowLUISetBinary(freq=args['freq_imgs']),
        # metric_binary_obs,
        # "ConvergenceAlmostBinary": obs.ConvergenceAlmostBinary(freq=100),
        # "ConvergenceBinary": obs.ConvergenceBinary(freq=args['freq_imgs']),

        obs.EpochReduceLrOnPlateau(patience=args['patience_reduce_lr'], on_train=True),
        obs.CheckLearningRate(freq=2 * args['freq_scalars']),
        # obs.EpochValEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        # "BatchEarlyStoppingLoss": obs.BatchEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss_batch'], mode="min"),
        # "BatchActivatedEarlyStopping": obs.BatchActivatedEarlyStopping(patience=0),
        # obs.BatchEarlyStopping(name="binary_dice", monitor="binary_mode/dice_train", stopping_threshold=1, patience=np.infty, mode="max"),
    ]

    if 'early_stopping' in experiment.args:
        observables += experiment.args['early_stopping']
    else:
        observables += [
            obs.EpochValEarlyStopping(name="loss", monitor="loss/train/loss", patience=args['patience_loss'], mode="min"),
        ]


    model_checkpoint_obs = ModelCheckpoint(
        monitor="metrics_epoch_mean/per_batch_step/loss_val",
        dirpath=join(experiment.log_dir, "best_weights"),
        save_weights_only=False,
        save_last=True
    )
    callbacks = [model_checkpoint_obs]

    return observables, callbacks, metric_float_obs, metric_binary_obs, model_checkpoint_obs

