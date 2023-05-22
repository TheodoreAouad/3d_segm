from functools import partial

import torch.nn as nn

from deep_morpho.experiments.enforcers import ArgsEnforcer
from deep_morpho.models.bise_base import ClosestSelemEnum, ClosestSelemDistanceEnum, BiseBiasOptimEnum, BiseWeightsOptimEnum
from deep_morpho.initializer import InitBimonnEnum, InitBiseEnum
from deep_morpho.loss import (
    MaskedMSELoss, MaskedDiceLoss, MaskedBCELoss, QuadraticBoundRegularization, LinearBoundRegularization,
    MaskedBCENormalizedLoss, MaskedNormalizedDiceLoss, BCENormalizedLoss, DiceLoss, NormalizedDiceLoss
)


loss_dict = {
    "MaskedMSELoss": MaskedMSELoss,
    "MaskedDiceLoss": MaskedDiceLoss,
    "DiceLoss": DiceLoss,
    "MaskedBCELoss": MaskedBCELoss,
    "MaskedBCENormalizedLoss": MaskedBCENormalizedLoss,
    "quadratic": QuadraticBoundRegularization,
    "linear": LinearBoundRegularization,
    "MaskedNormalizedDiceLoss": MaskedNormalizedDiceLoss,
    "MSELoss": nn.MSELoss,
    "BCENormalizedLoss": BCENormalizedLoss,
    'NormalizedDiceLoss': NormalizedDiceLoss,
    "BCELoss": nn.BCELoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "SquaredHingeLoss": partial(nn.MultiMarginLoss, p=2),
}


class ArgsEnforcersCurrent(ArgsEnforcer):
    def add_enforcer(self):
        def enforce_fn(experiment):
            # if experiment.args["model"].lower() == ("BimonnBiselDenseNotBinary").lower():
            #     experiment.args["channels"] = [experiment.args["channels"][0], experiment.args["channels"][0]]

            # Duality training
            # warnings.warn('Warning, duality training.')
            # if "erosion" in experiment.args['morp_operation'].name:
            #     experiment.args['random_gen_args']['p_invert'] = 1
            # elif "dilation" in experiment.args['morp_operation'].name:
            #     experiment.args['random_gen_args']['p_invert'] = 0

            # elif "closing" in experiment.args['morp_operation'].name:
            #     experiment.args['random_gen_args']['p_invert'] = 1
            # elif "opening" in experiment.args['morp_operation'].name:
            #     experiment.args['random_gen_args']['p_invert'] = 0

            # elif "white_tophat" in experiment.args['morp_operation'].name:
            #     experiment.args['random_gen_args']['p_invert'] = 1
            # elif "black_tophat" in experiment.args['morp_operation'].name:
            #     experiment.args['random_gen_args']['p_invert'] = 0

            experiment.args['patience_loss'] = experiment.args[f"patience_loss_{experiment.args['early_stopping_on']}"]
            experiment.args['patience_reduce_lr'] = max(int(experiment.args["patience_loss"] * experiment.args['patience_reduce_lr']) - 1, 1)

            if experiment.args['atomic_element'] == "dual_bisel":
                experiment.args['weights_optim_mode'] = BiseWeightsOptimEnum.NORMALIZED

            if experiment.args['weights_optim_mode'] == BiseWeightsOptimEnum.NORMALIZED:
                experiment.args['initializer_args'].update({
                    'bise_init_method': InitBiseEnum.CUSTOM_CONSTANT_DUAL_RANDOM_BIAS,
                    'lui_init_method': InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS_DUAL_RANDOM_BIAS,
                })

            experiment.args['init_bimonn_str'] = str(experiment.args["initializer_method"])
            if isinstance(experiment.args["initializer_args"], dict):
                experiment.args['init_bise_str'] = str(experiment.args["initializer_args"]["bise_init_method"])
            elif isinstance(experiment.args["initializer_args"], list):
                experiment.args['init_bise_str'] = [str(ar["bise_init_method"]) for ar in experiment.args["initializer_args"]]

            if experiment.args['atomic_element'] == "sybisel":
                experiment.args['threshold_mode'] = {'weight': experiment.args['threshold_mode']['weight'], 'activation': experiment.args['threshold_mode']['activation'] + "_symetric"}
                experiment.args['bias_optim_mode'] = BiseBiasOptimEnum.RAW
                if experiment.args["loss_data_str"] == "BCELoss":
                    experiment.args["loss_data_str"] = "BCENormalizedLoss"

            experiment.args["kwargs_loss"] = {}
            if "Normalized" in experiment.args['loss_data_str'] and experiment.args['atomic_element'] == 'sybisel':
                experiment.args["kwargs_loss"].update({"vmin": -1, "vmax": 1})

            experiment.args['loss_data'] = loss_dict[experiment.args['loss_data_str']](**experiment.args["kwargs_loss"])

            experiment.args['loss'] = {"loss_data": experiment.args['loss_data']}

            # if isinstance(experiment.args['threshold_mode'], str) or experiment.args['threshold_mode']['weight'] != "identity":
            #     experiment.args['loss_regu'] = "None"


            # if experiment.args['loss_regu'] != "None":
            #     experiment.args['loss_regu'] = (loss_dict[experiment.args['loss_regu'][0]], experiment.args['loss_regu'][1])
            #     experiment.args['loss']['loss_regu'] = experiment.args['loss_regu']

            for key in ['closest_selem_method', 'bias_optim_mode']:
                experiment.args[f'{key}_str'] = str(experiment.args[key])

            if experiment.args['dataset'] in ['mnist_gray', 'fashionmnist']:
                assert "gray" in experiment.args['morp_operation'].name
            elif experiment.args['dataset'] in ["mnist", "diskorectdataset", "inverted_mnist"]:
                assert "gray" not in experiment.args['morp_operation'].name

        self.enforcers.append(enforce_fn)


enforcers = [[ArgsEnforcersCurrent()]]
