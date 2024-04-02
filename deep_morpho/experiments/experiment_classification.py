from collections import namedtuple

import torch

from .experiment_base import ExperimentBase
from .load_observables_fn import (
    load_observables_classification_bimonn, load_observables_classification_channel_bimonn,
    load_observables_bimonn_axspa_spalike, load_observables_bimonn_axspa_spalike_merged,
    load_observables_bimonn_desir_merged
)
from .load_model_fn import load_model_bimonn_classical_classification
from .enforcers import (
    ArgsClassification, ArgsClassifChannel, ArgsClassifActivation, ArgsSpalike, ArgsDesirMerged, ArgsDesirRoiChan
)
from .context import Task


class ExperimentClassification(ExperimentBase):
    """Experiment class for classificaton task."""


    def __init__(self, *args, **kwargs):
        # kwargs["load_model_fn"] = load_model_bimonn_classical_classification
        kwargs["load_observables_fn"] = load_observables_classification_bimonn
        kwargs["args_enforcers"] = kwargs.get("args_enforcers", []) + [ArgsClassification(), ArgsClassifActivation()]
        super().__init__(*args, **kwargs)


class ExperimentClassificationChannel(ExperimentBase):
    def __init__(self, *args, **kwargs):
        # kwargs["load_model_fn"] = load_model_bimonn_classical_classification
        kwargs["load_observables_fn"] = load_observables_classification_channel_bimonn
        kwargs["args_enforcers"] = kwargs.get("args_enforcers", []) + [ArgsClassification(), ArgsClassifChannel(), ArgsClassifActivation()]
        super().__init__(*args, **kwargs)


class ExperimentSpalike(ExperimentBase):
    def __init__(self, *args, **kwargs):
        kwargs["load_observables_fn"] = load_observables_bimonn_axspa_spalike
        kwargs["args_enforcers"] = kwargs.get("args_enforcers", []) + [ArgsSpalike()]
        super().__init__(*args, **kwargs)

    def log_tensorboard_graph(self):
        with Task("Logging model to Tensorboard", self.console_logger, verbose=self.verbose):
            x, segm = self.input_sample
            x = x[0].unsqueeze(0).to(self.device)
            segm = segm[0].unsqueeze(0).to(self.device)
            self.tb_logger.experiment.add_graph(self.model, ((x, segm),))


class ExperimentSpalikeMerged(ExperimentBase):
    def __init__(self, *args, **kwargs):
        kwargs["load_observables_fn"] = load_observables_bimonn_axspa_spalike_merged
        kwargs["args_enforcers"] = kwargs.get("args_enforcers", []) + [ArgsSpalike()]
        super().__init__(*args, **kwargs)

    # def log_tensorboard_graph(self):
    #     with Task("Logging model to Tensorboard", self.console_logger, verbose=self.verbose):
    #         x, segm = self.input_sample
    #         x = x[0].unsqueeze(0).to(self.device)
    #         segm = segm[0].unsqueeze(0).to(self.device)
    #         self.tb_logger.experiment.add_graph(self.model, ((x, segm),))


class ExperimentDesirMerged(ExperimentBase):
    def __init__(self, *args, **kwargs):
        kwargs["load_observables_fn"] = load_observables_bimonn_desir_merged
        kwargs["args_enforcers"] = kwargs.get("args_enforcers", []) + [ArgsDesirMerged()]
        super().__init__(*args, **kwargs)


class ExperimentDesirRoiChan(ExperimentBase):
    def __init__(self, *args, **kwargs):
        kwargs["load_observables_fn"] = load_observables_bimonn_desir_merged
        kwargs["args_enforcers"] = kwargs.get("args_enforcers", []) + [ArgsDesirRoiChan()]
        super().__init__(*args, **kwargs)
