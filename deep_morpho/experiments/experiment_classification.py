from collections import namedtuple

import torch

from .experiment_base import ExperimentBase
from .load_observables_fn import load_observables_classification_bimonn, load_observables_classification_channel_bimonn
from .load_model_fn import load_model_bimonn_classical_classification
from .enforcers import ArgsClassification, ArgsClassifChannel, ArgsClassifActivation, ArgsGeneration
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
        kwargs["args_enforcers"] = kwargs.get("args_enforcers", []) + [ArgsGeneration()]
        super().__init__(*args, **kwargs)

    def log_tensorboard_graph(self):
        # class ModelWrapper(torch.nn.Module): 
        #     """ Wrapper class for model with dict/list rvalues. """
        #     def __init__(self, model: torch.nn.Module) -> None:
        #         """
        #         Init call.
        #         """
        #         super().__init__()
        #         self.model = model

        #     def forward(self, input_x: torch.Tensor):
        #         """
        #         Wrap forward call.
        #         """
        #         data = self.model(input_x)

        #         if isinstance(data, dict):
        #             data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
        #             data = data_named_tuple(**data)  # type: ignore

        #         elif isinstance(data, list):
        #             data = tuple(data)

        #         return data

        with Task("Logging model to Tensorboard", self.console_logger, verbose=self.verbose):
            x, segm = self.input_sample
            x = x[0].unsqueeze(0).to(self.device)
            segm = segm[0].unsqueeze(0).to(self.device)
            self.tb_logger.experiment.add_graph(self.model, ((x, segm),))
