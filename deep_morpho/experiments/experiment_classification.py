from .experiment_base import ExperimentBase
from .load_observables_fn import load_observables_classification_bimonn
from .load_model_fn import load_model_bimonn_classical


class ExperimentClassification(ExperimentBase):
    """Experiment class for classificaton task."""


    def __init__(self, *args, **kwargs):
        kwargs["load_model_fn"] = load_model_bimonn_classical
        kwargs["load_observables_fn"] = load_observables_classification_bimonn
        super().__init__(*args, **kwargs)


    def enforce_args(self):
        if self.args["n_atoms"] == 'adapt':
            self.args['n_atoms'] = len(self.args['channels']) - 1

    def _check_args(self) -> None:
        super()._check_args()
        assert f"morp_operation{self.args.dataset_args_suffix}" in self.args, f"Argument {f'morp_operation{self.args.dataset_args_suffix}'} is not given"
