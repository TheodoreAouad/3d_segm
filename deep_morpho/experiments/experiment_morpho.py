from typing import List
import pathlib
from .context import Task
from os.path import join

import matplotlib.pyplot as plt

from .experiment_base import ExperimentBase
from .load_observables_fn import load_observables_morpho_binary, load_observables_morpho_grayscale


class ExperimentMorphoBase(ExperimentBase):
    """Experiment class for learning morphological operators."""

    # @property
    # def necessary_args(self) -> List[str]:
    #     res = super().necessary_args
    #     res.extend([f"morp_operation{self.args.dataset_args_suffix}"])
    #     return res

    def enforce_args(self):
        if self.args["kernel_size"] == "adapt":
            self.args["kernel_size"] = int(max(self.args['morp_operation'].max_selem_shape))

        if self.args['channels'] == 'adapt':
            self.args['channels'] = self.args['morp_operation'].in_channels + [self.args['morp_operation'].out_channels[-1]]

        if self.args["n_atoms"] == 'adapt':
            self.args['n_atoms'] = len(self.args['morp_operation'])

    def _check_args(self) -> None:
        super()._check_args()
        assert f"morp_operation{self.args.dataset_args_suffix}" in self.args, f"Argument {f'morp_operation{self.args.dataset_args_suffix}'} is not given"

    def log_tensorboard(self):
        super().log_tensorboard()

        with Task("Logging morphological operations to Tensorboard", self.console_logger):
            self._log_morp_operations()

    def _log_morp_operations(self):
        pathlib.Path(join(self.log_dir, "target_SE")).mkdir(exist_ok=True, parents=True)
        figs_selems = self.args['morp_operation'].plot_selem_arrays()
        for (layer_idx, chan_input, chan_output), fig in figs_selems.items():
            fig.savefig(join(self.log_dir, "target_SE", f"target_SE_l_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
            self.tb_logger.experiment.add_figure(f"target_SE/target_SE_l_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig)
            plt.close(fig)

        pathlib.Path(join(self.log_dir, "target_UI")).mkdir(exist_ok=True, parents=True)
        figs_ui = self.args['morp_operation'].plot_ui_arrays()
        for (layer_idx, chan_output), fig in figs_ui.items():
            fig.savefig(join(self.log_dir, "target_UI", f"target_UI_l_{layer_idx}_chin_chout_{chan_output}.png"))
            self.tb_logger.experiment.add_figure(f"target_UI/target_UI_l_{layer_idx}_chin_chout_{chan_output}", fig)
            plt.close(fig)

        pathlib.Path(join(self.log_dir, "morp_operations")).mkdir(exist_ok=True, parents=True)
        fig_morp_operation = self.args['morp_operation'].vizualise().fig
        fig_morp_operation.savefig(join(self.log_dir, "morp_operations", "morp_operations.png"))
        self.tb_logger.experiment.add_figure("target_operations/morp_operations", fig_morp_operation)
        plt.close(fig_morp_operation)

    def get_experiment_name(self) -> str:
        name = super().get_experiment_name()
        name = join(name, self.args['morp_operation'].name)
        return name


class ExperimentMorphoBinary(ExperimentMorphoBase):
    def __init__(self, *args, **kwargs):
        kwargs["load_observables_fn"] = load_observables_morpho_binary
        super().__init__(*args, **kwargs)


class ExperimentMorphoGrayScale(ExperimentMorphoBase):
    def __init__(self, *args, **kwargs):
        kwargs["load_observables_fn"] = load_observables_morpho_grayscale
        super().__init__(*args, **kwargs)


class ExperimentDiskorect(ExperimentMorphoBinary):
    def enforce_args(self):
        super().enforce_args()

        self.args["n_inputs_train"] = self.args['n_steps'] * self.args['batch_size']
        self.args["n_inputs_val"] = self.args["batch_size"]
        self.args["n_inputs_test"] = self.args["batch_size"]

        self.args["random_gen_args"] = self.args["random_gen_args"].copy()
        # args["random_gen_args"]["border"] = (args["kernel_size"]//2 + 1, args["kernel_size"]//2 + 1)
        self.args['random_gen_args']['size'] = self.args['random_gen_args']['size'] + (self.args["morp_operation"].in_channels[0],)
