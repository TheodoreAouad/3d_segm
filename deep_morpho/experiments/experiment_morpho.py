from typing import List
import pathlib
from .context import Task
from os.path import join

import matplotlib.pyplot as plt

from .experiment_base import ExperimentBase
from .load_observables_fn import load_observables_morpho_binary, load_observables_morpho_grayscale


class ExperimentMorphoBase(ExperimentBase):
    """Experiment class for learning morphological operators."""

    @property
    def necessary_args(self) -> List[str]:
        res = super().necessary_args
        res.extend([f"morp_operation{self.args.dataset_args_suffix}"])
        return res

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
        super().__init__(load_observables_fn=load_observables_morpho_binary, *args, **kwargs)


class ExperimentMorphoGrayScale(ExperimentMorphoBase):
    def __init__(self, *args, **kwargs):
        super().__init__(load_observables_fn=load_observables_morpho_grayscale, *args, **kwargs)
