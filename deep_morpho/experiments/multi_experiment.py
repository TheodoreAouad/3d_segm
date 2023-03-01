import os
from os.path import join
from time import time
from typing import List, Dict, Type
import pathlib

import torch

from general.utils import format_time, log_console, create_logger, close_handlers, get_next_same_name
from general.code_saver import CodeSaver
from .experiment_base import ExperimentBase
from .experiment_morpho import ExperimentMorphoBinary, ExperimentMorphoGrayScale, ExperimentDiskorect
from .experiment_classification import ExperimentClassification, ExperimentClassificationChannel
from .context import Task
from .args_enforcers import ArgsMnist, ArgsCifar, ArgsMnistClassif, ArgsMnistClassifChannel
from general.nn.experiments.experiment_methods import ExperimentMethods
from deep_morpho.models import LightningBiMoNN



class MultiExperiment(ExperimentMethods):
    MORPHO_BINARY_DATASETS = [
        "mnistmorphodataset",
        "diskorectdataset",
    ]

    MORPHO_GRAYSCALE_DATASETS = [
        "mnistgrayscaledataset",
        "fashionmnistgrayscaledataset",
    ]

    MNIST_DATASETS = [
        "fashionmnistgrayscaledataset",
        "mnistmorphodataset",
        "mnistgrayscaledataset",
        "mnistclassifdataset",
        "mnistclassifchanneldataset",
    ]

    CIFAR_DATASETS = [
        "cifar10dataset",
        "cifar100dataset",
        "cifar10classical",
        "cifar100classical",
    ]

    CLASSIFICATION_DATASETS = [
        "mnistclassifdataset",
        "mnistclassifchanneldataset",
    ]

    CLASSIFICATION_CHANNEL_DATASETS = [
        "cifar10dataset",
        "cifar100dataset",
    ]


    def __init__(
        self,
        multi_args: List[Dict],
        dest_dir: str = "deep_morpho/results/results_tensorboards",
        experiment_class: Type[ExperimentBase] = ExperimentBase,
        enforce_experiment_class: bool = False,
        stop_on_error: bool = False,
    ):
        self.multi_args = multi_args
        self.dest_dir = dest_dir
        self.experiment_class = experiment_class
        self.stop_on_error = stop_on_error
        self.enforce_experiment_class = enforce_experiment_class

        self.experiments = []
        self.console_logger = None
        self.log_dir = None

    def save_code(self):
        CodeSaver(
            src_path=os.getcwd(),
            ignore_patterns=("*__pycache__*", "*results*", "data", "*.ipynb", '.git', 'ssm', 'sampling',
                            "tests", "Notebooks", "*geogebra*", ".github", ".idea", ".pytest_cache", ".vscode", "html_pages",
                            "paper_writer", "ruche_logs", ),
        ).save_directly_in_final_file(join(self.log_dir, 'code'))

    def infer_experiment_class(self, args) -> Type[ExperimentBase]:
        if LightningBiMoNN.is_child(args["model"]):

            if args["dataset"] == "diskorectdataset":
                return ExperimentDiskorect

            if args["dataset"] in self.MORPHO_BINARY_DATASETS:
                return ExperimentMorphoBinary

            if args["dataset"] in self.MORPHO_GRAYSCALE_DATASETS:
                return ExperimentMorphoGrayScale

            if args["dataset"] in self.CLASSIFICATION_DATASETS:
                return ExperimentClassification

            if args["dataset"] in self.CLASSIFICATION_CHANNEL_DATASETS:
                return ExperimentClassificationChannel

        return self.experiment_class

    def setup(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.log_console("device", self.device)

        self.log_dir = get_next_same_name(join(self.dest_dir, self.multi_args[0]['experiment_name']))
        self.log_console("log_dir", self.log_dir)

        with Task("Saving code", self.console_logger):
            self.save_code()

    def setup_experiment(self, args: Dict, **kwargs) -> ExperimentBase:
        if self.enforce_experiment_class:
            exp_class = self.experiment_class
        else:
            exp_class = self.infer_experiment_class(args)
        experiment = exp_class(args, **kwargs)
        experiment.args_enforcers += self.infer_args_enforcers(experiment)
        return experiment


    def infer_args_enforcers(self, experiment) -> List[Dict]:
        args = experiment.args
        args_enforcers = []

        if args["dataset"] in self.MNIST_DATASETS:
            args_enforcers.append(ArgsMnist())

        if args["dataset"] in self.CIFAR_DATASETS:
            args_enforcers.append(ArgsCifar())

        if args["dataset"] == "mnistclassifdataset":
            args_enforcers.append(ArgsMnistClassif())

        if args["dataset"] == "mnistclassifchanneldataset":
            args_enforcers.append(ArgsMnistClassifChannel())

        return args_enforcers

    def generate_experiments(self, **kwargs):
        for args_idx, args in enumerate(self.multi_args):
            experiment = self.setup_experiment(args, dest_dir=self.log_dir, **kwargs)
            self.experiments.append(experiment)

    def run_experiments(self):
        start_all = time()

        self.log_console('==================')
        bugged = []
        for exp_idx, experiment in enumerate(self.experiments):
            self.log_console('==================')
            self.log_console(f'Experiment number {exp_idx + 1} / {len(self.experiments)}')
            self.log_console('Time since beginning: {} '.format(format_time(time() - start_all)))

            with open(join(self.log_dir, 'state.txt'), 'w') as f:
                f.write(f'Args number {exp_idx + 1} / {len(self.experiments)} running. Time since beginning: {format_time(time() - start_all)}')

            pathlib.Path(experiment.tb_logger.log_dir).mkdir(parents=True, exist_ok=True)

            console_logger = create_logger(
                f'experiment_{exp_idx}',
                all_logs_path=join(experiment.tb_logger.log_dir, 'all_logs.log'),
                error_path=join(experiment.tb_logger.log_dir, 'error_logs.log')
            )
            experiment.console_logger = console_logger
            self.console_logger = console_logger

            if self.stop_on_error:
                experiment.run()
            else:
                try:
                    experiment.run()
                except Exception:
                    self.console_logger.exception(
                        f'Args nb {exp_idx + 1} / {len(self.multi_args)} failed : ')
                    bugged.append(exp_idx + 1)

            close_handlers(experiment.console_logger)

        with open(join(self.log_dir, 'state.txt'), 'w') as f:
            f.write(f'{len(self.multi_args)} experiments done in {format_time(time() - start_all)}')

        self.log_console(f'{len(bugged)} Experiments Bugged: ', bugged)
        self.log_console(f'{len(self.experiments)} experiments done in {format_time(time() - start_all)}')
        self.log_console(f"Log dir: {self.log_dir}")

    def log_console(self, *args, **kwargs):
        return log_console(*args, **kwargs, logger=self.console_logger)