from logging import Logger
from typing import Any, List, Dict
from os.path import join
import os
from time import time

import torch
from pytorch_lightning.loggers import TensorBoardLogger

from deep_morpho.trainers import Trainer
from deep_morpho.datasets import DataModule
from deep_morpho.models import GenericLightningModel
from deep_morpho.utils import Parser, set_seed
from .context import Task
from general.nn.experiments.experiment_methods import ExperimentMethods
from general.utils import format_time, log_console, create_logger, save_yaml, close_handlers, get_next_same_name
from general.code_saver import CodeSaver


class ExperimentBase(ExperimentMethods):
    datamodule_class = DataModule
    model_class = GenericLightningModel
    trainer_class = Trainer

    def __init__(self, args: Parser, tb_logger: TensorBoardLogger, console_logger: Logger):
        self.args = args

        self.device = None

        self.trainer = None
        self.observables = None
        self.trainloader, self.valloader, self.testloader = None, None, None

        self.tb_logger = tb_logger
        self.console_logger = console_logger

        self._check_args()

    @property
    def log_dir(self) -> str:
        return self.tb_logger.log_dir

    def _check_args(self) -> None:
        for arg in self.necessary_args:
            assert arg in self.args, f"Argument {arg} is not given"

    def _select_model(self, model_name: str) -> GenericLightningModel:
        return self.model_class.select(model_name)

    def _select_datamodule(self, datamodule_name: str) -> DataModule:
        return self.datamodule_class.select(datamodule_name)

    @property
    def necessary_args(self) -> List[str]:
        return [
            "model", "dataset", "n_inputs", "train_test_split", "batch_size", "num_workers", "learning_rate",
            "loss", "optimizer", "optimizer_args", "reduce_loss_fn",
        ]

    def load_model(self, model_name: str) -> GenericLightningModel:
        self.model = self._select_model(model_name)(
            model_args=self.args.model_args,
            learning_rate=self.args["learning_rate"],
            loss=self.args["loss"],
            optimizer=self.args["optimizer"],
            optimizer_args=self.args["optimizer_args"],
            observables=self.observables,
            reduce_loss_fn=self.args["reduce_loss_fn"],
        )
        self.model.to(self.device)

        return self.model

    def load_datamodule(self, datamodule_name: str) -> DataModule:
        prop_train, prop_val, prop_test = self.args['train_test_split']
        self.trainloader, self.valloader, self.testloader = self._select_datamodule(datamodule_name).get_train_val_test_loader(
            batch_size=self.args["batch_size"],
            num_workers=self.args["num_workers"],
            n_inputs_train=int(prop_train * self.args['n_inputs']),
            n_inputs_val=int(prop_val * self.args['n_inputs']),
            n_inputs_test=int(prop_test * self.args['n_inputs']),
            **self.args.dataset_args,
        )

        return self.trainloader, self.valloader, self.testloader

    def load_observables(self, *args, **kwargs):
        self.observables = []
        return self

    def setup_observables(self, observables: List[Any]) -> "ExperimentBase":
        self.observables = observables
        return self

    def train(self):
        self.trainer.fit(self.model, self.trainloader, self.valloader)

    def test(self):
        self.trainer.test(self.model, self.testloader)

    def save(self):
        for observable in self.observables:
            observable.save(join(self.trainer.log_dir, 'observables'))

    def log_console(self, message: str):
        return log_console(message, logger=self.console_logger)

    def setup(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.args["seed"] = set_seed(self.args['batch_seed'])
        self.log_console(f"Seed: {self.args['seed']}")

        with Task("Loading Model", self.console_logger):
            self.load_model(self.args["model"])

        with Task("Loading Data", self.console_logger):
            self.load_datamodule(self.args["dataset"])

        with Task("Loading Observables", self.console_logger):
            self.load_observables()

        with Task("Logging model to Tensorboard", self.console_logger):
            input_ = next(iter(self.trainloader))[0]
            self.logger.experiment.add_graph(self.model, input_[0].unsqueeze(0).to(self.device))

        with Task("Logging hyperparameters to Tensorboard", self.console_logger):
            hyperparam_str = ""
            for k, v in self.args.items():
                hyperparam_str += f"**{k}**: {v}  \n"
            hyperparam_str = hyperparam_str[:-2]

            self.logger.experiment.add_text("hyperparams", hyperparam_str, global_step=0)

        self.log_console(
            f"Binarizable parameters: {self.model.model.numel_binary()} / {self.model.model.numel_float()}"
            f" = {self.model.model.numel_binary()/self.model.model.numel_float():.2f}"
        )

        self.trainer = self.trainer_class(**self.args.trainer_args)

    def run(self):
        self.setup()

        with Task("Training", self.console_logger):
            self.train()

        with Task("Testing", self.console_logger):
            self.test()

        with Task("Saving", self.console_logger):
            self.save()


class MultiExperimentBase(ExperimentMethods):
    experiment_class = ExperimentBase

    def __init__(self, multi_args: List[Dict], dest_dir: str = "deep_moprho/results/results_tensorboards"):
        self.multi_args = multi_args
        self.experiments = []

        self.console_logger = None
        self.dest_dir = dest_dir
        self.log_dir = None

    def save_code(self):
        CodeSaver(
            src_path=os.getcwd(),
            ignore_patterns=("*__pycache__*", "*results*", "data", "*.ipynb", '.git', 'ssm', 'sampling',
                            "tests", "Notebooks", "*geogebra*", ".github", ".idea", ".pytest_cache", ".vscode", "html_pages",
                            "paper_writer", "ruche_logs", ),
        ).save_directly_in_final_file(join(self.log_dir, 'code'))


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

    def setup_experiment(self, args: Dict, tb_logger: TensorBoardLogger) -> ExperimentBase:
        return self.experiment_class(args, tb_logger=tb_logger, console_logger=self.console_logger)

    def get_experiment_name(self, args: Dict) -> str:
        return join(args["dataset"], args["model"])

    def run(self):
        start_all = time()

        self.log_console('==================')
        bugged = []
        for args_idx, args in enumerate(self.multi_args):
            self.log_console('==================')
            self.log_console(f'Args number {args_idx + 1} / {len(self.multi_args)}')
            self.log_console('Time since beginning: {} '.format(format_time(time() - start_all)))

            with open(join(self.log_dir, 'state.txt'), 'w') as f:
                f.write(f'Args number {args_idx + 1} / {len(self.multi_args)}. Time since beginning: {format_time(time() - start_all)}')

            name = self.get_experiment_name(args)
            logger = TensorBoardLogger(self.log_dir, name=name, default_hp_metric=False)

            if self.console_logger is not None:
                close_handlers(self.console_logger)

            self.console_logger = create_logger(
                f'args_{args_idx}', all_logs_path=join(logger.log_dir, 'all_logs.log'), error_path=join(logger.log_dir, 'error_logs.log')
            )

            experiment = self.setup_experiment(args, logger)
            self.experiments.append(experiment)

            try:
                experiment.run()
            except Exception:
                self.console_logger.exception(
                    f'Args nb {args_idx + 1} / {len(self.multi_args)} failed : ')
                bugged.append(args_idx+1)

        with open(join(self.log_dir, 'state.txt'), 'w') as f:
            f.write(f'{len(self.multi_args)} experiments done in {format_time(time() - start_all)}')

        self.log_console(f'{len(bugged)} Args Bugged: ', bugged)
        self.log_console(f'{len(self.multi_args)} experiments done in {format_time(time() - start_all)}')
        self.log_console(f"Log dir: {self.log_dir}")

    def log_console(self, message: str):
        return log_console(message, logger=self.console_logger)
