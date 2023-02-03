import warnings
from typing import Dict
from sys import argv
from argparse import ArgumentParser

from deep_morpho.datasets import DataModule
from deep_morpho.models import BiMoNN


class Parser(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.root_parser = ArgumentParser(add_help=False)
        self.parsers: Dict = {}

        self.root_parser.add_argument("-m", "--model", help="Model")
        self.root_parser.add_argument("-d", "--dataset", help="Dataset")

        self.given_args = set([arg for arg in argv if arg.startswith("--")])


    def _parse_args(self, parser, *args, **kwargs):
        args, _ = parser.parse_known_args(*args, **kwargs)

        self.update({k: v for k, v in args.__dict__.items() if k in self.given_args})  # CLI arguments
        self.update({k: v for k, v in args.__dict__.items() if k not in self.given_args.union(self.keys())})  # Default arguments

        return self

    def parse_args(self, *args, **kwargs):
        self._parse_args(self.root_parser, *args, **kwargs)

        model_name = self["model"].lower()
        datamodule_name = self["dataset"].lower()

        datamodule = DataModule.select(datamodule_name)
        model = BiMoNN.select(model_name)

        parser = ArgumentParser()
        seen_args = set()

        for arg_name, arg_dict in datamodule.default_args().items():
            parser.add_argument(f"--{arg_name}", **arg_dict)
            seen_args.add(arg_name)

        for arg_name, arg_dict in model.default_args().items():
            if arg_name not in seen_args:
                parser.add_argument(f"--{arg_name}", **arg_dict)
                warnings.warn(f"Argument {arg_name} is both in the datamodule and the model. The datamodule's argument will be used.")

        self._parse_args(parser, *args, **kwargs)
        return self
