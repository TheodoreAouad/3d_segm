import warnings
from typing import Dict, List
from sys import argv
from argparse import ArgumentParser, Namespace

from deep_morpho.datasets import DataModule
from deep_morpho.models import BiMoNN

from general.utils import dict_cross


class Parser(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.root_parser = ArgumentParser(allow_abbrev=False)
        self.parsers: Dict = {}

        self.root_parser.add_argument("--model", help="Model")
        self.root_parser.add_argument("--dataset", help="Dataset")

        self.given_args = None
        # self.given_args = set(["model", "dataset"])

    def _parse_args(
        self,
        parser,
        args=None,
        namespace=None,
        add_known: bool = True,
        add_unknown: bool = True,
        add_default: bool = True,
    ) -> "Parser":
        args_known, args_unknown = parser.parse_known_args(args, namespace)

        self.add_cli_args(args_known, args_unknown, add_known=add_known, add_unknown=add_unknown)
        if add_default:
            self.add_default_args(args_known)

        return self

    @staticmethod
    def _parse_unkown(args) -> Namespace:
        parser = ArgumentParser()
        for arg in args:
            if arg.startswith("--"):
                parser.add_argument(arg)
        return parser.parse_known_args(args)[0]

    def add_cli_args(self, args_known: Namespace, args_unknown: List[str], add_known: bool = True, add_unknown: bool = True) -> None:
        if add_known:
            self.update({k: v for k, v in args_known.__dict__.items() if k in self.given_args})  # known args
        if add_unknown:
            self.update(self._parse_unkown(args_unknown).__dict__)  # unknown args

    def add_default_args(self, args: Namespace) -> None:
        self.update({k: v for k, v in args.__dict__.items() if k not in self.given_args.union(self.keys())})

    def parse_args(self, args=None, namespace=None, add_argv: bool = True) -> "Parser":
        if add_argv:
            if args is not None:
                args += argv[1:]
            else:
                args = argv[1:]

        # self.given_args.update([arg[2:] for arg in args if arg.startswith("--")])
        self.given_args = set([arg[2:] for arg in args if arg.startswith("--")])

        self._parse_args(self.root_parser, args, namespace, add_unknown=False, add_default=False)  # parse dataset and model

        assert "model" in self, "Model not specified"
        assert "dataset" in self, "Dataset not specified"

        model_name = self["model"].lower()
        datamodule_name = self["dataset"].lower()

        datamodule = DataModule.select(datamodule_name)
        model = BiMoNN.select(model_name)

        parser = ArgumentParser()
        # seen_args = set()  # Handle cases where the same args is used for dataset and model

        # We use suffix instead of prefix because the abbrev of argparse matches the beginning of the str.
        for arg_name, arg_dict in datamodule.default_args().items():
            parser.add_argument(f"--{arg_name}{self.dataset_args_suffix}", **arg_dict)
            # seen_args.add(arg_name)

        for arg_name, arg_dict in model.default_args().items():
            parser.add_argument(f"--{arg_name}{self.model_args_suffix}", **arg_dict)
            # if arg_name not in seen_args:
            # else:
            #     warnings.warn(f"Argument {arg_name} is both in the datamodule and the model. The datamodule's argument will be used.")

        self._parse_args(parser, args, namespace)  # parse remaining args
        return self


    @property
    def dataset_args_suffix(self) -> str:
        return ".datamodule"

    @property
    def model_args_suffix(self) -> str:
        return ".net"

    def dataset_keys(self) -> List[str]:
        return [k for k in self.keys() if k.endswith(self.dataset_args_suffix)]

    def model_keys(self) -> List[str]:
        return [k for k in self.keys() if k.endswith(self.model_args_suffix)]

    def dataset_args(self) -> Dict:
        return {k[:-len(self.dataset_args_suffix)]: self[k] for k in self.dataset_keys()}

    def model_args(self) -> Dict:
        return {k[:-len(self.model_args_suffix)]: self[k] for k in self.model_keys()}


class MultiParser(Parser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_args = None

    def add_default_args(self, args):
        self.update({k: [v] for k, v in args.__dict__.items() if k not in self.given_args.union(self.keys())})

    def cross_args(self):
        self.multi_args = dict_cross(self)

    def post_process(self):
        return self
