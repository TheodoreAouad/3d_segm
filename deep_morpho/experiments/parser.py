import warnings
from typing import Dict, List, Any
from sys import argv
from argparse import ArgumentParser, Namespace, Action

from deep_morpho.datasets import DataModule
from deep_morpho.models import BiMoNN
from deep_morpho.trainers.trainer import Trainer

from general.utils import dict_cross


class Parser(dict):
    trainer_class = Trainer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.root_parser = ArgumentParser(allow_abbrev=False)
        self.parsers: Dict = {}

        self.given_args = set()

        # Track if an argument is given in CLI
        class ArgIsStored(Action):
            def __call__(self2, parser, namespace, values, option_string=None):
                self.given_args.add(self2.dest)
                setattr(namespace, self2.dest, values)
        self.arg_is_stored_class = ArgIsStored

        self.parser_add_argument(self.root_parser, "model", help="Model")
        self.parser_add_argument(self.root_parser, "dataset", help="Dataset")

    def _parse_args_to_dict(
        self,
        dict_,
        parser,
        args=None,
        namespace=None,
        add_known: bool = True,
        add_unknown: bool = True,
        add_default: bool = True,
    ) -> "Parser":
        args_known, args_unknown = parser.parse_known_args(args, namespace)

        self.add_cli_args_to_dict(dict_, args_known, args_unknown, add_known=add_known, add_unknown=add_unknown)
        if add_default:
            self.add_default_args_to_dict(dict_, args_known)

        return self

    def _parse_unknown(self, args) -> Namespace:
        parser = ArgumentParser()
        for arg in args:
            if arg.startswith("--"):
                # parser.add_argument(arg)
                self.parser_add_argument(parser, arg[2:])
        return parser.parse_known_args(args)[0]

    def add_cli_args_to_dict(self, dict_, args_known: Namespace, args_unknown: List[str], add_known: bool = True, add_unknown: bool = True) -> None:
        if add_known:
            dict_.update({k: v for k, v in args_known.__dict__.items() if k in self.given_args})  # known args
        if add_unknown:
            dict_.update(self._parse_unknown(args_unknown).__dict__)  # unknown args

    def add_default_args_to_dict(self, dict_, args: Namespace) -> None:
        dict_.update({k: v for k, v in args.__dict__.items() if k not in self.given_args.union(self.keys())})

    def parse_args(self, args=None, namespace=None, add_argv: bool = True) -> "Parser":
        if add_argv:
            if args is not None:
                args += argv[1:]
            else:
                args = argv[1:]

        # self.given_args = set([arg[2:] for arg in args if arg.startswith("--")])

        self._parse_args_to_dict(dict_=self, parser=self.root_parser, args=args, namespace=namespace, add_unknown=False, add_default=False)  # parse dataset and model

        assert "model" in self, "Model not specified"
        assert "dataset" in self, "Dataset not specified"

        model_name = self["model"].lower()
        datamodule_name = self["dataset"].lower()

        datamodule = DataModule.select(datamodule_name)
        model = BiMoNN.select(model_name)

        parser = ArgumentParser()

        # We use suffix instead of prefix because the abbrev of argparse matches the beginning of the str.
        for arg_name, arg_dict in datamodule.default_args().items():
            self.parser_add_argument(parser=parser, name=f"{arg_name}{self.dataset_args_suffix}", **arg_dict)

        for arg_name, arg_dict in model.default_args().items():
            self.parser_add_argument(parser=parser, name=f"{arg_name}{self.model_args_suffix}", **arg_dict)

        for arg_name, arg_dict in self.trainer_class.default_args().items():
            self.parser_add_argument(parser=parser, name=f"{arg_name}{self.trainer_args_suffix}", **arg_dict)

        self._parse_args_to_dict(dict_=self, parser=parser, args=args, namespace=namespace)  # parse remaining args
        return self

    def parser_add_argument(self, parser, name: str, **kwargs):
        parser.add_argument(f"--{name}", action=self.arg_is_stored_class, **kwargs)

    @property
    def dataset_args_suffix(self) -> str:
        return ".datamodule"

    @property
    def model_args_suffix(self) -> str:
        return ".net"

    @property
    def trainer_args_suffix(self) -> str:
        return ".trainer"

    def dataset_keys(self) -> List[str]:
        return [k for k in self.keys() if k.endswith(self.dataset_args_suffix)]

    def model_keys(self) -> List[str]:
        return [k for k in self.keys() if k.endswith(self.model_args_suffix)]

    def trainer_keys(self) -> List[str]:
        return [k for k in self.keys() if k.endswith(self.trainer_args_suffix)]

    def dataset_args(self) -> Dict:
        return {k[:-len(self.dataset_args_suffix)]: self[k] for k in self.dataset_keys()}

    def model_args(self) -> Dict:
        return {k[:-len(self.model_args_suffix)]: self[k] for k in self.model_keys()}

    def trainer_args(self) -> Dict:
        return {k[:-len(self.trainer_args_suffix)]: self[k] for k in self.trainer_keys()}

    def __setitem__(self, key: str, value: Any):
        if not isinstance(key, str):
            raise TypeError("Key must be a string")

        if len(key) == 0:
            raise ValueError("Key must not be empty")

        if key not in self:
            is_dataset_arg = key + self.dataset_args_suffix in self
            is_model_arg = key + self.model_args_suffix in self
            is_trainer_arg = key + self.trainer_args_suffix in self

            if is_dataset_arg + is_model_arg + is_trainer_arg == 1:
                if is_dataset_arg:
                    key += self.dataset_args_suffix

                if is_model_arg:
                    key += self.model_args_suffix

                if is_trainer_arg:
                    key += self.trainer_args_suffix

        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Any:
        if not isinstance(key, str):
            raise TypeError("Key must be a string")

        if len(key) == 0:
            raise ValueError("Key must not be empty")

        if key not in self:
            is_dataset_arg = key + self.dataset_args_suffix in self
            is_model_arg = key + self.model_args_suffix in self
            is_trainer_arg = key + self.trainer_args_suffix in self

            if is_dataset_arg + is_model_arg + is_trainer_arg > 1:
                raise ValueError(f"Key {key} is ambiguous. It could be a dataset, model or trainer arg.")

            if is_dataset_arg:
                key += self.dataset_args_suffix

            elif is_model_arg:
                key += self.model_args_suffix

            elif is_trainer_arg:
                key += self.trainer_args_suffix

        return super().__getitem__(key)


class MultiParser(Parser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_args = None
        self.all_given_args = set()

    def add_default_args_to_dict(self, dict_, args):
        dict_.update({k: [v] for k, v in args.__dict__.items() if k not in self.given_args.union(self.keys())})

    def parser_add_argument(self, parser, name: str, **kwargs):
        parser.add_argument(f"--{name}", nargs="+", action=self.arg_is_stored_class, **kwargs)

    def post_process(self):
        return self

    def parse_args(self, args=None, namespace=None, add_argv: bool = True) -> "Parser":
        if add_argv:
            if args is not None:
                args += argv[1:]
            else:
                args = argv[1:]

        self.given_args = set([arg[2:] for arg in args if arg.startswith("--")])

        self._parse_args_to_dict(self, self.root_parser, args, namespace, add_unknown=False, add_default=False)  # parse dataset and model

        assert "model" in self, "Model not specified"
        assert "dataset" in self, "Dataset not specified"

        self.multi_args = []

        for model_name in self["model"]:
            for datamodule_name in self["dataset"]:

                new_dict = self.copy()
                self.given_args = set()

                model_name = model_name.lower()
                datamodule_name = datamodule_name.lower()

                datamodule = DataModule.select(datamodule_name)
                model = BiMoNN.select(model_name)

                parser = ArgumentParser()

                # We use suffix instead of prefix because the abbrev of argparse matches the beginning of the str.
                for arg_name, arg_dict in datamodule.default_args().items():
                    self.parser_add_argument(parser, f"{arg_name}{self.dataset_args_suffix}", **arg_dict)

                for arg_name, arg_dict in model.default_args().items():
                    self.parser_add_argument(parser, f"{arg_name}{self.model_args_suffix}", **arg_dict)

                for arg_name, arg_dict in self.trainer_class.default_args().items():
                    self.parser_add_argument(parser, f"{arg_name}{self.trainer_args_suffix}", **arg_dict)

                self._parse_args_to_dict(new_dict, parser, args, namespace)  # parse remaining args

                new_dict["model"] = [model_name]
                new_dict["dataset"] = [datamodule_name]

                new_args = [Parser(d) for d in dict_cross(new_dict)]
                for arg in new_args:
                    arg.given_args = set(self.given_args)

                self.all_given_args.update(self.given_args)

                self.multi_args += new_args



        return self

    # def __getitem__(self, idx):
    #     return self.multi_args[idx]

    def __len__(self):
        return len(self.multi_args)
