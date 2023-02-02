from argparse import ArgumentParser

from deep_morpho.datasets import DataModule


class Parser(dict, ArgumentParser):

    def __init__(self):
        ArgumentParser.__init__(self)
        dict.__init__(self)

        self.subparsers = self.add_subparsers(
            help="Dataset to train on",
            dest="dataset",
            parser_class=ArgumentParser,
        )
        self.dataset_parsers = {}


        for datamodule_name in DataModule.listing():
            datamodule = DataModule.select(datamodule_name)

            self.dataset_parsers[datamodule_name] = self.subparsers.add_parser(datamodule_name)

            for arg_name, arg_dict in datamodule.default_args().items():
                print(arg_name, arg_dict)
                self.dataset_parsers[datamodule_name].add_argument(
                    f"--{arg_name}",
                    **arg_dict
                )


    def parse_args(self):
        args = ArgumentParser.parse_args(self)
        self.update(args.__dict__)

        for arg_name, arg_value in args.__dict__.items():
            if arg_value is not None:
                self[arg_name] = arg_value

        return self
