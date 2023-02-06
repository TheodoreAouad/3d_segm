import sys

from mock import patch
import pytest

from deep_morpho.binarization.bise_closest_selem import ClosestSelemEnum
from deep_morpho.models.bise import BiseWeightsOptimEnum


class TestParser:
    @staticmethod
    def test_parser(mocker):
        def mock_init(self, arg1="arg1", arg2="banane", kernel_size=[1, 2]):
            pass

        mocker.patch("deep_morpho.models.BiMoNN.__init__", mock_init)
        from deep_morpho.experiments.parser import Parser

        prs = Parser()
        prs["dataset"] = "cifar10dataset"
        prs["model"] = "BiMoNN"
        prs["net.kernel_size"] = [3, 3, 3]


        prs.parse_args("--atomic_element bisel".split())

        assert prs["dataset"] == "cifar10dataset"
        assert prs["model"] == "BiMoNN"

        assert prs["net.kernel_size"] == [3, 3, 3]
        assert prs["atomic_element"] == "bisel"
        assert prs["net.arg1"] == "arg1"
        assert prs["net.arg2"] == "banane"

        # assert prs.model_keys() == ["net.arg1", "net.arg2", "net.kernel_size"]
        assert set(prs.model_keys()) == {"net.arg1", "net.arg2", "net.kernel_size"}
        assert prs.model_args() == {"arg1": "arg1", "arg2": "banane", "kernel_size": [3, 3, 3]}


    @staticmethod
    def test_cli_args(mocker):
        testargs = ["file.py", "--dataset", "cifar10dataset", "--model", "BiMoNN"]
        mocker.patch("sys.argv", testargs)
        from deep_morpho.experiments.parser import Parser

        prs = Parser()
        prs.parse_args()
        assert prs["model"] == "BiMoNN"
        assert prs["dataset"] == "cifar10dataset"

    @staticmethod
    def test_parse_args():
        # testargs = ["file.py", "--dataset", "cifar10dataset", "--model", "BiMoNN"]
        # mocker.patch("sys.argv", testargs)
        from deep_morpho.experiments.parser import Parser

        prs = Parser(["--dataset", "cifar10dataset", "--model", "BiMoNN"])
        prs.parse_args()
        assert prs["model"] == "BiMoNN"
        assert prs["dataset"] == "cifar10dataset"



    @staticmethod
    def test_cli_and_new_args(mocker):
        testargs = ["file.py", "--dataset", "cifar10dataset", "--model", "BiMoNN", "--some_arg", "banan"]
        mocker.patch("sys.argv", testargs)
        from deep_morpho.experiments.parser import Parser

        prs = Parser()
        prs.parse_args()
        assert prs["model"] == "BiMoNN"
        assert prs["dataset"] == "cifar10dataset"
        assert prs["some_arg"] == "banan"

    @staticmethod
    def test_cli_and_default_args1(mocker):
        def mock_model_init(self, arg1="arg1", arg2="banane", kernel_size=[1, 2]):
            pass

        def mock_dataset_init(self, arg1="arg1", arg2="banane", n_inputs="all"):
            pass

        mocker.patch("deep_morpho.models.BiMoNN.__init__", mock_model_init)
        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR10Dataset.__init__", mock_dataset_init)

        testargs = ["file.py", "--dataset", "cifar10dataset", "--model", "BiMoNN"]
        mocker.patch("sys.argv", testargs)

        from deep_morpho.experiments.parser import Parser

        prs = Parser()
        prs["model"] = "BiMoNN"
        prs["dataset"] = "cifar10dataset"
        prs.parse_args()
        assert prs["model"] == "BiMoNN"
        assert prs["dataset"] == "cifar10dataset"
        assert prs["arg1.net"] == "arg1"
        assert prs["arg2.net"] == "banane"
        assert prs["kernel_size.net"] == [1, 2]
        assert prs["arg1.datamodule"] == "arg1"
        assert prs["arg2.datamodule"] == "banane"
        assert prs["n_inputs.datamodule"] == "all"

    @staticmethod
    def test_cli_and_default_args2(mocker):
        def mock_model_init(self, arg1="arg1", arg2="banane", kernel_size=[1, 2]):
            pass

        def mock_dataset_init(self, arg1="arg1", arg2="banane", n_inputs="all"):
            pass

        mocker.patch("deep_morpho.models.BiMoNN.__init__", mock_model_init)
        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR10Dataset.__init__", mock_dataset_init)

        testargs = ["file.py", "--dataset", "cifar10dataset", "--model", "BiMoNN", "--arg1.net", "banane1"]
        mocker.patch("sys.argv", testargs)

        from deep_morpho.experiments.parser import Parser

        prs = Parser()
        prs["model"] = "BiMoNN"
        prs["dataset"] = "cifar10dataset"
        prs.parse_args()
        assert prs["model"] == "BiMoNN"
        assert prs["dataset"] == "cifar10dataset"
        assert prs["arg1.net"] == "banane1"
        assert prs["arg2.net"] == "banane"
        assert prs["kernel_size.net"] == [1, 2]
        assert prs["arg1.datamodule"] == "arg1"
        assert prs["arg2.datamodule"] == "banane"
        assert prs["n_inputs.datamodule"] == "all"

    @staticmethod
    def test_cli_abbrev(mocker):
        def mock_model_init(self, arg1="arg1", arg2="banane", kernel_size=[1, 2]):
            pass

        def mock_dataset_init(self, n_inputs="all"):
            pass

        mocker.patch("deep_morpho.models.BiMoNN.__init__", mock_model_init)
        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR10Dataset.__init__", mock_dataset_init)

        testargs = ["file.py", "--dataset", "cifar10dataset", "--model", "BiMoNN", "--n_inputs", "1000"]
        mocker.patch("sys.argv", testargs)

        from deep_morpho.experiments.parser import Parser

        prs = Parser()
        prs["model"] = "BiMoNN"
        prs["dataset"] = "cifar10dataset"
        prs.parse_args()
        assert prs["model"] == "BiMoNN"
        assert prs["dataset"] == "cifar10dataset"
        assert prs["n_inputs.datamodule"] == "1000"

    @staticmethod
    def test_dict_args(mocker):
        def mock_model_init(self, arg1="arg1", arg2="banane", kernel_size=[1, 2]):
            pass

        def mock_dataset_init(self, n_inputs="all"):
            pass

        mocker.patch("deep_morpho.models.BiMoNN.__init__", mock_model_init)
        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR10Dataset.__init__", mock_dataset_init)

        testargs = ["file.py", "--n_inputs", "1000"]
        mocker.patch("sys.argv", testargs)

        from deep_morpho.experiments.parser import Parser

        prs = Parser()
        prs["model"] = "BiMoNN"
        prs["dataset"] = "cifar10dataset"
        prs.parse_args()
        assert prs["model"] == "BiMoNN"
        assert prs["dataset"] == "cifar10dataset"
        assert prs["n_inputs.datamodule"] == "1000"
        assert prs["arg1.net"] == "arg1"

    @staticmethod
    def test_cli_over_dict(mocker):
        testargs = ["file.py", "--dataset", "mnistgrayscaledataset", "--model", "bimonnclassifier"]
        mocker.patch("sys.argv", testargs)
        from deep_morpho.experiments.parser import Parser

        prs = Parser()
        prs["model"] = "BiMoNN"
        prs["dataset"] = "cifar10dataset"
        prs.parse_args()

        assert prs["model"] == "bimonnclassifier"
        assert prs["dataset"] == "mnistgrayscaledataset"

    @staticmethod
    def test_same_arg_abbrev_error(mocker):
        def mock_model_init(self, arg1="arg1",):
            pass

        def mock_dataset_init(self, arg1="all"):
            pass

        mocker.patch("deep_morpho.models.BiMoNN.__init__", mock_model_init)
        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR10Dataset.__init__", mock_dataset_init)

        from deep_morpho.experiments.parser import Parser
        prs = Parser()
        # prs.parse_args(["--model", "BiMoNN", "--dataset", "cifar10dataset", "--arg1", "1000"])
        with pytest.raises(SystemExit, match="2"):
            prs.parse_args(["--model", "BiMoNN", "--dataset", "cifar10dataset", "--arg1", "1000"])

    @staticmethod
    def test_model_args1():
        from deep_morpho.experiments.parser import Parser
        prs = Parser().parse_args(""
            '--model BiMoNN --dataset cifar10dataset --atomic_element bisel --constant_P_lui True --closest_selem_method MIN_DIST_DIST_TO_CST  --weights_optim_mode NORMALIZED'
            "".split(" ")
        )
        assert prs["model"] == "BiMoNN"
        assert prs["dataset"] == "cifar10dataset"
        assert prs["atomic_element.net"] == "bisel"
        assert prs["constant_P_lui.net"] == True
        assert prs["closest_selem_method.net"] == ClosestSelemEnum.MIN_DIST_DIST_TO_CST
        assert prs["weights_optim_mode.net"] == BiseWeightsOptimEnum.NORMALIZED

    @staticmethod
    def test_unknown_args():
        from deep_morpho.experiments.parser import Parser
        prs = Parser().parse_args(""
            '--model BiMoNN --dataset cifar10dataset --epoch 20'
            "".split(" ")
        )

        assert prs["model"] == "BiMoNN"
        assert prs["dataset"] == "cifar10dataset"
        assert prs["epoch"] == 20
