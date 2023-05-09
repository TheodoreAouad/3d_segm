# import sys

from mock import patch
import pytest


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
        prs["kernel_size.net"] = [3, 3, 3]


        prs.parse_args("--atomic_element bisel".split())

        assert prs["dataset"] == "cifar10dataset"
        assert prs["model"] == "BiMoNN"

        assert prs["kernel_size.net"] == [3, 3, 3]
        assert prs["atomic_element"] == "bisel"
        assert prs["arg1.net"] == "arg1"
        assert prs["arg2.net"] == "banane"

        # assert prs.model_keys() == ["net.arg1", "net.arg2", "net.kernel_size"]
        # assert set(prs.model_keys()) == {"arg1.net", "arg2.net", "kernel_size.net"}
        # assert prs.model_args() == {"arg1": "arg1", "arg2": "banane", "kernel_size": [3, 3, 3]}


    @staticmethod
    def test_cli_args(mocker):
        testargs = ["file.py", "--dataset", "cifar10dataset", "--model", "BiMoNN"]
        mocker.patch("sys.argv", testargs)
        from deep_morpho.experiments.parser import Parser

        prs = Parser()
        prs.parse_args(add_argv=False)
        assert prs["model"] == "BiMoNN"
        assert prs["dataset"] == "cifar10dataset"

    @staticmethod
    def test_parse_args():
        # testargs = ["file.py", "--dataset", "cifar10dataset", "--model", "BiMoNN"]
        # mocker.patch("sys.argv", testargs)
        from deep_morpho.experiments.parser import Parser

        prs = Parser()
        prs.parse_args(["--dataset", "cifar10dataset", "--model", "BiMoNN"])
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

        def mock_dataset_init(self, n_inputs="alliori"):
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
        import sys

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
        from deep_morpho.binarization.bise_closest_selem import ClosestSelemEnum
        from deep_morpho.models.bise import BiseWeightsOptimEnum

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
        assert prs["epoch"] == '20'


    @staticmethod
    def test_dict_to_parse():
        from deep_morpho.experiments.parser import Parser
        prs = Parser()

        prs["dataset"] = "cifar10dataset"
        prs["model"] = "BiMoNN"

    @staticmethod
    def test_key_assignment(mocker):
        def mock_model_init(self, arg1="arg1",):
            pass

        def mock_dataset_init(self, bbl="all"):
            pass


        mocker.patch("deep_morpho.models.BiMoNN.__init__", mock_model_init)
        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR10Dataset.__init__", mock_dataset_init)

        from deep_morpho.experiments.parser import Parser
        prs = Parser()

        prs["dataset"] = "cifar10dataset"
        prs["model"] = "BiMoNN"

        prs.parse_args()

        prs["arg1"] = "value"
        prs["bbl"] = "second"

        assert prs["dataset"] == "cifar10dataset"
        assert prs["model"] == "BiMoNN"
        assert prs["arg1.net"] == "value"
        assert prs["bbl.datamodule"] == "second"


    @staticmethod
    def test_given_args(mocker):
        testargs = ["file.py", "--dataset", "mnistgrayscaledataset", "--model", "bimonnclassifier", "--arg1", "1000"]
        mocker.patch("sys.argv", testargs)


        def mock_model_init(self, arg1="arg1",):
            pass

        def mock_dataset_init(self, bbl="all"):
            pass


        mocker.patch("deep_morpho.models.BiMoNN.__init__", mock_model_init)
        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR10Dataset.__init__", mock_dataset_init)

        from deep_morpho.experiments.parser import Parser

        prs = Parser()
        prs.parse_args()

        assert prs.given_args == set(["dataset", "model", "arg1.net"])

    @staticmethod
    def test_trainset_args(mocker):
        def mock_dataset_init(self, bbl="all"):
            pass

        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR10Dataset.__init__", mock_dataset_init)

        from deep_morpho.experiments.parser import Parser

        prs = Parser()
        prs["bbl.train"] = "train"
        prs["bbl.val"] = "val"
        prs["bbl"] = "test"
        prs["model"] = "BiMoNN"
        prs["dataset"] = "cifar10dataset"
        prs.parse_args([])

        assert prs.trainset_args()["bbl"] == "train"
        assert prs.valset_args()["bbl"] == "val"
        assert prs.testset_args()["bbl"] == "test"



class TestMultiParser:

    @staticmethod
    def test_cli_args(mocker):
        testargs = ["file.py", "--dataset", "cifar10dataset", "--model", "BiMoNN", "bimonnclassifier"]
        mocker.patch("sys.argv", testargs)
        from deep_morpho.experiments.parser import MultiParser

        prs = MultiParser()
        prs.parse_args()
        assert prs["model"] == ["BiMoNN", "bimonnclassifier"]
        assert prs["dataset"] == ["cifar10dataset"]

    @staticmethod
    def test_suffix_replace_arg(mocker):
        def mock_model_init(self, arg1="arg1",):
            pass

        def mock_dataset_init(self, bbl="all"):
            pass

        mocker.patch("deep_morpho.models.BiMoNN.__init__", mock_model_init)
        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR10Dataset.__init__", mock_dataset_init)

        from deep_morpho.experiments.parser import MultiParser

        prs = MultiParser()
        prs["model"] = ["BiMoNN"]
        prs["dataset"] = ["cifar10dataset"]
        prs["arg1"] = ["value"]
        # prs["arg1.datamodule"] = "value"
        prs["bbl"] = ["value"]

        prs.parse_args([], add_argv=False)

        assert "arg1" in prs
        assert "bbl" in prs
        assert "arg1.net" not in prs
        assert "bbl.datamodule" not in prs
        assert prs["arg1"] == ["value"]
        assert prs["bbl"] == ["value"]

        assert "arg1.net" in prs.multi_args[0]
        assert "bbl.datamodule" in prs.multi_args[0]
        assert prs.multi_args[0]["arg1"] == "value"
        assert prs.multi_args[0]["bbl"] == "value"

    @staticmethod
    def test_unknown_args(mocker):
        testargs = ["file.py", ]
        mocker.patch("sys.argv", testargs)
        from deep_morpho.experiments.parser import MultiParser

        prs = MultiParser().parse_args(""
            '--model BiMoNN --dataset cifar10dataset --epoch 20 40'
            "".split(" ")
        )

        assert len(prs.multi_args) == 2
        assert prs.multi_args[0]["model"] == "bimonn"
        assert prs.multi_args[0]["dataset"] == "cifar10dataset"
        assert prs.multi_args[0]["epoch"] == "20"

        assert prs.multi_args[1]["model"] == "bimonn"
        assert prs.multi_args[1]["dataset"] == "cifar10dataset"
        assert prs.multi_args[1]["epoch"] == "40"


    @staticmethod
    def test_cli_over_dict(mocker):
        testargs = ["file.py", "--dataset", "mnistgrayscaledataset", "--model", "bimonnclassifier", "bimonn"]
        mocker.patch("sys.argv", testargs)
        from deep_morpho.experiments.parser import MultiParser

        prs = MultiParser()
        prs["model"] = "BiMoNN"
        prs["dataset"] = "cifar10dataset"
        prs.parse_args()

        # assert prs["model"] == "bimonnclassifier"
        # assert prs["dataset"] == "mnistgrayscaledataset"
        assert len(prs.multi_args) == 2
        assert prs.multi_args[0]["model"] == "bimonnclassifier"
        assert prs.multi_args[0]["dataset"] == "mnistgrayscaledataset"

        assert prs.multi_args[1]["model"] == "bimonn"
        assert prs.multi_args[1]["dataset"] == "mnistgrayscaledataset"


    @staticmethod
    def test_cli_over_dict2(mocker):
        testargs = ["file.py", "--dataset", "mnistgrayscaledataset", "--kernel_size", "adapt"]
        mocker.patch("sys.argv", testargs)
        from deep_morpho.experiments.parser import MultiParser

        prs = MultiParser()
        prs["model"] = ["BiMoNN"]
        prs["dataset"] = ["cifar10dataset"]
        prs.parse_args()

        # assert prs["model"] == "bimonnclassifier"
        # assert prs["dataset"] == "mnistgrayscaledataset"
        assert len(prs.multi_args) == 1
        assert prs.multi_args[0]["model"] == "bimonn"
        assert prs.multi_args[0]["dataset"] == "mnistgrayscaledataset"
        assert prs.multi_args[0]["kernel_size"] == "adapt"


    @staticmethod
    def test_multi_dict(mocker):
        from deep_morpho.experiments.parser import MultiParser

        prs = MultiParser()
        prs["model"] = ["BiMoNN", "bimonnclassifier"]
        prs["dataset"] = ["cifar10dataset"]
        prs.parse_args()

        assert len(prs.multi_args) == 2
        assert prs.multi_args[0]["model"] == "bimonn"
        assert prs.multi_args[0]["dataset"] == "cifar10dataset"

        assert prs.multi_args[1]["model"] == "bimonnclassifier"
        assert prs.multi_args[1]["dataset"] == "cifar10dataset"


    @staticmethod
    def test_given_args(mocker):
        testargs = ["file.py", "--dataset", "cifar10dataset", "cifar100dataset", "--model", "bimonnclassifier", "--bbl", "100", "--aau", "200"]
        mocker.patch("sys.argv", testargs)


        def mock_model_init(self, arg1="arg1",):
            pass

        def mock_dataset_init1(self, bbl="all"):
            pass

        def mock_dataset_init2(self, aau="all"):
            pass


        mocker.patch("deep_morpho.models.BiMoNN.__init__", mock_model_init)
        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR10Dataset.__init__", mock_dataset_init1)
        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR100Dataset.__init__", mock_dataset_init2)

        from deep_morpho.experiments.parser import MultiParser

        prs = MultiParser()
        prs.parse_args()

        assert prs.multi_args[0].given_args == set(["dataset", "model", "bbl.datamodule", "aau"])
        assert prs.multi_args[1].given_args == set(["dataset", "model", "aau.datamodule", "bbl"])

    @staticmethod
    def test_trainset_args(mocker):
        def mock_dataset_init(self, bbl="all"):
            pass

        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR10Dataset.__init__", mock_dataset_init)

        from deep_morpho.experiments.parser import MultiParser

        prs = MultiParser()
        prs["bbl.train"] = ["train"]
        prs["bbl.val"] = ["val"]
        prs["bbl"] = ["test"]
        prs["model"] = ["BiMoNN"]
        prs["dataset"] = ["cifar10dataset"]
        prs.parse_args([], add_argv=False)

        assert prs.multi_args[0].trainset_args()["bbl"] == "train"
        assert prs.multi_args[0].valset_args()["bbl"] == "val"
        assert prs.multi_args[0].testset_args()["bbl"] == "test"
    
    @staticmethod
    def test_multi_dataset_args(mocker):
        def mock_dataset_init(self, bbl="all"):
            pass

        def mock_dataset_init2(self, aau="all"):
            pass

        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR10Dataset.__init__", mock_dataset_init)
        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR100Dataset.__init__", mock_dataset_init2)

        from deep_morpho.experiments.parser import MultiParser

        prs = MultiParser()
        prs["bbl"] = ["train"]
        prs["aau"] = ["seee"]
        prs["model"] = ["BiMoNN"]
        prs["dataset"] = ["cifar10dataset", "cifar100dataset"]
        prs.parse_args([], add_argv=False)

        assert prs.multi_args[0].trainset_args() == {"bbl": "train"}
        assert "aau.datamodule" not in prs.multi_args[0].keys()
        assert "bbl.datamodule" in prs.multi_args[0].keys()

        assert prs.multi_args[1].trainset_args() == {"aau": "seee"}
        assert "bbl.datamodule" not in prs.multi_args[1].keys()
        assert "aau.datamodule" in prs.multi_args[1].keys()
    
    @staticmethod
    def test_multi_model_args(mocker):
        def mock_model_init(self, bbl="all"):
            pass

        def mock_model_init2(self, aau="all"):
            pass

        # mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR10Dataset.__init__", mock_dataset_init)
        # mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR100Dataset.__init__", mock_dataset_init2)
        mocker.patch("deep_morpho.models.BiMoNN.__init__", mock_model_init)
        mocker.patch("deep_morpho.models.BiMoNNClassifier.__init__", mock_model_init2)

        from deep_morpho.experiments.parser import MultiParser

        prs = MultiParser()
        prs["bbl"] = ["train"]
        prs["aau"] = ["seee"]
        prs["model"] = ["BiMoNN", "BiMoNNClassifier"]
        prs["dataset"] = ["cifar10dataset"]
        prs.parse_args([], add_argv=False)

        assert prs.multi_args[0].model_args()["bbl"] == "train"
        assert "aau" not in prs.multi_args[0].model_args().keys()
        assert "aau.net" not in prs.multi_args[0].keys()
        assert "bbl.net" in prs.multi_args[0].keys()

        assert prs.multi_args[1].model_args()["aau"] == "seee"
        assert "bbl" not in prs.multi_args[1].model_args().keys()
        assert "bbl.net" not in prs.multi_args[1].keys()
        assert "aau.net" in prs.multi_args[1].keys()

    @staticmethod
    def test_dict_copy():
        from deep_morpho.experiments.parser import MultiParser

        prs = MultiParser()
        prs["model"] = ["BiMoNN"]
        prs["dataset"] = ["cifar10dataset"]

        prs["banane"] = [{"a": 1, "b": 2}]
        prs["proxy"] = [1, 2]

        prs.parse_args([])

        # prs.multi_args[0]["banane"]["a"] = 5

        assert prs.multi_args[0]["banane"] is not prs.multi_args[1]["banane"]
        # assert prs.multi_args[0]["banane"]["a"] == 5
        # assert prs.multi_args[1]["banane"]["a"] == 3

