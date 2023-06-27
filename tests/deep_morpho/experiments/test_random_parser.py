from mock import patch
import pytest


class TestRandomParser:

    @staticmethod
    def test_cli_args(mocker):
        testargs = ["file.py", "--dataset", "cifar10dataset", "--model", "BiMoNN"]
        mocker.patch("sys.argv", testargs)
        from deep_morpho.experiments.random_parser import RandomParser

        prs = RandomParser()
        prs.parse_args()
        prs = prs.get_args()
        assert prs["model"] == "BiMoNN"
        assert prs["dataset"] == "cifar10dataset"

    @staticmethod
    def test_suffix_replace_arg(mocker):
        def mock_model_init(self, arg1="arg1",):
            pass

        def mock_dataset_init(self, bbl="all"):
            pass

        mocker.patch("deep_morpho.models.BiMoNN.__init__", mock_model_init)
        mocker.patch("deep_morpho.datasets.cifar_dataset.CIFAR10Dataset.__init__", mock_dataset_init)

        from deep_morpho.experiments.random_parser import RandomParser

        prs = RandomParser()
        prs["model"] = ["BiMoNN"]
        prs["dataset"] = ["cifar10dataset"]
        prs["arg1"] = ["value"]
        # prs["arg1.datamodule"] = "value"
        prs["bbl"] = ["value2"]

        # prs.parse_args([], add_argv=False)
        prs = prs.get_args()

        assert "arg1" not in prs
        assert "bbl" not in prs
        assert "arg1.net" in prs
        assert "bbl.datamodule" in prs
        assert prs["arg1"] == "value"
        assert prs["bbl"] == "value2"


    @staticmethod
    def test_unknown_args(mocker):
        testargs = ["file.py", "--dataset", "cifar10dataset", "--model", "BiMoNN", "--epoch", "20"]
        mocker.patch("sys.argv", testargs)
        from deep_morpho.experiments.random_parser import RandomParser

        prs = RandomParser()
        prs.parse_args()
        prs = prs.get_args()

        assert prs["model"] == "BiMoNN"
        assert prs["dataset"] == "cifar10dataset"
        assert prs["epoch"] == "20"
