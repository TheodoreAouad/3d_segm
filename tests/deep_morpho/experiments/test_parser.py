def mock_init(self, arg1="arg1", arg2="banane", kernel_size=[1, 2]):
    pass


class TestParser:
    @staticmethod
    def test_parser(mocker):
        mocker.patch("deep_morpho.models.BiMoNN.__init__", mock_init)
        from deep_morpho.experiments.parser import Parser

        prs = Parser()
        prs["dataset"] = "cifar10dataset"
        prs["model"] = "BiMoNN"
        prs["kernel_size"] = [3, 3, 3]


        prs.parse_args("--atomic_element bisel".split())

        assert prs["dataset"] == "cifar10dataset"
        assert prs["model"] == "BiMoNN"

        assert prs["kernel_size"] == [3, 3, 3]
        assert prs["atomic_element"] == "bisel"
        assert prs["arg1"] == "arg1"
        assert prs["arg2"] == "banane"
