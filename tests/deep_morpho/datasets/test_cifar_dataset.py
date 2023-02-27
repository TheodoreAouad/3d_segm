from torchvision import transforms

from deep_morpho.experiments.parser import Parser
from deep_morpho.datasets.cifar_dataset import transform_default


class TestCifar10Dataset:

    @staticmethod
    def test_cifar10_transforms():
        args = Parser()
        args["dataset"] = "CIFAR10Classical"
        args["model"] = "BiMoNN"

        args["transform.train"] = transforms.Compose([
            transforms.RandomRotation(degrees=20),
            transform_default,
        ])

        args.parse_args([], add_argv=False)

        assert len(args["transform.train"].transforms) == 2
        assert args["transform"] == transform_default
