import torch

from deep_morpho.models import BiMoNN, BiMoNNClassifierMaxPool, BiMoNNClassifierMaxPoolNotBinary
from deep_morpho.datasets import MnistClassifDataset, CIFAR10Dataset
from deep_morpho.initializer import InitBiseEnum
from deep_morpho.initializer import InitBimonnEnum, InitBiseEnum
from deep_morpho.models.bise_base import ClosestSelemEnum, ClosestSelemDistanceEnum, BiseBiasOptimEnum, BiseWeightsOptimEnum


class TestBimonn:

    @staticmethod
    def test_bisel_init():
        model = BiMoNN(kernel_size=(7, 7), channels=[3, 5, 3], atomic_element="bisel")
        model

    @staticmethod
    def test_sybisel_init():
        model = BiMoNN(kernel_size=(7, 7), channels=[3, 5, 3], atomic_element="sybisel")
        model

    @staticmethod
    def test_bisel_multi_kernel():
        model = BiMoNN(kernel_size=[(7, 7), (3, 3)], channels=[3, 5, 3], atomic_element="bisel")
        assert model.kernel_size == [(7, 7), (3, 3)]

    @staticmethod
    def test_bisel_one_kernel():
        model = BiMoNN(kernel_size=(7, 3), channels=[3, 5, 3], atomic_element="bisel")
        assert model.kernel_size == [(7, 3), (7, 3)]

    @staticmethod
    def test_bisel_input_mean():
        model = BiMoNN(kernel_size=(7, 7), channels=[3, 5, 3, 3], atomic_element="bisel",
            initializer_args={"input_mean": 0.3, "bise_init_args": {"init_bias_value": 1}, "bise_init_method": InitBiseEnum.CUSTOM_HEURISTIC})
        assert [k.bise_initializer.input_mean for k in model.bisel_initializers] == [0.3, 0.5, 0.5]
        assert [k.lui_initializer.input_mean for k in model.bisel_initializers] == [0.5, 0.5, 0.5]

    @staticmethod
    def test_sybisel_input_mean():
        model = BiMoNN(kernel_size=(7, 7), channels=[3, 5, 3, 3], atomic_element="sybisel",
            initializer_args={"input_mean": 0.3, "bise_init_args": {"mean_weight": "auto"}, "bise_init_method": InitBiseEnum.CUSTOM_CONSTANT})
        assert [k.bise_initializer.input_mean for k in model.bisel_initializers] == [0.3, 0, 0]
        assert [k.lui_initializer.input_mean for k in model.bisel_initializers] == [0, 0, 0]

    # @staticmethod
    # def test_mix_bisel_input_mean():
    #     model = BiMoNN(kernel_size=(7, 7), channels=[3, 5, 3, 3], atomic_element=["bisel", "sybisel", "bisel"], input_mean=0.3)
    #     assert model.input_mean == [0.3, 0.5, 0]


class TestBimonnClassifierMaxPool():

    @staticmethod
    def test_init():
        model = BiMoNNClassifierMaxPool(
            kernel_size=(7, 7),
            channels=[3, 5, 3],
            atomic_element='bisel',
            input_size=(50, 50),
            n_classes=10,
        )
        model


    @staticmethod
    def test_forward():
        x = torch.ones((3, 1, 50, 50))
        n_classes = 10

        model = BiMoNNClassifierMaxPool(
            kernel_size=(7, 7),
            channels=[x.shape[1], 5, 3, 3],
            atomic_element='bisel',
            input_size=x.shape[-2:],
            n_classes=n_classes,
        )

        otp = model(x)
        assert otp.shape == (x.shape[0], n_classes)

    @staticmethod
    def test_forward_shallow():
        x = torch.ones((3, 1, 50, 50))
        n_classes = 10

        model = BiMoNNClassifierMaxPool(
            kernel_size=(7, 7),
            channels=[x.shape[1], 5],
            atomic_element='bisel',
            input_size=x.shape[-2:],
            n_classes=n_classes,
        )

        otp = model(x)
        assert otp.shape == (x.shape[0], n_classes)

    @staticmethod
    def test_forward_odd():
        x = torch.ones((3, 1, 51, 51))
        n_classes = 10

        model = BiMoNNClassifierMaxPool(
            kernel_size=(7, 7),
            channels=[x.shape[1], 5, 3, 3],
            atomic_element='bisel',
            input_size=x.shape[-2:],
            n_classes=n_classes,
        )

        otp = model(x)
        assert otp.shape == (x.shape[0], n_classes)

    @staticmethod
    def test_numel_binary():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.ones((3, 1, 51, 51)).to(device)
        n_classes = 10

        model = BiMoNNClassifierMaxPool(
            kernel_size=(7, 7),
            channels=[x.shape[1], 200, 200],
            atomic_element='bisel',
            input_size=x.shape[-2:],
            n_classes=n_classes,
        ).to(device)

        otp = model(x)
        assert not otp.isnan().any()

        n_binary = model.numel_binary()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_binary == n_params

    @staticmethod
    def test_numel_binary_not_binary():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.ones((3, 1, 51, 51)).to(device)
        n_classes = 10

        model = BiMoNNClassifierMaxPoolNotBinary(
            kernel_size=(7, 7),
            channels=[x.shape[1], 200, 200],
            atomic_element='bisel',
            input_size=x.shape[-2:],
            n_classes=n_classes,
            threshold_mode={
                "weight": 'softplus',
                "activation": 'tanh',
            },
        ).to(device)

        otp = model(x)
        assert not otp.isnan().any()
        n_binary = model.numel_binary()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        n_lui_params = sum(p.numel() for name, p in model.classification_layer.named_parameters() if "lui" in name and p.requires_grad)

        assert n_binary + n_lui_params == n_params


    @staticmethod
    def test_numel_binary_not_binary_mnist():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        trainloader, valloader, testloader = MnistClassifDataset.get_train_val_test_loader(
            n_inputs_train=500,
            n_inputs_val=1,
            n_inputs_test=1,
            batch_size=32,
            preprocessing=None,
            num_workers=0,
            do_symetric_output=False,
            **{"threshold": 30, "size": (50, 50), "invert_input_proba": 0, },
        )
        x = next(iter(trainloader))[0]

        n_classes = 10

        model = BiMoNNClassifierMaxPoolNotBinary(
            kernel_size=(5, 5),
            channels=[x.shape[1], 200, 200],
            atomic_element='bisel',
            input_size=x.shape[-2:],
            n_classes=n_classes,
            threshold_mode={
                "weight": 'softplus',
                "activation": 'tanh',
            },
            initializer_method=InitBimonnEnum.INPUT_MEAN,
            initializer_args={
                "bise_init_method": InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
                "lui_init_method": InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS_RANDOM_BIAS,
                "bise_init_args": {"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto"},
                "input_mean": x.mean(),
            },
            bias_optim_mode=BiseBiasOptimEnum.POSITIVE,
            bias_optim_args={"offset": 0},
            weights_optim_mode=BiseWeightsOptimEnum.THRESHOLDED,
            weights_optim_args={"constant_P": True, "factor": 1},
        )

        otp = model(x)
        assert not otp.isnan().any()
        n_binary = model.numel_binary()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        n_lui_params = sum(p.numel() for name, p in model.classification_layer.named_parameters() if "lui" in name and p.requires_grad)

        assert n_binary + n_lui_params == n_params

    @staticmethod
    def test_forward_classif_channel():
        x = next(iter(CIFAR10Dataset.get_loader(batch_size=1, train=True, levelset_handler_args={"n_values": 10})))[0]
        n_classes = 10

        model = BiMoNNClassifierMaxPoolNotBinary(
            kernel_size=(5, 5),
            channels=[10*3, 5, ],
            atomic_element='bisel',
            input_size=x.shape[-2:],
            n_classes=n_classes,
        )

        otp = model(x)
        assert otp.shape == (x.shape[0], n_classes)
