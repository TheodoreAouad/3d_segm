import torch
import numpy as np

from deep_morpho.models import BimonnDense, BimonnDenseNotBinary, BimonnBiselDenseNotBinary
from deep_morpho.initializer import InitBiseEnum
from deep_morpho.models.bise_base import ClosestSelemEnum, ClosestSelemDistanceEnum, BiseBiasOptimEnum, BiseWeightsOptimEnum


class TestBimonnDense:

    @staticmethod
    def test_forward():
        x = torch.rand(12, 50)
        model = BimonnDense(
            channels=[200, 200, ],
            input_size=x.shape[1],
            n_classes=14,
            threshold_mode={
                "weight": 'softplus',
                "activation": 'tanh',
            },
            initializer_method=InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
            initializer_args={"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": .5},
            input_mean=x.mean(),
            bias_optim_mode=BiseBiasOptimEnum.POSITIVE,
            bias_optim_args={"offset": 0},
            weights_optim_mode=BiseWeightsOptimEnum.THRESHOLDED,
            weights_optim_args={"constant_P": True, "factor": 1},
        )
        out = model(x)
        assert out.shape == (12, 14)

    @staticmethod
    def test_numel_binary():
        x = torch.rand(12, 50)
        model = BimonnDense(
            channels=[200, 200, ],
            input_size=x.shape[1],
            n_classes=14,
            threshold_mode={
                "weight": 'softplus',
                "activation": 'tanh',
            },
            initializer_method=InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
            initializer_args={"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": .5},
            first_init_args={"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": x.mean()},
            bias_optim_mode=BiseBiasOptimEnum.POSITIVE,
            bias_optim_args={"offset": 0},
            weights_optim_mode=BiseWeightsOptimEnum.THRESHOLDED,
            weights_optim_args={"constant_P": True, "factor": 1},
        )
        n_binary = model.numel_binary()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_binary == n_params


    @staticmethod
    def test_forward_binary():
        x = torch.randint(0, 2, (12, 50)).float()
        model = BimonnDense(
            channels=[200, 200, ],
            input_size=x.shape[1],
            n_classes=14,
            threshold_mode={
                "weight": 'softplus',
                "activation": 'tanh',
            },
            initializer_method=InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
            initializer_args={"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": .5},
            first_init_args={"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": x.mean()},
            bias_optim_mode=BiseBiasOptimEnum.POSITIVE,
            bias_optim_args={"offset": 0},
            weights_optim_mode=BiseWeightsOptimEnum.THRESHOLDED,
            weights_optim_args={"constant_P": True, "factor": 1},
        )
        model.binary(update_binaries=True)
        out = model(x).detach().cpu().numpy()
        assert np.isin(out, [0, 1]).all()


class TestBimonnDenseNotBinary:

    @staticmethod
    def test_forward():
        x = torch.rand(12, 50)
        model = BimonnDenseNotBinary(
            channels=[200, 200, ],
            input_size=x.shape[1],
            n_classes=14,
            threshold_mode={
                "weight": 'softplus',
                "activation": 'tanh',
            },
            initializer_method=InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
            initializer_args={"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": .5},
            input_mean=x.mean(),
            bias_optim_mode=BiseBiasOptimEnum.POSITIVE,
            bias_optim_args={"offset": 0},
            weights_optim_mode=BiseWeightsOptimEnum.THRESHOLDED,
            weights_optim_args={"constant_P": True, "factor": 1},
        )
        out = model(x)
        assert out.shape == (12, 14)

    @staticmethod
    def test_numel_binary():
        x = torch.rand(12, 50)
        model = BimonnDenseNotBinary(
            channels=[200, 200, ],
            input_size=x.shape[1],
            n_classes=14,
            threshold_mode={
                "weight": 'softplus',
                "activation": 'tanh',
            },
            initializer_method=InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
            initializer_args={"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": .5},
            first_init_args={"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": x.mean()},
            bias_optim_mode=BiseBiasOptimEnum.POSITIVE,
            bias_optim_args={"offset": 0},
            weights_optim_mode=BiseWeightsOptimEnum.THRESHOLDED,
            weights_optim_args={"constant_P": True, "factor": 1},
        )
        n_binary = model.numel_binary()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_params_not = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad if "classification" in name)
        assert n_binary + n_params_not == n_params


    @staticmethod
    def test_forward_binary():
        x = torch.randint(0, 2, (12, 50)).float()
        model = BimonnDenseNotBinary(
            channels=[200, 200, ],
            input_size=x.shape[1],
            n_classes=14,
            threshold_mode={
                "weight": 'softplus',
                "activation": 'tanh',
            },
            initializer_method=InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS,
            initializer_args={"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": .5},
            first_init_args={"ub": 1e-2, "max_output_value": 0.95, "p_for_init": "auto", "input_mean": x.mean()},
            bias_optim_mode=BiseBiasOptimEnum.POSITIVE,
            bias_optim_args={"offset": 0},
            weights_optim_mode=BiseWeightsOptimEnum.THRESHOLDED,
            weights_optim_args={"constant_P": True, "factor": 1},
        )
        model.binary(update_binaries=True)
        out = model(x).detach().cpu().numpy()
        # assert np.isin(out, [0, 1]).all()


class TestBimonnBiselDenseNotBinary:

    @staticmethod
    def test_forward():
        x = torch.rand(12, 1, 28, 28)
        model = BimonnBiselDenseNotBinary(
            kernel_size=(5, 5),
            channels=[50, 50, 20],
            input_size=x.shape[1:],
            n_classes=14,
            threshold_mode={
                "weight": 'softplus',
                "activation": 'tanh',
            },
            input_mean=x.mean(),
            bias_optim_mode=BiseBiasOptimEnum.POSITIVE,
            bias_optim_args={"offset": 0},
            weights_optim_mode=BiseWeightsOptimEnum.THRESHOLDED,
            weights_optim_args={"constant_P": True, "factor": 1},
        )
        out = model(x)
        assert out.shape == (12, 14)

    @staticmethod
    def test_numel_binary():
        x = torch.rand(12, 1, 28, 28)
        model = BimonnBiselDenseNotBinary(
            kernel_size=(5, 5),
            channels=[50, 50, ],
            input_size=x.shape[1:],
            n_classes=14,
            threshold_mode={
                "weight": 'softplus',
                "activation": 'tanh',
            },
            input_mean=x.mean(),
            bias_optim_mode=BiseBiasOptimEnum.POSITIVE,
            bias_optim_args={"offset": 0},
            weights_optim_mode=BiseWeightsOptimEnum.THRESHOLDED,
            weights_optim_args={"constant_P": True, "factor": 1},
        )
        n_binary = model.numel_binary()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_params_not = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad if "classification" in name)
        assert n_binary + n_params_not == n_params


    @staticmethod
    def test_forward_binary():
        x = torch.randint(0, 2, (12, 1, 28, 28)).float()
        model = BimonnBiselDenseNotBinary(
            kernel_size=(5, 5),
            channels=[50, 50, ],
            input_size=x.shape[1:],
            n_classes=14,
            threshold_mode={
                "weight": 'softplus',
                "activation": 'tanh',
            },
            input_mean=x.mean(),
            bias_optim_mode=BiseBiasOptimEnum.POSITIVE,
            bias_optim_args={"offset": 0},
            weights_optim_mode=BiseWeightsOptimEnum.THRESHOLDED,
            weights_optim_args={"constant_P": True, "factor": 1},
        )
        model.binary(update_binaries=True)
        out = model(x).detach().cpu().numpy()
        # assert np.isin(out, [0, 1]).all()
