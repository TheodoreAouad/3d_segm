from enum import Enum
from typing import Tuple

import numpy as np
import torch.nn as nn
import torch

from general.utils import uniform_sampling_bound


class InitBiseEnum(Enum):
    NORMAL = 1
    KAIMING_UNIFORM = 2
    CUSTOM_HEURISTIC = 3
    CUSTOM_CONSTANT = 4
    IDENTITY = 5
    CUSTOM_CONSTANT_RANDOM_BIAS = 6
    CUSTOM_HEURISTIC_RANDOM_BIAS = 7


class BiseInitializer:

    def initialize(self, module: nn.Module):
        return module.weight, module.bias


class InitWeightsThenBias(BiseInitializer):
    def initialize(self, module: nn.Module):
        self.init_weights(module)
        self.init_bias(module)
        return module.weight, module.bias

    def init_weights(self, module):
        pass

    def init_bias(self, module):
        pass


class InitBiasFixed(InitWeightsThenBias):
    def __init__(self, init_bias_value: float, *args, **kwargs) -> None:
        self.init_bias_value = init_bias_value

    def init_bias(self, module):
        module.set_bias(
            torch.zeros_like(module.bias) - self.init_bias_value
        )


class InitNormalIdentity(InitBiasFixed):

    def __init__(self, init_bias_value: float, mean: float = 1, std: float = .3, *args, **kwargs) -> None:
        super().__init__(init_bias_value=init_bias_value)
        self.mean = mean
        self.std = std

    @staticmethod
    def _init_normal_identity(kernel_size, chan_output, std=0.3, mean=1) -> torch.Tensor:
        weights = torch.randn((chan_output,) + kernel_size)[:, None, ...] * std - mean
        weights[..., kernel_size[0] // 2, kernel_size[1] // 2] += 2*mean
        return weights


    def init_weights(self, module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        module.set_normalized_weights(self._init_normal_identity(module.kernel_size, module.out_channels))


class InitIdentity(InitBiasFixed):
    def init_weights(self, module: nn.Module):
        module.conv.weight.data.fill_(-1)
        shape = module.conv.weight.shape
        module.conv.weight.data[..., shape[-2]//2, shape[-1]//2] = 1


class InitKaimingUniform(InitBiasFixed):
    def init_weights(self, module: nn.Module):
        module.set_normalized_weights(module.conv.weight + 1)


class InitSybiseBias(InitWeightsThenBias):
    def __init__(self, init_class: BiseInitializer, input_mean: float = 0, *args, **kwargs):
        self.input_mean = input_mean
        self.init_class = init_class(*args, **kwargs)

    def get_bias_from_weights(self, module):
        self.init_bias_value = self.input_mean * module.weight.mean() * torch.tensor(module._normalized_weights.shape[1:]).prod()

    def init_weights(self, module):
        return self.init_class(module)

    def init_bias(self, module):
        return module.set_bias(
            torch.zeros_like(module.bias) - self.get_bias_from_weights(module)
        )


class InitIdentitySybise(InitSybiseBias):
    def __init__(self, *args, **kwargs):
        super().__init__(init_class=InitIdentity, *args, **kwargs)


class InitNormalIdentitySybise(InitSybiseBias):
    def __init__(self, *args, **kwargs):
        super().__init__(init_class=InitNormalIdentity, *args, **kwargs)


class InitKaimingUniformSybise(InitSybiseBias):
    def __init__(self, *args, **kwargs):
        super().__init__(init_class=InitKaimingUniform, *args, **kwargs)


class InitBiseHeuristicWeights(InitBiasFixed):
    def __init__(self, init_bias_value: float, input_mean: float = 0.5, *args, **kwargs):
        super().__init__(init_bias_value=init_bias_value)
        self.input_mean = input_mean

    def init_weights(self, module):
        nb_params = torch.tensor(module._normalized_weights.shape[1:]).prod()
        mean = self.init_bias_value / (self.input_mean * nb_params)
        std = .5
        lb = mean * (1 - std)
        ub = mean * (1 + std)
        module.set_normalized_weights(
            torch.rand_like(module.weights) * (lb - ub) + ub
        )


class InitBiseConstantVarianceWeights(InitBiasFixed):
    def __init__(self, init_bias_value: float = "auto", input_mean: float = 0.5, *args, **kwargs):
        super().__init__(init_bias_value=init_bias_value)
        self.input_mean = input_mean

    def init_weights(self, module):
        p = 1
        nb_params = torch.tensor(module._normalized_weights.shape[1:]).prod()

        if self.init_bias_value == "auto":
            # To keep track with previous experiment
            # if self.input_mean > 0.7:
            #     lb1 = 1/p * torch.sqrt(6*nb_params / (12 + 1/self.input_mean**2))
            # else:
            #     lb1 = 1/(2*p) * torch.sqrt(3/2 * nb_params)
            lb1 = 1/p * torch.sqrt(6*nb_params / (12 + 1/self.input_mean**2))
            lb2 = 1 / p * torch.sqrt(nb_params / 2)
            self.init_bias_value = (lb1 + lb2) / 2

        mean = self.init_bias_value / (self.input_mean * nb_params)
        sigma = (2 * nb_params - 4 * self.init_bias_value**2 * p ** 2) / (p ** 2 * nb_params ** 2)
        diff = torch.sqrt(3 * sigma)
        lb = mean - diff
        ub = mean + diff

        new_weights = torch.rand_like(module.weights) * (lb - ub) + ub

        module.set_normalized_weights(
            new_weights
        )


class InitSybiseHeuristicWeights(InitWeightsThenBias):
    def __init__(self, input_mean: float = 0, mean_weight: float = "auto", init_bias_value: float = 0, *args, **kwargs) -> None:
        self.input_mean = input_mean
        self.mean_weight = mean_weight
        self.init_bias_value = init_bias_value

        self.mean = None
        self.nb_params = None

    def init_weights(self, module):
        nb_params = torch.tensor(module._normalized_weights.shape[1:]).prod()
        if self.mean_weight == 'auto':
            mean = .5
        else:
            mean = self.mean_weight
        # std = mean / (2 * np.sqrt(3))
        # lb = mean * (1 - std)
        # ub = mean * (1 + std)
        lb = mean / 2
        ub = 3 * lb
        module.set_normalized_weights(
            torch.rand_like(module.weights) * (lb - ub) + ub
        )

        self.mean = mean
        self.nb_params = nb_params

    def init_bias(self, module):
        new_value = self.input_mean * self.mean * self.nb_params
        module.set_bias(
            torch.zeros_like(module.bias) - new_value + self.init_bias_value
        )


class InitSybiseConstantVarianceWeights(InitWeightsThenBias):
    def __init__(self, input_mean: float = 0, mean_weight: float = "auto", init_bias_value: float = 0, *args, **kwargs) -> None:
        self.input_mean = input_mean
        self.mean_weight = mean_weight
        self.init_bias_value = init_bias_value

        self.mean = None
        self.nb_params = None

    def init_weights(self, module):
        p = 1
        nb_params = torch.tensor(module._normalized_weights.shape[1:]).prod()

        if self.mean_weight == "auto":
            ub = 1 / (p * torch.sqrt(nb_params))
            lb = np.sqrt(3 / 4) * ub
            mean = (lb + ub) / 2

        sigma = 1 / (p ** 2 * nb_params) - mean ** 2
        diff = torch.sqrt(3 * sigma)
        lb = mean - diff
        ub = mean + diff
        module.set_normalized_weights(
            torch.rand_like(module.weights) * (lb - ub) + ub
        )

        self.mean = mean
        self.nb_params = nb_params

    def init_bias(self, module):
        new_value = self.input_mean * self.mean * self.nb_params
        module.set_bias(
            torch.zeros_like(module.bias) - new_value + self.init_bias_value
        )


class InitSybiseConstantVarianceWeightsRandomBias(InitSybiseConstantVarianceWeights):
    def __init__(self, ub: float = 0.01, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ub = ub

    def init_bias(self, module):
        new_value = self.input_mean * self.mean * self.nb_params + uniform_sampling_bound(-self.ub, self.ub).astype(np.float32)
        module.set_bias(
            torch.zeros_like(module.bias) - new_value + self.init_bias_value
        )


class InitSybiseHeuristicWeightsRandomBias(InitSybiseHeuristicWeights):
    def __init__(self, ub: float = 0.01, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ub = ub

    def init_bias(self, module):
        new_value = self.input_mean * self.mean * self.nb_params + uniform_sampling_bound(-self.ub, self.ub).astype(np.float32)
        module.set_bias(
            torch.zeros_like(module.bias) - new_value + self.init_bias_value
        )
