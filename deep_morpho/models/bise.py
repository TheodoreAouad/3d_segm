from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


from .threshold_layer import dispatcher
from .complementation_layer import ComplementationLayer


class BiSE(nn.Module):

    def __init__(
        self,
        kernel_size: Tuple,
        weight_P: float = 1,
        threshold_mode: str = "sigmoid",
        activation_P: float = 10,
        constant_activation_P: bool = False,
        constant_weight_P: bool = False,
        shared_weights: torch.tensor = None,
        shared_weight_P: torch.tensor = None,
        init_bias_value: float = -2,
        init_weight_identity: bool = True,
        out_channels=1,
        *args,
        **kwargs
    ):
        super().__init__()
        self.threshold_mode = self._init_threshold_mode(threshold_mode)
        self._weight_P = nn.Parameter(torch.tensor([weight_P]).float())
        self.activation_P_init = activation_P
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size[0]//2,
            padding_mode='replicate',
            *args,
            **kwargs
        )
        with torch.no_grad():
            self.conv.bias.fill_(init_bias_value)
        if init_weight_identity:
            self._init_as_identity()

        self.shared_weights = shared_weights
        self.shared_weight_P = shared_weight_P

        self.weight_threshold_layer = dispatcher[self.weight_threshold_mode](P_=self.weight_P, constant_P=constant_weight_P)
        self.activation_threshold_layer = dispatcher[self.activation_threshold_mode](P_=activation_P, constant_P=constant_activation_P)

    def activation_threshold_fn(self, x):
        return self.activation_threshold_layer.threshold_fn(x)

    def weight_threshold_fn(self, x):
        return self.weight_threshold_layer.threshold_fn(x)

    def forward(self, x: Tensor) -> Tensor:
        output = self.conv._conv_forward(x, self._normalized_weight, self.bias, )
        output = self.activation_threshold_layer(output)
        return output

    def _init_as_identity(self):
        self.conv.weight.data.fill_(-1)
        shape = self.conv.weight.shape
        self.conv.weight.data[..., shape[-2]//2, shape[-1]//2] = 1

    def is_erosion_by(self, S: np.ndarray, v1: float = 0, v2: float = 1):
        assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
        lb, ub = self.bias_bounds_erosion(S=S, v1=v1, v2=v2)
        return lb < -self.bias < ub

    def is_dilation_by(self, S: np.ndarray, v1: float = 0, v2: float = 1):
        assert np.isin(np.unique(S), [0, 1]).all(), "S must be binary matrix"
        lb, ub = self.bias_bounds_dilation(S=S, v1=v1, v2=v2)
        return lb < -self.bias < ub

    def bias_bounds_erosion(self, S: np.ndarray, v1: float = 0, v2: float = 1):
        S = S.astype(bool)
        W = self._normalized_weight.squeeze().cpu().detach().numpy()
        return W.sum() - (1 - v1) * W[S].min(), v2 * W[S].sum()

    def bias_bounds_dilation(self, S: np.ndarray, v1: float = 0, v2: float = 1):
        S = S.astype(bool)
        W = self._normalized_weight.squeeze().cpu().detach().numpy()
        return W[~S].sum() + v1 * W[S].sum(), v2 * W[S].min()

    def find_selem_for_operation(self, operation: str, v1: float = 0, v2: float = 1):
        """
        We find the selem for either a dilation or an erosion. In theory, there is at most one selem that works.
        We verify this theory.

        Args:
            operation (str): 'dilation' or 'erosion', the operation we want to check for
            v1 (float): the lower value of the almost binary
            v2 (float): the upper value of the almost binary (input not in ]v1, v2[)

        Returns:
            np.ndarray if a selem is found
            None if none is found
        """
        weight_values = self._normalized_weight.unique()
        res = []
        is_op_fn = {'dilation': self.is_dilation_by, 'erosion': self.is_erosion_by}[operation]
        for value in weight_values:
            S = (self._normalized_weight >= value).squeeze().cpu().detach().numpy()
            if is_op_fn(S=S, v1=v1, v2=v2):
                res.append(S)

        assert len(res) <= 1, "There should only be at most one working selem. Check for errors."
        if len(res) == 1:
            return res[0]
        return None

    def find_selem_dilation(self, v1: float = 0, v2: float = 1):
        return self.find_selem_for_operation('dilation', v1=v1, v2=v2)

    def find_selem_erosion(self, v1: float = 0, v2: float = 1):
        return self.find_selem_for_operation('erosion', v1=v1, v2=v2)

    def find_selem_and_operation(self, v1: float = 0, v2: float = 1):
        """Find the selem and the operation given the almost binary features.

        Args:
            v1 (float): lower bound of almost binary input deadzone. Defaults to 0.
            v2 (float): upper bound of almost binary input deadzone. Defaults to 1.

        Returns:
            (np.ndarray, operation): if the selem is found, returns the selem and the operation
            (None, None): if nothing is found, returns None
        """
        for operation in ['dilation', 'erosion']:
            selem = self.find_selem_for_operation(operation, v1=v1, v2=v2)
            if selem is not None:
                return selem, operation
        return None, None

    def get_outputs_bounds(self, v1: float = 0, v2: float = 1):
        """If the BiSE is learned, returns the bounds of the deadzone of the almost binary output.

        Args:
            v1 (float): lower bound of almost binary input deadzone. Defaults to 0.
            v2 (float): upper bound of almost binary input deadzone. Defaults to 1.

        Returns:
            (float, float): if the bise is learned, bounds of the deadzone of the almost binary output
            (None, None): if the bise is not learned
        """
        selem, operation = self.find_selem_and_operation(v1=v1, v2=v2)

        if selem is None:
            return None, None

        if operation == 'dilation':
            b1, b2 = self.bias_bounds_dilation(selem, v1=v1, v2=v2)
        if operation == 'erosion':
            b1, b2 = self.bias_bounds_erosion(selem, v1=v1, v2=v2)

        with torch.no_grad():
            res = [self.activation_threshold_layer(b1 + self.bias), self.activation_threshold_layer(b2 + self.bias)]
        res = [i.item() for i in res]
        return res
        # return 0, 1

    @staticmethod
    def _init_threshold_mode(threshold_mode):
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["weight", "activation"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode

    @property
    def weight_threshold_mode(self):
        return self.threshold_mode["weight"]

    @property
    def activation_threshold_mode(self):
        return self.threshold_mode["activation"]

    @property
    def _normalized_weight(self):
        conv_weight = self.weight_threshold_layer(self.weight)
        return conv_weight

    @property
    def weight(self):
        if self.shared_weights is not None:
            return self.shared_weights
        return self.conv.weight

    @property
    def weight_P(self):
        if self.shared_weight_P is not None:
            return self.shared_weight_P
        return self._weight_P

    @property
    def activation_P(self):
        return self.activation_threshold_layer.P_

    @property
    def weights(self):
        return self.weight

    @property
    def bias(self):
        return self.conv.bias


class BiSEC(BiSE):

    def __init__(self, *args, alpha_init=0, invert_thresholded_alpha=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv.bias = None
        self.complementation_layer = ComplementationLayer(
            self.complementation_threshold_mode, alpha_init=alpha_init, invert_thresholded_alpha=invert_thresholded_alpha
        )

        self._bias = nn.Parameter(torch.tensor([0.5]).float(), requires_grad=False)


    def forward(self, x: Tensor) -> Tensor:
        output = self.complementation_layer(x)
        output = super().forward(output)
        if self.thresholded_alpha < 1/2:
            return 1 - output
        # output = self.complementation_layer(output)
        return output

    @property
    def bias(self):
        return -(
            min(self.thresholded_alpha, 1 - self.thresholded_alpha) *
            (self._normalized_weight.sum() - 1) + self._bias
        )

    @property
    def alpha(self):
        return self.complementation_layer.alpha

    @property
    def thresholded_alpha(self):
        return self.complementation_layer.thresholded_alpha

    @property
    def complementation_threshold_mode(self):
        return self.threshold_mode["complementation"]

    @staticmethod
    def _init_threshold_mode(threshold_mode):
        if isinstance(threshold_mode, str):
            threshold_mode = {k: threshold_mode.lower() for k in ["weight", "activation", "complementation"]}
        elif not isinstance(threshold_mode, dict):
            raise NotImplementedError(f"threshold_mode type {type(threshold_mode)} not supported.")
        return threshold_mode
