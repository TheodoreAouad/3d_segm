import torch
import torch.nn as nn
import numpy as np

from .threshold_layer import dispatcher


class LUI(nn.Module):

    def __init__(self, threshold_mode: str, chan_inputs: int, chan_outputs: int, P_: float = 1, constant_P: bool = False,):
        super().__init__()

        self.threshold_mode = threshold_mode
        self.constant_P = constant_P
        self.chan_inputs = chan_inputs
        self.chan_outputs = chan_outputs

        self.threshold_layer = dispatcher[self.threshold_mode](P_=P_, constant_P=self.constant_P, n_channels=chan_outputs)
        self.linear = nn.Linear(chan_inputs, chan_outputs)

        with torch.no_grad():
            self.init_coefs()
            self.init_bias()

    def forward(self, x):
        return self.threshold_layer(self.linear(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)

    @property
    def bias(self):
        return self.linear.bias

    @property
    def weight(self):
        return self.linear.weight

    @property
    def weights(self):
        return self.weight

    def init_coefs(self):
        self.linear.weight.fill_(1)

    def init_bias(self):
        self.linear.bias.fill_(-.5)

    @property
    def activation_P(self):
        return self.threshold_layer.P_

    @staticmethod
    def bias_bounds_intersection(betas: torch.Tensor, C: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> bool:
        LUI.check_dims_bounds_params(betas, C, v1, v2)
        C = C.astype(bool)
        betas = betas.squeeze().cpu().detach().numpy()
        return betas.sum() - ((1 - v1) * betas)[C].min(), (betas * v2)[C].sum()

    @staticmethod
    def bias_bounds_union(betas: torch.Tensor, C: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> bool:
        LUI.check_dims_bounds_params(betas, C, v1, v2)
        C = C.astype(bool)
        betas = betas.squeeze().cpu().detach().numpy()
        return (betas * v1)[C].sum() + betas[~C].sum(), (betas * v2)[C].min()

    @staticmethod
    def init_v1_v2(betas: np.ndarray, v1: np.ndarray, v2: np.ndarray):
        if v1 is None:
            v1 = np.zeros(betas.shape[0])
        if v2 is None:
            v2 = np.ones(betas.shape[0])
        return v1, v2

    @staticmethod
    def is_union_by(betas: torch.Tensor, bias: torch.Tensor, C: np.ndarray, v1: np.ndarray = None, v2: np.ndarray = None):
        assert np.isin(np.unique(C), [0, 1]).all(), "C must be binary matrix"
        v1, v2 = LUI.init_v1_v2(betas, v1, v2)
        lb, ub = LUI.bias_bounds_union(betas, C, v1, v2)
        return lb < - bias < ub

    @staticmethod
    def is_intersection_by(betas: torch.Tensor, bias: torch.Tensor, C: np.ndarray, v1: np.ndarray = None, v2: np.ndarray = None):
        assert np.isin(np.unique(C), [0, 1]).all(), "C must be binary matrix"
        v1, v2 = LUI.init_v1_v2(betas, v1, v2)
        lb, ub = LUI.bias_bounds_intersection(betas, C, v1, v2)
        return lb < - bias < ub

    @staticmethod
    def check_dims_bounds_params(betas: torch.Tensor, C: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> None:
        assert len(C) == len(betas), f"The set C must have be of shape (len(betas)), but len(betas)={len(betas)} and len(C) = {len(C)}"
        assert len(v1) == len(v2), f"The almost binary bounds must be of shame shape. len(v1) = {len(v1)} , len(v2) = {len(v2)}"
        assert len(betas) == len(v1), f"You must give one almost binary bound per beta. len(betas) = {len(betas)} , len(v1) = {len(v1)}"

    def find_set_for_operation_chan(self, idx: int, operation: str, v1: np.ndarray = None, v2: np.ndarray = None):
        """
        We find the set for either an intersection or a union. In theory, there is at most one set that works.
        We verify this theory.

        Args:
            operation (str): 'union' or 'intersection', the operation we want to check for
            v1 (np.ndarray): the lower values of the almost binary. Size must be self.chan_inputs.
            v2 (np.ndarray): the upper values of the almost binary. Size must be self.chan_inputs. (input not in ]v1, v2[)

        Returns:
            np.ndarray if a set is found
            None if none is found
        """
        coefs = self.weight[idx]
        coefs_values = coefs.unique()
        bias = self.bias[idx]

        is_op_fn = {'union': self.is_union_by, 'intersection': self.is_intersection_by}[operation]

        for thresh in coefs_values:
            C = (coefs >= thresh).squeeze().detach().cpu().numpy()
            if is_op_fn(coefs, bias, C, v1, v2):
                return C
        return None

    def find_set_and_operation_chan(self, idx: int, v1: np.ndarray = None, v2: np.ndarray = None):
        """Find the selem and the operation given the almost binary features.

        Args:
            v1 (np.ndarray): lower bounds of almost binary input deadzone. Must be of shape self.chan_inputs. Defaults to 0.
            v2 (np.ndarray): upper bounds of almost binary input deadzone. Must be of shape self.chan_inputs. Defaults to 1.

        Returns:
            (np.ndarray, operation): if the selem is found, returns the selem and the operation
            (None, None): if nothing is found, returns None
        """
        for operation in ['union', 'intersection']:
            with torch.no_grad():
                C = self.find_set_for_operation_chan(idx, operation, v1=v1, v2=v2)
            if C is not None:
                return C, operation
        return None, None


    def set_weights(self, new_weights: torch.Tensor) -> torch.Tensor:
        assert self.weight.shape == new_weights.shape, f"Weights must be of same shape {self.weight.shape}"
        self.linear.weight.data = new_weights
        return new_weights

    def set_bias(self, new_bias: torch.Tensor) -> torch.Tensor:
        assert self.bias.shape == new_bias.shape
        self.linear.bias.data = new_bias
        return new_bias

    @staticmethod
    def from_set(C: np.ndarray, operation: str, threshold_mode: str = "tanh", **kwargs):
        net = LUI(chan_inputs=len(C), chan_outputs=1, threshold_mode=threshold_mode, **kwargs)
        assert set(np.unique(C)).issubset([0, 1])
        net.set_weights(torch.tensor(C)[None, :])
        bias_value = -.5 if operation == "union" else -float(C.sum()) + .5
        net.set_bias(torch.FloatTensor([bias_value]))
        return net
