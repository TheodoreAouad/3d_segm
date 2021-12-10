import torch.nn as nn

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

    def forward(self, x):
        return self.threshold_layer(self.linear(x))

    @property
    def bias(self):
        return self.linear.bias

    @property
    def weight(self):
        return self.linear.weight

    @property
    def weights(self):
        return self.weight
