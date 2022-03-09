import torch.nn as nn

from deep_morpho.models import BiMoNN



def extract_params_to_normalize(model):
    for bisel_layer in model.layers:
        yield bisel_layer.normalized_weight


class QuadraticNormRegularization(nn.Module):

    def __init__(self, model, lower_bound=0, upper_bound=1):
        super().__init__()
        assert isinstance(model, BiMoNN)
        self.model = model
        self.parameters = extract_params_to_normalize(model)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self):
        loss = 0
        for weights in self.parameters:
            loss += self.penalize_weight(weights)
        return loss

    def penalize_weight(self, weights):
        return (
            self.lower_bound_pen(weights[weights < self.lower_bound]).sum() +
            self.upper_bound_pen(weights[weights > self.upper_bound]).sum()
        )

    def lower_bound_pen(self, x):
        return x ** 2

    def upper_bound_pen(self, x):
        return (x - 1) ** 2
