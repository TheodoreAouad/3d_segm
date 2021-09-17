import torch.nn as nn

from .bise import BiSE, BiSEC


class COBiSE_base(nn.Module):
    def forward(self, x):
        output = self.bises[0](x)
        output = self.bises[1](output)
        return output

    def activation_threshold_fn(self, *args, **kwargs):
        return self.dilations1.activation_threshold_fn(*args, **kwargs)

    @property
    def _normalized_weight(self):
        return self.dilations1._normalized_weight

    @property
    def bias(self):
        return self.dilations1.bias

    @property
    def weight(self):
        return self.dilations1.weight

    @property
    def weight_P(self):
        return self.dilations1.weight_P

    @property
    def activation_P(self):
        return self.dilations1.activation_P





class COBiSE(COBiSE_base):

    def __init__(self, share_weights: bool = True, **bise_args):
        super().__init__()
        self.share_weights = share_weights


        self.dilations1 = BiSE(**bise_args)
        if self.share_weights:
            self.dilations2 = BiSE(shared_weights=self.dilations1.weights, shared_weight_P=self.dilations1.weight_P, **bise_args)
        else:
            self.dilations2 = BiSE(**bise_args)

        self.bises = [self.dilations1, self.dilations2]


class COBiSEC(COBiSE_base):

    def __init__(self, share_weights: bool = True, **bisec_args):
        super().__init__()
        self.share_weights = share_weights


        self.dilations1 = BiSEC(**bisec_args)
        if self.share_weights:
            bisec_args['alpha_init'] = self.dilations1.alpha
            self.dilations2 = BiSEC(
                shared_weights=self.dilations1.weights,
                shared_weight_P=self.dilations1.weight_P,
                invert_thresholded_alpha=True,
                **bisec_args
            )
        else:
            self.dilations2 = BiSEC(invert_thresholded_alpha=True, **bisec_args)

        self.bises = [self.dilations1, self.dilations2]

    @property
    def thresholded_alpha(self):
        return self.dilations1.thresholded_alpha