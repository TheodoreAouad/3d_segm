import torch.nn as nn

from .bise import BiSE, BiSEC


class COBiSE_base(nn.Module):
    def forward(self, x):
        """
        Run the Baumises and returns the output.

        Args:
            self: write your description
            x: write your description
        """
        output = self.bises[0](x)
        output = self.bises[1](output)
        return output

    def activation_threshold_fn(self, *args, **kwargs):
        """
        The activation threshold function for the first layer.

        Args:
            self: write your description
        """
        return self.dilations1.activation_threshold_fn(*args, **kwargs)

    @property
    def _normalized_weight(self):
        """
        The weight of the first dilations.

        Args:
            self: write your description
        """
        return self.dilations1._normalized_weight

    @property
    def bias(self):
        """
        The bias of the first dilation.

        Args:
            self: write your description
        """
        return self.dilations1.bias

    @property
    def weight(self):
        """
        Weight of the first dilation.

        Args:
            self: write your description
        """
        return self.dilations1.weight

    @property
    def weight_P(self):
        """
        The weight of the first node in the tree.

        Args:
            self: write your description
        """
        return self.dilations1.weight_P

    @property
    def activation_P(self):
        """
        The activation P of the first dilation.

        Args:
            self: write your description
        """
        return self.dilations1.activation_P





class COBiSE(COBiSE_base):

    def __init__(self, share_weights: bool = True, **bise_args):
        """
        Initialise the Bise instance

        Args:
            self: write your description
            share_weights: write your description
            bise_args: write your description
        """
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
        """
        Initialize the bisecant system

        Args:
            self: write your description
            share_weights: write your description
            bisec_args: write your description
        """
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
        """
        The thresholded alpha of the first dilation.

        Args:
            self: write your description
        """
        return self.dilations1.thresholded_alpha