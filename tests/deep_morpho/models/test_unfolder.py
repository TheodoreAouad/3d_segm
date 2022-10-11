import torch
import torch.nn as nn

from deep_morpho.models import Unfolder


class TestUnfolder:

    @staticmethod
    def test_conv():
        unfolder = Unfolder(kernel_size=(3, 3), padding=1)
        x = torch.randint(0, 2, (50, 50)).float()

        conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding='same', padding_mode="zeros", bias=False)
        ot1 = conv(x[None, None, ...]).squeeze()

        weights = conv.weight[0, 0]
        kron_weight = torch.kron(
            torch.ones(
                [1 + x.shape[k] + 1 - 3 + 1 for k in range(2)],
            ),
            weights
            # weights.flip([0, 1])
        ).squeeze()

        conv_sum = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(3, 3), bias=False)
        conv_sum.weight.data = torch.ones_like(conv_sum.weight.data, requires_grad=False)
        ot2 = conv_sum((unfolder(x) * kron_weight)[None, None, ...]).squeeze()

        assert (ot1 - ot2).abs().sum() < 1e-5

    @staticmethod
    def test_grad():
        unfolder = Unfolder(kernel_size=(3, 3), padding=1)
        x = torch.randint(0, 2, (50, 50)).float()

        conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding='same', padding_mode="zeros", bias=False)
        ot1 = conv(x[None, None, ...]).squeeze()

        weights = torch.zeros_like(conv.weight, requires_grad=True)
        kron_weight = torch.kron(
            torch.ones(
                [1 + x.shape[k] + 1 - 3 + 1 for k in range(2)],
            ),
            weights[0, 0]
            # weights.flip([0, 1])
        ).squeeze()

        conv_sum = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(3, 3), bias=False)
        conv_sum.weight.data = torch.ones_like(conv_sum.weight.data, requires_grad=False)
        ot2 = conv_sum((unfolder(x) * kron_weight)[None, None, ...]).squeeze()

        ot1.sum().backward()
        ot2.sum().backward()

        assert (conv.weight.grad - weights.grad).abs().sum() < 1e-6