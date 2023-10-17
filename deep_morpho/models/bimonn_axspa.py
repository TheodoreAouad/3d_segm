"""Models for AxSpA classification using BiMoNN."""

import torch
import torch.nn as nn
# from torchvision.models.resnet import resnet50


from .bimonn import BiMoNN
from general.nn.models.unet import UNet
from general.nn.models.resnet import ResNet_N
from .binary_nn import BinaryNN



class BimonnAxspaPipeline(BinaryNN):

    def __init__(self, pretrained_classification: bool = False):
        super().__init__()

        self.segmenter: nn.Module = UNet(
            in_channels=2,
            n_classes=3,
            do_activation=False,
        )

        self.bimonn: BiMoNN = BiMoNN(
            channels=[3, 7, 7, 1],
            kernel_size=[7, 7, 7],
        )

        self.classification: nn.Module = ResNet_N(
            in_channels=2, n_classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        segm = self.segmenter(x)
        bimonn = self.bimonn(segm)
        return {
            "segmentation": segm,
            "bimonn": bimonn,
            "pred": self.classification(x * bimonn),
        }


class BimonnAxspaFromSegm(BinaryNN):

    def __init__(self, pretrained_classification: bool = False):
        super().__init__()

        self.segmenter: nn.Module = UNet(
            in_channels=2,
            n_classes=3,
            do_activation=False,
        )

        self.bimonn: BiMoNN = BiMoNN(
            channels=[3, 7, 7, 1],
            kernel_size=[7, 7, 7],
        )

        self.classification: nn.Module = ResNet_N(
            in_channels=2, n_classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        segm = self.segmenter(x).argmax(dim=1, keepdim=True).detach()
        segm = torch.cat([segm == 0, segm == 1, segm == 2], dim=1).float()
        bimonn = self.bimonn(segm)
        return {
            "segmentation": segm,
            "bimonn": bimonn,
            "pred": self.classification(x * bimonn),
        }
