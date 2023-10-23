"""Models for AxSpA classification using BiMoNN."""
from typing import List, Tuple

import torch
import torch.nn as nn
# from torchvision.models.resnet import resnet50


from .bimonn import BiMoNN
from general.nn.models.unet import UNet
from general.nn.models.resnet import ResNet_N
from .binary_nn import BinaryNN



# class BimonnAxspaPipeline(BinaryNN):

#     def __init__(self, pretrained_classification: bool = False):
#         super().__init__()

#         self.segmenter: nn.Module = UNet(
#             in_channels=2,
#             n_classes=3,
#             do_activation=False,
#         )

#         self.bimonn: BiMoNN = BiMoNN(
#             channels=[3, 7, 7, 1],
#             kernel_size=[7, 7, 7],
#         )

#         self.classification: nn.Module = ResNet_N(
#             in_channels=2, n_classes=1,
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         segm = self.segmenter(x)
#         bimonn = self.bimonn(segm)
#         return {
#             "segmentation": segm,
#             "bimonn": bimonn,
#             "pred": self.classification(x * bimonn)[..., 0],  # output shape (batch_size,) instead of (batch_size, 1)
#         }


class BimonnAxspaFromSegm(BinaryNN):

    def __init__(
        self,
        bimonn_channels: List[int],
        bimonn_kernel_size: List[int],
        *args, **kwargs
    ):
        super().__init__()

        self.bimonn_channels = bimonn_channels
        self.bimonn_kernel_size = bimonn_kernel_size

        self.segmenter: nn.Module = UNet(
            in_channels=1,
            n_classes=3,
            do_activation=False,
        )

        self.bimonn: BiMoNN = BiMoNN(
            channels=[2] + self.bimonn_channels + [1],
            kernel_size=self.bimonn_kernel_size,
            *args, **kwargs
        )

        self.classification: nn.Module = ResNet_N(
            in_channels=1, n_classes=1,
        )

        self.current_output = {
            "segmentation": None,
            "bimonn": None,
            "pred": None,
        }

    def forward(self, input_: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, segm = input_
        bimonn = self.bimonn(segm)
        pred = self.classification(x * bimonn)[..., 0]  # output shape (batch_size,) instead of (batch_size, 1)
        self.current_output = {
            "segmentation": segm,
            "bimonn": bimonn,
            "pred": pred
        }
        return pred
