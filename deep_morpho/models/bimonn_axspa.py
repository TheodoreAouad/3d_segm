"""Models for AxSpA classification using BiMoNN."""
from typing import List, Tuple, Dict
from abc import ABC

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


class SpalikeMergedInputModel(BinaryNN, ABC):
    """Model to classify Spalike dataset with merged input and segmentation."""
    pass



class BimonnAxspaClassifier(BinaryNN, ABC):

    def __init__(
        self,
        bimonn_channels: List[int],
        bimonn_kernel_size: List[int],
        *args, **kwargs
    ):
        super().__init__()

        self.bimonn_channels = bimonn_channels
        self.bimonn_kernel_size = bimonn_kernel_size

        self.bimonn: BiMoNN = BiMoNN(
            channels=[2] + self.bimonn_channels + [1],
            kernel_size=self.bimonn_kernel_size,
            *args, **kwargs
        )

        self.classification: SpalikeMergedInputModel

        self.current_output = {
            "segmentation": None,
            "bimonn": None,
            "pred": None,
        }

    def forward(self, input_: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, segm = input_
        bimonn = self.bimonn(segm)
        pred = self.classification(x * bimonn)
        self.current_output = {
            "segmentation": segm,
            "bimonn": bimonn,
            "pred": pred
        }
        return pred
    
    @classmethod
    def default_args(cls) -> Dict[str, dict]:
        """Return the default arguments of the model, in the format of argparse.ArgumentParser"""
        res = super().default_args()
        res.update({
            k: v for k, v in BiMoNN.default_args().items()
            if k not in res
            and k not in [
                "channels", "kernel_size",
            ]
        })
        return res



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.activation_fn = activation_fn()

    def forward(self, x):
        return self.activation_fn(x + self.conv(x))



class ConvSpalikeMerged(SpalikeMergedInputModel):

    def __init__(
        self,
        channels: List[int],
        kernel_size: List[int],
        classif_neurons: List[int] = [],
        activation_fn: nn.Module = nn.ReLU,
        do_maxpool: bool = True,
    ):
        super().__init__()

        self.channels = [1] + channels
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * (len(self.channels) - 1)
        self.kernel_size = kernel_size
        self.classif_neurons = [channels[-1]] + classif_neurons
        self.do_maxpool = do_maxpool

        assert len(self.channels) == len(self.kernel_size) + 1, "Number of channels and kernel sizes must match"

        self.conv_layers = []
        self.conv_layers.append(nn.Conv2d(self.channels[0], self.channels[1], kernel_size[0], stride=2))
        self.conv_layers.append(activation_fn())
        if self.do_maxpool:
            self.conv_layers.append(nn.MaxPool2d(2))

        for i in range(1, len(self.channels) - 1):
            self.conv_layers.append(nn.Conv2d(self.channels[i], self.channels[i + 1], kernel_size[i]))
            self.conv_layers.append(activation_fn())
            # self.conv_layers.append(ResidualBlock(self.channels[i], self.channels[i + 1], kernel_size[i + 1]))
            if self.do_maxpool:
                self.conv_layers.append(nn.MaxPool2d(2))
        self.conv_layers = nn.Sequential(*self.conv_layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.classif_layers = []
        for i in range(len(self.classif_neurons) - 1):
            self.classif_layers.append(nn.Linear(self.classif_neurons[i], self.classif_neurons[i + 1]))
            self.classif_layers.append(activation_fn())
        self.classif_layers.append(nn.Linear(self.classif_neurons[-1], 1))
        self.classif_layers = nn.Sequential(*self.classif_layers)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classif_layers(x)[..., 0]
        return x



class ResnetSpalikeMerged(SpalikeMergedInputModel):

    def __init__(
        self,
        classif_layers="resnet10",
        do_batchnorm=False,
    ):
        super().__init__()

        self.classification: nn.Module = ResNet_N(
            in_channels=1, n_classes=1, layers=classif_layers, do_batchnorm=do_batchnorm,
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = self.classification(x)[..., 0]
        return x


class BimonnAxspaResnet(BimonnAxspaClassifier):

    def __init__(
        self,
        classif_layers="resnet10",
        do_batchnorm=False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.classification: nn.Module = ResnetSpalikeMerged(
            classif_layers=classif_layers, do_batchnorm=do_batchnorm,
        )


class BimonnAxspaConv(BimonnAxspaClassifier):

    def __init__(
        self,
        classif_channels: List[int],
        classif_kernel_size: List[int],
        classif_neurons: List[int] = [],
        do_maxpool: bool = True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.classif_channels = classif_channels
        self.classif_kernel_size = classif_kernel_size
        self.classif_neurons = classif_neurons
        self.do_maxpool = do_maxpool

        self.classification: nn.Module = ConvSpalikeMerged(
            channels=self.classif_channels,
            kernel_size=self.classif_kernel_size,
            classif_neurons=self.classif_neurons,
            activation_fn=nn.ReLU,
            do_maxpool=self.do_maxpool,
        )
