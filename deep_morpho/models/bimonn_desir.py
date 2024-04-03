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


class DesirMergedInputModel(BinaryNN, ABC):
    """Model to classify Desir dataset with merged input and segmentation."""
    pass



class BimonnDesirClassifierMerged(BinaryNN, ABC):

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

        self.classification: DesirMergedInputModel

        self.current_output = {
            "segmentation": None,
            "bimonn": None,
            "pred": None,
        }

    def forward(self, input_: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, segm = input_
        bimonn_output = self.bimonn(segm)
        pred = self.classification(x * bimonn_output)
        self.current_output = {
            "segmentation": segm,
            "bimonn": bimonn_output,
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


class BimonnDesirClassifierChannel(BimonnDesirClassifierMerged):
    def forward(self, input_: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, segm = input_
        bimonn_output = self.bimonn(segm)
        classif_input = torch.cat([x, bimonn_output], dim=1)
        pred = self.classification(classif_input)
        self.current_output = {
            "segmentation": segm,
            "bimonn": bimonn_output,
            "pred": pred
        }
        return pred


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.activation_fn = activation_fn()

    def forward(self, x):
        return self.activation_fn(x + self.conv(x))




class ResnetDesirMerged(DesirMergedInputModel):

    def __init__(
        self,
        classif_layers="resnet18",
        do_batchnorm=False,
        in_channels=2,
    ):
        super().__init__()

        self.classification: nn.Module = ResNet_N(
            in_channels=in_channels, n_classes=1, layers=classif_layers, do_batchnorm=do_batchnorm,
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = self.classification(x)[..., 0]
        return x


class BimonnDesirResnetMerged(BimonnDesirClassifierMerged):

    def __init__(
        self,
        classif_layers="resnet18",
        do_batchnorm=False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.classification: nn.Module = ResnetDesirMerged(
            in_channels=2, classif_layers=classif_layers, do_batchnorm=do_batchnorm,
        )

class BimonnDesirResnetChannel(BimonnDesirClassifierChannel):

    def __init__(
        self,
        classif_layers="resnet18",
        do_batchnorm=False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.classification: nn.Module = ResnetDesirMerged(
            in_channels=3, classif_layers=classif_layers, do_batchnorm=do_batchnorm,
        )