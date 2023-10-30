"""Some code taken from torch source code <https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py>"""
from typing import Optional, Callable

import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
# from torchvision.models.utils import load_state_dict_from_url

from .masked_layers import MaskedConv2d, MaskedBatchNorm2d

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

model_layers = {
    'resnet50': [3, 4, 6, 3],  # difference between 50 and 34 is the block
    'resnet34': [3, 4, 6, 3],
    'resnet18': [2, 2, 2, 2],
    'resnet10': [1, 1, 1, 1],
}

model_blocks = {
    'resnet10': resnet.BasicBlock,
    'resnet18': resnet.BasicBlock,
    'resnet34': resnet.BasicBlock,
    'resnet50': resnet.Bottleneck,
}



class BottleneckNoBatchnorm(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv1x1(inplanes, width)
        self.conv2 = resnet.conv3x3(width, width, stride, groups, dilation)
        self.conv3 = resnet.conv1x1(width, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockNoBatchnorm(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


model_blocks_no_batchnorm = {
    'resnet10': BasicBlockNoBatchnorm,
    'resnet18': BasicBlockNoBatchnorm,
    'resnet34': BasicBlockNoBatchnorm,
    'resnet50': BottleneckNoBatchnorm,
}


class ResNet(resnet.ResNet):
    r""" ResNet model from torch, modified to have more flexibility with layers. From
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    def __init__(
        self,
        n_classes=1000,
        layers=[2, 2, 2, 2],
        planes=[64, 128, 256, 512],
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        block=BasicBlockNoBatchnorm,
        **kwargs
    ):
        super(resnet.ResNet, self).__init__()
        # block = resnet.BasicBlock
        self.layers = layers
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead

            replace_stride_with_dilation = [False] * (len(layers) - 1)

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, planes[0], layers[0])
        for layer_idx in range(1, len(layers)):
            setattr(
                self,
                'layer{}'.format(layer_idx+1),
                self._make_layer(block, planes[layer_idx], layers[layer_idx], stride=2,
                                dilate=replace_stride_with_dilation[layer_idx-1])
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.inplanes * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(len(self.layers)):
            x = getattr(self, 'layer{}'.format(i+1))(x)


        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    @property
    def device(self):
        for p in self.parameters():
            return p.device

    def _get_resolution(self, input_res, filter_size, stride, padding):
        return int((input_res - filter_size + 2*padding) / stride) + 1


class ResNet_N(ResNet):

    def __init__(
        self,
        in_channels=1,
        n_classes=1,
        use_mask=False,
        bg_in=-1,
        bg_transit=0,
        pretrained=False,
        progress=True,
        layers='resnet18',
        planes=[64, 128, 256, 512],
        block=resnet.BasicBlock,
        do_activation=False,
        do_batchnorm=False,
        **kwargs
    ):
        if type(layers) == str:
            model_name = layers
            block = model_blocks[layers] if do_batchnorm else model_blocks_no_batchnorm[layers]
            layers = model_layers[layers]
        else:
            pretrained = False
        
        
        norm_layer = None if do_batchnorm else lambda x: nn.Identity()
        super().__init__(
            layers=layers,
            planes=planes,
            block=block,
            norm_layer=norm_layer,
            **kwargs
        )

        self.bg_in = bg_in
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.do_batchnorm = do_batchnorm

        # if type(pretrained) == bool and pretrained:
        #     state_dict = load_state_dict_from_url(model_urls[model_name],
        #                                         progress=progress)
        #     self.load_state_dict(state_dict)

        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3,
                            bias=False)

        if use_mask:
            self._replace_layers(
                bg_transit=bg_transit,
                transform_conv=False,
                transform_bn=True,
                transform_relu=True
            )
            self.conv1.bg_in = self.bg_in

        # self.bn1 = self._norm_layer(64)

        self.fc = nn.Linear(self.inplanes * resnet.BasicBlock.expansion, n_classes)

        self.do_activation = do_activation
        self.final_activation = torch.sigmoid if n_classes == 1 else lambda x: torch.softmax(x, dim=-1)

        if type(pretrained) == str and pretrained:
            self.load_state_dict(torch.load(pretrained))

    def forward(self, x, do_activation=None):
        if do_activation is None:
            act = self.do_activation
        else:
            act = do_activation
        x = super().forward(x)
        if act:
            return self.final_activation(x)
        return x

    def _replace_layers(self, bg_transit, transform_conv=True, transform_bn=True, transform_relu=True):
        for name, layer in self.named_modules():
            if transform_conv and isinstance(layer, nn.Conv2d):
                self._replace_layer(
                    layer, name, MaskedConv2d,
                    args={'conv2d': layer, "bg_in": bg_transit, "bg_out": bg_transit}
                )
            if transform_bn and isinstance(layer, nn.BatchNorm2d):
                self._replace_layer(
                    layer, name, MaskedBatchNorm2d,
                    args={
                        'num_features': layer.num_features, 'bg_in': bg_transit, 'bg_out': bg_transit,
                        'eps': layer.eps, 'momentum': layer.momentum,
                        'affine': layer.affine, 'track_running_stats': layer.track_running_stats,
                    }
                )
            if transform_relu and isinstance(layer, nn.ReLU):
                self._replace_layer(
                    layer, name, nn.LeakyReLU,
                    args={"negative_slope": 0.01}
                )

    def _replace_layer(self, layer, name, NewLayer, args):
        mods = name.split('.')
        cur_module = self
        for mod_attr in mods[:-1]:
            cur_module = getattr(cur_module, mod_attr)
        setattr(cur_module, mods[-1], NewLayer(**args))
