"""Some code taken from torch source code <https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py>"""

import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
# from torchvision.models.utils import load_state_dict_from_url

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
    'resnet18': resnet.BasicBlock,
    'resnet34': resnet.BasicBlock,
    'resnet50': resnet.Bottleneck,
}


class MaskedConv2d(nn.Module):
    """
    This layer performs a conv2d then masks the output. All the outputs that touched the
    background are put to 0.
    """
    def __init__(
        self,
        conv2d,
        bg_in=-1,
        bg_out=0,
    ):
        super().__init__()
        self.bg_in = bg_in
        self.bg_out = bg_out

        self.in_channels = conv2d.in_channels
        self.out_channels = conv2d.out_channels
        self.kernel_size = conv2d.kernel_size
        self.stride = conv2d.stride
        self.padding = conv2d.padding
        self.bias = conv2d.bias is None

        self.conv2d = conv2d

    def forward(self, x):

        xout = self.conv2d(x)
        if self.bg_in is None:
            return xout

        mask_in = (x == self.bg_in).float().to(self.device).detach()
        kern = torch.ones((1, x.shape[1], *self.kernel_size)).to(self.device)
        mask_out = (F.conv2d(input=mask_in, weight=kern, stride=self.stride, padding=self.padding) > 0).to(self.device).detach()
        return (xout - self.bg_out) * (~ mask_out) + self.bg_out

    @property
    def device(self):
        for p in self.parameters():
            return p.device


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class MaskedBatchNorm2d(nn.BatchNorm2d):
    """
    Base taken from <https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py>.
    """
    def __init__(self, num_features, bg_in=None, bg_out=None, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats,
        )
        self.bg_in = bg_in
        self.bg_out = bg_out


    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.bg_in is not None:
            mask = input != self.bg_in
        else:
            mask = torch.ones_like(input)

        if self.training:

            # calculate running estimates
            sum_chan = mask[:, 0, ...].sum((1, 2)).float().detach()

            # Computing mean of foreground
            mean = (mask * input).sum([2, 3])/sum_chan[:, None]
            mean = mean.mean(0)

            # Computing var of foreground
            var = (input - mean[None, :, None, None])**2
            var = (mask * var).sum([2, 3])/sum_chan[:, None]
            var = var.mean(0)

            # TODO: use biased var
            # var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = ((input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps)))
        if self.affine:
            input = (input * self.weight[None, :, None, None] + self.bias[None, :, None, None])

        if self.bg_out is not None:
            input = (input - self.bg_out) * mask + self.bg_out      # put all elements of mask to bg_out faster than with input[mask]

        return input


    def extra_repr(self):
        return '{num_features}, bg_in={bg_in}, bg_out={bg_out}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class MyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input



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
        block=resnet.BasicBlock,
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
        self.flatten = Flatten()
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
        use_mask=True,
        bg_in=-1,
        bg_transit=0,
        pretrained=False,
        progress=True,
        layers='resnet18',
        planes=[64, 128, 256, 512],
        block=resnet.BasicBlock,
        do_activation=False,
        **kwargs
    ):
        if type(layers) == str:
            model_name = layers
            block = model_blocks[layers]
            layers = model_layers[layers]
        else:
            pretrained = False
        super().__init__(
            layers=layers,
            planes=planes,
            block=block,
            **kwargs
        )

        self.bg_in = bg_in
        self.n_classes = n_classes
        self.in_channels = in_channels

        if type(pretrained) == bool and pretrained:
            # state_dict = load_state_dict_from_url(model_urls[model_name],
            #                                     progress=progress)
            # self.load_state_dict(state_dict)
            raise NotImplementedError("Pretrained models not implemented yet")

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
