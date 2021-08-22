import os
from pathlib import Path
from typing import Type, Union

from dill.tests.test_restricted import restricted_func
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False)


def conv3x3(in_channels, out_channels, stride=1, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False, groups=groups)


class BasicBlock(nn.Module):
    factor = 1

    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3(in_channels=out_channels, out_channels=out_channels)
        self.bn1 = norm_layer(out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample if downsample else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.downsample(x) + self.bn2(self.conv2(out)))


class BottleNeck(nn.Module):
    factor = 4

    def __init__(self, in_channels, out_channels, stride, norm_layer, downsample=None, groups=1, base_width=64):
        super(BottleNeck, self).__init__()
        width = int(out_channels * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(in_channels, width)
        self.conv2 = conv3x3(width, width, stride, groups=groups)
        self.conv3 = conv1x1(width, out_channels * self.factor)
        self.bn1 = norm_layer(width)
        self.bn2 = norm_layer(width)
        self.bn3 = norm_layer(out_channels * self.factor)
        self.downsample = downsample if downsample else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return F.relu(self.downsample(x) + self.bn3(self.conv3(out)))


class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, BottleNeck]], nblock: list, nclass: int = 1000,
                 channels: list = [64, 128, 256, 512], norm_layer: nn.Module = nn.BatchNorm2d, groups=1,
                 base_width=64) -> None:
        super(ResNet, self).__init__()
        self.groups = groups
        self.base_width = base_width
        self.norm_layer = norm_layer
        self.in_channels = channels[0]

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=(7, 7), stride=2, padding=(1, 1), bias=False)
        self.bn1 = self.norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1] * block.factor, nclass)

        self.layers = [self.make_layer(block=block, nblock=nblock[i], channels=channels[i]) for i in range(len(nblock))]
        self.register_layer()

    def register_layer(self):
        for i, layer in enumerate(self.layers):
            exec('self.layer{} = {}'.format(i + 1, 'layer'))

    def make_layer(self, block: Type[Union[BasicBlock, BottleNeck]], nblock: int, channels: int) -> nn.Sequential:
        layers = []
        downsample = None
        stride = 1
        if self.in_channels != channels * block.factor:
            stride = 2
            downsample = nn.Sequential(
                conv1x1(self.in_channels, channels * block.factor, stride=stride),
                nn.BatchNorm2d(channels * block.factor)
            )
        for i in range(nblock):
            if i == 1:
                stride = 1
                downsample = None
                self.in_channels = channels * block.factor
            layers.append(block(in_channels=self.in_channels, out_channels=channels,
                                stride=stride, norm_layer=self.norm_layer, downsample=downsample,
                                groups=self.groups, base_width=self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for layer in self.layers:
            x = layer(x)
        return self.fc(self.flatten(self.avgpool(x)))


def get_model(model_name: str, device='cpu', nclass=1000, zero_init_residual=False, pretrained=False) -> nn.Module:
    if model_name == 'resnet18':
        model = ResNet(block=BasicBlock, nblock=[2, 2, 2, 2], nclass=nclass)
    elif model_name == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], nclass=nclass)
    elif model_name == 'resnet50':
        model = ResNet(BottleNeck, [3, 4, 6, 3], nclass=nclass)
    elif model_name == 'resnet101':
        model = ResNet(BottleNeck, [3, 4, 23, 3], nclass=nclass)
    elif model_name == 'resnet152':
        model = ResNet(BottleNeck, [3, 8, 36, 3], nclass=nclass)
    elif model_name == 'resnext50_32x4d':
        model = ResNet(BottleNeck, [3, 8, 36, 3], nclass=nclass, groups=32, base_width=4)
    elif model_name == 'wide_resnet50_2':
        model = ResNet(BottleNeck, [3, 8, 36, 3], nclass=nclass, base_width=128)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, BottleNeck) and zero_init_residual:
            nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock) and zero_init_residual:
            nn.init.constant_(m.bn2.weight, 0)

    if pretrained:
        Path(os.path.join('pretrained', model_name)).mkdir(parents=True, exist_ok=True)
        state_dict = load_state_dict_from_url(url=model_urls[model_name],
                                              model_dir=os.path.join('pretrained', model_name),
                                              progress=True, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    return model
