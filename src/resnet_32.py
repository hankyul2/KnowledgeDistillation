from typing import Type, Union

import torch
from torch import nn
import torch.nn.functional as F

from src.log import get_base_model


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


class ResNet_32(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, BottleNeck]], nblock: list, nclass: int = 1000,
                 channels: list = [16, 32, 64], norm_layer: nn.Module = nn.BatchNorm2d, groups=1,
                 base_width=64):
        super(ResNet_32, self).__init__()
        self.groups = groups
        self.base_width = base_width
        self.norm_layer = norm_layer
        self.in_channels = channels[0]

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=(7, 7), stride=1, padding=(1, 1), bias=False)
        self.bn1 = self.norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1] * block.factor, nclass)

        self.layers = [self.make_layer(block=block, nblock=nblock[i], channels=channels[i]) for i in range(len(nblock))]
        self.register_layer()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        feature = self.flatten(self.avgpool(x))
        out = self.fc(feature)
        return activations, feature, out

    def predict(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            x = layer(x)
        return self.fc(self.flatten(self.avgpool(x)))

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


def get_model(model_name: str, nclass=1000, zero_init_residual=False, pretrained_dataset=None) -> nn.Module:
    if model_name == 'resnet_32_20':
        model = ResNet_32(block=BasicBlock, nblock=[3, 3, 3], nclass=nclass)
    elif model_name == 'resnet_32_110':
        model = ResNet_32(block=BasicBlock, nblock=[18, 18, 18], nclass=nclass)

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

    if pretrained_dataset:
        model.load_state_dict(torch.load(get_base_model(model_name, pretrained_dataset), map_location='cpu')['weight'])

    return model
