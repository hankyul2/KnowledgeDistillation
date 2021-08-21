from typing import Optional, Type, Union

import torch
from torch import nn
import torch.nn.functional as F


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3),
                     stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                    stride=stride, padding=(0,0), groups=1, bias=False, dilation=1)


class BasicBlock(nn.Module):
    factor = 1
    def __init__(self, in_channels, out_channels, stride=1, norm_layer=nn.BatchNorm2d,
                 downsample =None, groups=1, dilations=1, base_width=64.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn1 = norm_layer(num_features=out_channels)
        self.bn2 = norm_layer(num_features=out_channels)
        self.downsample  = downsample if downsample else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.downsample(x) + self.bn2(self.conv2(out)))


class Bottleneck(nn.Module):
    factor = 4
    def __init__(self, in_channels, out_channels, stride, norm_layer=nn.BatchNorm2d,
                 downsample=None, groups=1, dilation=1, base_width=64.0):
        super(Bottleneck, self).__init__()
        width = int(out_channels * (base_width/64.0)) * groups # ResnetX, WideResNet
        self.conv1 = conv1x1(in_channels, width)
        self.conv2 = conv3x3(width, width, stride, dilation, groups) # Resnet 1.5
        self.conv3 = conv1x1(width, out_channels*self.factor)
        self.bn1 = norm_layer(width)
        self.bn2 = norm_layer(width)
        self.bn3 = norm_layer(out_channels*self.factor)
        self.downsample = downsample if downsample else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return F.relu(self.downsample(x) + self.bn3(self.conv3(out)))


class ResNet(nn.Module):
    def __init__(self, block, nlayer, nclass=1000, norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=(7, 7), stride=2, padding=(1, 1), bias=False)
        self.bn1 = norm_layer(num_features=self.in_channel)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=(1,1))
        self.layer1 = self.make_layer(block, nlayer[0], 64, 1)
        self.layer2 = self.make_layer(block, nlayer[1], 128, 2)
        self.layer3 = self.make_layer(block, nlayer[2], 256, 2)
        self.layer4 = self.make_layer(block, nlayer[3], 512, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512*block.factor, nclass)

    def make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], nblock: int, out_channels: int, stride:int) -> nn.Sequential:
        downsample = None
        if stride == 2 or self.in_channel != out_channels * block.factor:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, out_channels * block.factor, stride=stride),
                self.norm_layer(out_channels * block.factor)
            )

        layers = []
        for i in range(nblock):
            if i == 1:
                self.in_channel = out_channels * block.factor
                stride = 1
                downsample = None
            layers.append(block(in_channels=self.in_channel, out_channels=out_channels,
                                stride=stride, norm_layer=self.norm_layer, downsample=downsample))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.flatten(self.avg_pool(x)))


def get_model(model_name: str) -> nn.Module:
    if model_name == 'resnet18':
        model = ResNet(block=BasicBlock, nlayer=[2, 2, 2, 2], nclass=1000)
    elif model_name == 'resnet34':
        model = ResNet(block=BasicBlock, nlayer=[3, 4, 6, 3], nclass=1000)
    elif model_name == 'resnet50':
        model = ResNet(block=Bottleneck, nlayer=[3, 4, 6, 3], nclass=1000)
    return model
