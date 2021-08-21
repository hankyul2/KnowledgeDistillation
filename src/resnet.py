import torch
from torch import nn
import torch.nn.functional as F


def conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3),
                        stride=1 if in_channels==out_channels else 2, padding=(1,1), groups=1, dilation=1)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.w1 = conv3x3(in_channels, out_channels)
        self.w2 = conv3x3(out_channels, out_channels)
        if in_channels == out_channels:
            self.skip_connection = lambda x: x
        else:
            self.skip_connection = lambda x: torch.cat([F.max_pool2d(x, kernel_size=(3,3), stride=2, padding=(1,1)), torch.zeros_like(F.max_pool2d(x, kernel_size=(3,3), stride=2, padding=1))], dim=1)

    def forward(self, x):
        return F.relu(self.skip_connection(x) + self.w2(F.relu(self.w1(x))))


class ResNetLayer(nn.Module):
    def __init__(self, nlayer, in_channels, out_channels, bottleneck):
        super(ResNetLayer, self).__init__()
        self.nlayer = nlayer
        self.resBlocks = nn.ModuleList([
            BasicBlock(in_channels = in_channels if i == 0 else out_channels, out_channels = out_channels) for i in range(nlayer)
        ])

    def forward(self, x):
        for block in self.resBlocks:
            x = block(x)
        return x


class ResNet(nn.Module):
    def __init__(self, layer_group: list, bottleneck: bool, in_channels=[64, 64, 128, 256], out_channels=[64, 128, 256, 512]) -> nn.Module:
        """
        ResNet(He, 2015) model
        :rtype: ResNet Model
        """
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64)
        self.layers = nn.ModuleList(
            [ResNetLayer(nlayer=layer_group[i], in_channels=in_channels[i], out_channels=out_channels[i], bottleneck=bottleneck) for i in range(len(layer_group))]
        )

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.layers:
            x = layer(x)
        return F.adaptive_max_pool2d(x, 1)




def get_model(model_name: str) -> nn.Module:
    if model_name == 'resnet34':
        model = ResNet(layer_group=[3, 4, 6, 3], bottleneck=False)
    return model
