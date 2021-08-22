from typing import Type, Union

from torch import nn

from src.resnet import ResNet, BasicBlock, BottleNeck


class ResNet_32(ResNet):
    def __init__(self, block: Type[Union[BasicBlock, BottleNeck]], nblock: list, nclass: int = 1000,
                 channels: list = [16, 32, 64], norm_layer: nn.Module = nn.BatchNorm2d, groups=1,
                 base_width=64):
        super(ResNet_32, self).__init__(block, nblock, nclass)
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
        for layer in self.layers:
            x = layer(x)
        return self.fc(self.flatten(self.avgpool(x)))


def get_model(model_name: str, device:str ='cpu', nclass=1000, zero_init_residual=False) -> nn.Module:
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

    return model
