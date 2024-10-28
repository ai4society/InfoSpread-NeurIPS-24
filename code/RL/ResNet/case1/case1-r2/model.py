import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.sigmoid(self.bn1(self.conv1(x)))  # Replaced ReLU with Sigmoid
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.sigmoid(out)  # Replaced ReLU with Sigmoid
        return out

class ResNet(nn.Module):
    def __init__(self, num_nodes, num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2)
        self.fc = nn.Linear(128, 1)  # Changed output features to 1

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), 1, int(x.size(1)**0.5), int(x.size(1)**0.5))
        out = torch.sigmoid(self.bn1(self.conv1(x)))  # Replaced ReLU with Sigmoid
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        _, _, h, w = out.size()
        out = F.avg_pool2d(out, (h, w))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
