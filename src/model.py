# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - TorchVision

import numpy as np

import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """ 1x1 convolution """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class WideBasicBlock(nn.Module):
    """ BN-ReLU-Conv - DropOut - BN-ReLU-Conv """
    def __init__(self, in_planes, out_planes, stride=1, dropout_rate=0):
        super(WideBasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, out_planes, stride=stride)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes, stride=1)

        shortcut = []
        if stride != 1 or in_planes != out_planes:
            shortcut.append(conv1x1(in_planes, out_planes, stride=stride))
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        x   = self.relu(self.bn1(x))
        out = self.dropout(self.conv1(x))
        out = self.conv2(self.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    """ WRN-n-k (n: number of layers; k: widening factor """
    def __init__(self, cfg):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((cfg.num_layers - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        N = (cfg.num_layers - 4) / 6
        k = cfg.widening_factor
        nGroups = [16, 16*k, 32*k, 64*k]

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(cfg.num_channels, nGroups[0])
        self.layer1 = self._make_layer(WideBasicBlock, nGroups[1], N, stride=1, dropout_rate=cfg.dropout_rate)
        self.layer2 = self._make_layer(WideBasicBlock, nGroups[2], N, stride=2, dropout_rate=cfg.dropout_rate)
        self.layer3 = self._make_layer(WideBasicBlock, nGroups[3], N, stride=2, dropout_rate=cfg.dropout_rate)
        self.bn1 = nn.BatchNorm2d(nGroups[3])
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.linear = nn.Linear(nGroups[3], cfg.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def get_params_and_initialize(model):
    weights, biases = [], []
    for name, param in model.named_parameters():
        if "conv" in name:
            nn.init.kaiming_normal_(param, nonlinearity='relu')
            weights.append(param)
        elif "weight" in name:
            if "bn" in name: nn.init.ones_(param)
            else:            nn.init.kaiming_normal_(param, nonlinearity='relu')
            weights.append(param)
        else:
            nn.init.zeros_(param)
            biases.append(param)

    return weights, biases

