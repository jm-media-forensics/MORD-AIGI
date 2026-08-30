# -*- coding: utf-8 -*-
"""
NPR detector backbone.
Single-branch truncated ResNet (conv1 + layer1 + layer2) on the NPR residual:

    NPR = x - interpolate(x, 0.5)

i.e. each pixel minus the value of a 2x2 nearest down-up-sampled version
(the neighboring-pixel-relationship residual of Tan et al., CVPR 2024).

Same backbone capacity as the MORD branches (truncated ResNet-50, 512-d head)
so the comparison against MORD is fair; only the residual operator differs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNetNPR(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128 * block.expansion, num_classes)  # 512 -> num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def interpolate(self, img, factor):
        return F.interpolate(
            F.interpolate(img, scale_factor=factor, mode='nearest',
                          recompute_scale_factor=True),
            scale_factor=1.0 / factor, mode='nearest', recompute_scale_factor=True)

    def forward(self, x):
        # NPR residual: pixel minus its 2x2 down-up-sampled value
        n, c, w, h = x.shape
        if w % 2 == 1:
            x = x[:, :, :-1, :]
        if h % 2 == 1:
            x = x[:, :, :, :-1]
        NPR = x - self.interpolate(x, 0.5)

        z = self.conv1(NPR * 2.0)
        z = self.maxpool(self.relu(self.bn1(z)))
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.avgpool(z).view(z.size(0), -1)
        return self.fc1(z)            # [B, num_classes]


def resnet50(num_classes=1, **kwargs):
    return ResNetNPR(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
