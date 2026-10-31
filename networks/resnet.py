import torch.nn as nn
import torch

import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
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

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()

        # ---- shared stem activations ------------------------------------
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ---- Branch 1: BCPR (PCA block residual) ------------------------
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.fc1 = nn.Linear(512, num_classes)

        # ---- Branch 2: PiD (YUV pixel residual) -------------------------
        self.inplanes = 64
        self.conv1_2 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.layer1_2 = self._make_layer(block, 64,  layers[0])
        self.layer2_2 = self._make_layer(block, 128, layers[1], stride=2)
        self.fc1_2 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def pid_residual_round(self, img, delta=(1.0, 1.0, 1.0)):
        """
        PiD residual with hard (round) quantization in YUV.
        Args:
            img   : [B,3,H,W], float RGB in [0, 1], no normalization
            delta : quantization steps (dY, dU, dV), scalar or triple
        Returns:
            residual: [B,3,H,W], img minus its quantized reconstruction
        """
        B, C, H, W = img.shape
        assert C == 3, "RGB input required"
        device, dtype = img.device, img.dtype

        # linear RGB -> YUV transform (no offset)
        M = torch.tensor([[0.299000, 0.587000, 0.114000],
                          [-0.168736, -0.331264, 0.500000],
                          [0.500000, -0.418688, -0.081312]], device=device, dtype=dtype)
        Minv = torch.linalg.inv(M)

        # reshape: [B,3,H,W] -> [B, N, 3]
        X = img.permute(0, 2, 3, 1).reshape(B, -1, 3)

        # RGB -> YUV
        YUV = X @ M.T

        # quantization steps
        if not isinstance(delta, (list, tuple)):
            delta = (float(delta),) * 3
        d = torch.tensor(delta, device=device, dtype=dtype).view(1, 1, 3)

        # hard quantization
        YUVq = torch.round(YUV / d) * d

        # YUV -> RGB
        Xq = YUVq @ Minv.T
        Xq = Xq.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        # residual
        residual = img - Xq
        return residual

    def pca_block_round_residual_fast_accel_1stRGB_2ndPCA(self, img, q=4.0, block=32):
        """
        BCPR residual: block-wise PCA of the local color covariance, quantization of the
        basis and of the coefficients, and reconstruction.

        Args:
            img:   [B,3,H,W], RGB in [0, 1]
            q:     coefficient quantization step
            block: block size k

        Returns:
            residual_pca: [B,3,H,W]  BCPR residual (img minus reconstruction), used by MORD
            content_pca:  [B,3,H,W]  reconstruction
            rgb_residual: [B,3,H,W]  RGB quantization residual with step q (not used by MORD)
        """
        assert img.dim() == 4 and img.shape[1] == 3
        B, C, H, W = img.shape
        dtype = img.dtype
    
        # =========================
        # (A) RGB quantization residual
        # =========================
        img32 = img.to(torch.float32)
        rgb_q = torch.round(img32 / q) * q
        rgb_residual = (img32 - rgb_q).to(dtype)   # [B,3,H,W]
    
        # =========================
        # (B) block-wise PCA quantization residual (BCPR)
        # =========================
    
        # -------- 1. unfold --------
        patches = F.unfold(img32, kernel_size=block, stride=block)
        Nblocks = patches.shape[-1]
        patches = patches.view(B, C, block * block, Nblocks)
        patches = patches.permute(0, 3, 1, 2).reshape(-1, 3, block * block)  # [M,3,Npix]
        M = patches.shape[0]
    
        # -------- 2. PCA --------
        mu = patches.mean(dim=2, keepdim=True)
        Xc = patches - mu
        Cmat = torch.bmm(Xc, Xc.transpose(1, 2)) / (Xc.shape[2] - 1)
    
        eigvals, V = torch.linalg.eigh(Cmat)
        idx = torch.argsort(eigvals, dim=1, descending=True)
        V = torch.gather(V, 2, idx.unsqueeze(1).expand(-1, 3, -1))
    
        # quantize the PCA basis with step q_v = 0.01
        V = torch.round(V / 0.01) * 0.01
    
        # -------- 3. projection --------
        Z = torch.bmm(V.transpose(1, 2), Xc)  # [M,3,Npix]
    
        # -------- 4. variance re-scaling (sigma, sigma') and coefficient quantization --------
        sigma = torch.tensor([0.08, 0.02, 0.01], device=Z.device, dtype=Z.dtype).view(1,3,1)
        sigma2 = torch.tensor([0.2, 0.05, 0.05], device=Z.device, dtype=Z.dtype).view(1,3,1)
    
        Z_norm = Z / sigma * sigma2
        Z_norm_q = torch.round(Z_norm / q) * q
        Zq = Z_norm_q * sigma / sigma2
    
        # -------- 5. reconstruction --------
        Xq = torch.bmm(V, Zq) + mu
    
        Xq = Xq.view(B, Nblocks, 3, block * block).permute(0, 2, 3, 1)
        Xq = Xq.reshape(B, 3 * block * block, Nblocks)
        Xq = F.fold(Xq, (H, W), kernel_size=block, stride=block).to(dtype)
    
        #Xq = torch.clamp(Xq, 0.0, 255.0).to(dtype)
        residual_pca = img - Xq
        content_pca = Xq
    
        return residual_pca, content_pca, rgb_residual

    def forward(self, x):
        """Two-branch residual model with confidence-based routing.

        Returns a [B, 3] tensor:
            out[:, 0]  BCPR branch logit  (trainable -> updates BCPR encoder)
            out[:, 1]  PiD  branch logit  (trainable -> updates PiD encoder)
            out[:, 2]  fused logit chosen by max-|logit| confidence routing.
                       DETACHED -- for inference / observation only. A loss
                       on column 2 produces no gradient. If you want a
                       consistency loss to update the encoder, put it on
                       column 0 or 1.
        """
        q = 0.006 * 4

        # ---- Branch 1: BCPR (PCA block quantization residual) -----------
        res1, _, _ = self.pca_block_round_residual_fast_accel_1stRGB_2ndPCA(x, q, block=4)
        x1 = self.conv1(res1 * 3.464 / q)
        x1 = self.relu(self.bn1(x1))
        x1 = self.maxpool(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.avgpool(x1).view(x1.size(0), -1)
        out1 = self.fc1(x1)

        # ---- Branch 2: PiD (YUV pixel quantization residual) ------------
        res2 = self.pid_residual_round(x, (0.004, 0.004, 0.004))
        x2 = self.conv1_2(res2 * 3.464 / 0.004)
        x2 = self.relu(self.bn1_2(x2))
        x2 = self.maxpool(x2)
        x2 = self.layer1_2(x2)
        x2 = self.layer2_2(x2)
        x2 = self.avgpool(x2).view(x2.size(0), -1)
        out2 = self.fc1_2(x2)

        # ---- Confidence routing (inference fusion, detached) ------------
        # |logit| measures distance to the decision boundary -- the bigger
        # one is the more confident branch. Detach so a BCE on column 2
        # cannot double-count gradient onto either encoder.
        use_pid = out2.abs() > out1.abs()
        out_fused = torch.where(use_pid, out2.detach(), out1.detach())

        return torch.cat([out1, out2, out_fused], dim=1)  # [B, 3]


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
