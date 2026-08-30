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

        self.unfoldSize = 2
        self.unfoldIndex = 0
        assert self.unfoldSize > 1
        assert -1 < self.unfoldIndex and self.unfoldIndex < self.unfoldSize*self.unfoldSize
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 , layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(512 * block.expansion, 1)
        self.fc1 = nn.Linear(512, num_classes)
        self.fc1_2 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn_fc3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, num_classes)
        
        self.inplanes = 64
        self.conv1_2 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.layer1_2 = self._make_layer(block, 64 , layers[0])
        self.layer2_2 = self._make_layer(block, 128, layers[1], stride=2)
        self.fc_comb = nn.Linear(1024, 1)
        self.gate = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
            )

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
        
    #change 
    def pca_block_round_residual(self, img, block=32):
    
      B, C, H, W = img.shape
      assert C == 3, "RGB only"
      device = img.device
      residual = torch.zeros_like(img)
  
     
      for by in range(0, H, block):
          for bx in range(0, W, block):
              y2 = min(by + block, H)
              x2 = min(bx + block, W)
  
              patch = img[:, :, by:y2, bx:x2]  # [B,3,hb,wb]
              ph, pw = patch.shape[-2:]
  
              # reshape  (B, N, 3)
              X = patch.permute(0, 2, 3, 1).reshape(B, -1, 3)
              mu = X.mean(dim=1, keepdim=True)
              Xc = X - mu
  
              
              Cmat = torch.bmm(Xc.transpose(1,2), Xc) / (Xc.shape[1]-1)
              eigvals, V = torch.linalg.eigh(Cmat)      # [B,3]
              idx = torch.argsort(eigvals, dim=1, descending=True)
              V = torch.stack([V[b,:,idx[b]] for b in range(B)], dim=0)
  
             
              Z = torch.bmm(Xc, V)
  
              
              Zq = torch.round(Z)
  
                
              Xq = torch.bmm(Zq, V.transpose(1,2)) + mu
              Xq = Xq.reshape(B, ph, pw, 3).permute(0, 3, 1, 2)
              residual[:, :, by:y2, bx:x2] = patch - Xq

      return residual

    @torch.no_grad()
    def pca_block_round_residual_fast_accel(self,img, q=1.0, block=32):
        """
        加速版 PCA 分块量化取整残差计算函数
        支持 GPU 向量化、自动混合精度，无 Python 循环
        ----------------------------------------------------
        Args:
            img: torch.Tensor [B, 3, H, W]
            q:   量化步长
            block: 分块大小 (建议 8,16,32)
        Returns:
            residual: torch.Tensor [B,3,H,W]
            content:  torch.Tensor [B,3,H,W]
        """
    
        assert img.dim() == 4 and img.shape[1] == 3, "输入必须为 [B,3,H,W]"
        B, C, H, W = img.shape
        device = img.device
    
        with torch.amp.autocast('cuda', enabled=True):
            # -------- 1. unfold 分块 (更快的底层C实现) --------
            patches = F.unfold(img, kernel_size=block, stride=block)
            # [B, C*block*block, N]
            Nblocks = patches.shape[-1]
            patches = patches.view(B, C, block*block, Nblocks)
            patches = patches.permute(0, 3, 1, 2).reshape(-1, 3, block*block)  # [B*N,3,Npix]
    
            # -------- 2. 计算PCA基 --------
            mu = patches.mean(dim=2, keepdim=True)
            #mu = 0
            Xc = patches - mu
            Cmat = torch.bmm(Xc, Xc.transpose(1, 2)) / (Xc.shape[2]-1)  # [B*N,3,3]
    
            # -------- 3. 批量特征分解（GPU并行） --------
            eigvals, V = torch.linalg.eigh(Cmat.to(torch.float32))
            idx = torch.argsort(eigvals, dim=1, descending=True)
            V = torch.gather(V, 2, idx.unsqueeze(1).expand(-1, 3, -1))
    
            # -------- 4. 投影 + 量化 + 重建 --------
            Z = torch.bmm(V.transpose(1, 2), Xc)
            Zq = torch.round(Z / q) * q
            Xq = torch.bmm(V, Zq) + mu
    
            # -------- 5. 重新拼回图像 --------
            Xq = Xq.view(B, Nblocks, 3, block*block).permute(0, 2, 3, 1)
            Xq = Xq.reshape(B, 3*block*block, Nblocks)
            Xq = F.fold(Xq, (H, W), kernel_size=block, stride=block)
    
            residual = img - Xq
            content = Xq
    
        return residual, content
    def pid_residual_round(self, img, delta=(1.0, 1.0, 1.0)):
        """
        仅硬量化(round)的 PID 残差
        Args:
            img   : [B,3,H,W]，float；建议像素量纲 0..255（不做归一化）
            delta : 三通道量化步长 (ΔY, ΔU, ΔV)，可标量或三元组
        Returns:
            residual: [B,3,H,W]，img - 重建图（浮点残差，量纲与输入一致）
        """
        B, C, H, W = img.shape
        assert C == 3, "需要 RGB 三通道输入"
        device, dtype = img.device, img.dtype

        # YUV 线性变换（无偏置）
        M = torch.tensor([[0.299000, 0.587000, 0.114000],
                          [-0.168736, -0.331264, 0.500000],
                          [0.500000, -0.418688, -0.081312]], device=device, dtype=dtype)
        Minv = torch.linalg.inv(M)

        # reshape: [B,3,H,W] -> [B, N, 3]
        X = img.permute(0, 2, 3, 1).reshape(B, -1, 3)

        # RGB -> YUV
        YUV = X @ M.T

        # 步长
        if not isinstance(delta, (list, tuple)):
            delta = (float(delta),) * 3
        d = torch.tensor(delta, device=device, dtype=dtype).view(1, 1, 3)

        # 硬量化（取整）
        YUVq = torch.round(YUV / d) * d

        # YUV -> RGB
        Xq = YUVq @ Minv.T
        Xq = Xq.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        # 残差
        residual = img - Xq
        return residual
        
    def pca_block_round_residual_fast_accel3(self, img, q=0.02, block=32):
        """
        加速版 PCA 分块量化取整残差计算函数（自适应量化，避免 NaN 版）
        Args:
            img: torch.Tensor [B, 3, H, W]，建议已归一化到 [0,1]
            q:   相对量化步长（在单位标准差空间里），例如 0.01 ~ 0.03
            block: 分块大小
        Returns:
            residual: torch.Tensor [B,3,H,W]
            content:  torch.Tensor [B,3,H,W]
        """
    
        assert img.dim() == 4 and img.shape[1] == 3, "输入必须为 [B,3,H,W]"
        B, C, H, W = img.shape
        dtype = img.dtype
        device = img.device
    
        # --- 关键：PCA 整段用 float32，避免半精度数值炸掉 ---
        img32 = img.to(torch.float32)
    
        # -------- 1. unfold 分块 --------
        patches = F.unfold(img32, kernel_size=block, stride=block)
        # [B, C*block*block, N]
        Nblocks = patches.shape[-1]
        patches = patches.view(B, C, block * block, Nblocks)
        patches = patches.permute(0, 3, 1, 2).reshape(-1, 3, block * block)  # [M,3,Npix]
        M = patches.shape[0]
    
        # -------- 2. 计算 PCA 基 --------
        mu = patches.mean(dim=2, keepdim=True)          # [M,3,1]
        Xc = patches - mu                               # [M,3,Npix]
        Cmat = torch.bmm(Xc, Xc.transpose(1, 2)) / (Xc.shape[2] - 1)  # [M,3,3]
    
        # 特征分解（float32）
        eigvals, V = torch.linalg.eigh(Cmat)            # [M,3], [M,3,3]
        idx = torch.argsort(eigvals, dim=1, descending=True)
        V = torch.gather(V, 2, idx.unsqueeze(1).expand(-1, 3, -1))  # 排序后的特征向量
        # 可选：量化方向向量
        V = torch.round(V / 0.01) * 0.01
    
        # -------- 3. 投影到 PCA 空间 --------
        Z = torch.bmm(V.transpose(1, 2), Xc)            # [M,3,Npix]  float32
    
        # -------- 4. 基于 Z 的方差自适应量化步长 --------
        eps = 1e-8
        # 先算方差，再 clamp，再 sqrt —— 顺序不能反
        #var = Z.var(dim=2, unbiased=False, keepdim=True)      # [M,3,1]
        #var = torch.clamp(var, min=eps)
        #sigma = var.sqrt()                                    # [M,3,1]
        #print(sigma)
        #exit()
        #sigma_vals = [0.0984, 0.0075, 0.0022]
        sigma_vals = [0.08,0.02,0.01]
        #sigma_vals = [1, 1, 1]
        #sigma_vals = [0.0984, 0.0075, 0.0075]
        sigma = torch.tensor(
              sigma_vals,
              device=Z.device,
              dtype=Z.dtype
          ).view(1, 3, 1)   # [1,3,1]
        sigma_vals2 = [0.2, 0.05, 0.05]
        #sigma_vals = [0.0984, 0.0075, 0.0075]
        sigma2 = torch.tensor(
              sigma_vals2,
              device=Z.device,
              dtype=Z.dtype
          ).view(1, 3, 1)   # [1,3,1]
        # 单位标准差空间
        Z_norm = Z / sigma  *   sigma2                             # [M,3,Npix]
        #print(Z_norm)
        #exit()
        # 在单位标准差空间里用统一步长 q 量化
        Z_norm_q = torch.round(Z_norm / q) * q
    
        # 乘回真实尺度
        Zq = Z_norm_q * sigma  /     sigma2                         # [M,3,Npix]
        #Zq 
        # -------- 5. 重建图像 --------
        Xq = torch.bmm(V, Zq) + mu                            # [M,3,Npix]
    
        Xq = Xq.view(B, Nblocks, 3, block * block).permute(0, 2, 3, 1)
        Xq = Xq.reshape(B, 3 * block * block, Nblocks)
        Xq = F.fold(Xq, (H, W), kernel_size=block, stride=block)  # [B,3,H,W] float32
    
        # 如果后面网络在 autocast 下跑，用原 dtype 喂回去
        Xq = Xq.to(dtype)
        residual = img - Xq
        content = Xq
        
        #print(residual)
        #exit()
        return residual, content
    def pca_block_round_residual_fast_accel_1stRGB_2ndPCA(self, img, q=4.0, block=32):
        """
        保留 PCA 二阶特征（PCA->Z->量化->重建得到 residual/content）
        一阶特征改为 RGB 像素域 q 量化残差（不再使用 Z-space 一阶）
    
        Args:
            img: [B,3,H,W], 像素范围 [0,255]
            q:   像素域量化步长（同样用于 Z-space 量化；如果你想二者分开见下方备注）
            block: 分块大小
    
        Returns:
            residual_pca: [B,3,H,W]  # 二阶（PCA量化重建残差）
            content_pca:  [B,3,H,W]  # 二阶（PCA量化重建内容）
            rgb_residual: [B,3,H,W]  # 一阶（RGB量化残差，推荐给ResNet）
        """
        assert img.dim() == 4 and img.shape[1] == 3
        B, C, H, W = img.shape
        dtype = img.dtype
    
        # =========================
        # (A) 一阶：RGB 像素域量化残差
        # =========================
        img32 = img.to(torch.float32)
        rgb_q = torch.round(img32 / q) * q
        rgb_residual = (img32 - rgb_q).to(dtype)   # [B,3,H,W]
    
        # =========================
        # (B) 二阶：PCA 分块量化重建残差（你原来的主体）
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
    
        # 可选：量化 PCA 基（你的原逻辑）
        V = torch.round(V / 0.01) * 0.01
    
        # -------- 3. 投影 --------
        Z = torch.bmm(V.transpose(1, 2), Xc)  # [M,3,Npix]
    
        # -------- 4. Z-space 归一化 + 量化（保持你原来的二阶定义）--------
        sigma = torch.tensor([0.08, 0.02, 0.01], device=Z.device, dtype=Z.dtype).view(1,3,1)
        sigma2 = torch.tensor([0.2, 0.05, 0.05], device=Z.device, dtype=Z.dtype).view(1,3,1)
    
        Z_norm = Z / sigma * sigma2
        Z_norm_q = torch.round(Z_norm / q) * q
        Zq = Z_norm_q * sigma / sigma2
    
        # -------- 5. 重建 --------
        Xq = torch.bmm(V, Zq) + mu
    
        Xq = Xq.view(B, Nblocks, 3, block * block).permute(0, 2, 3, 1)
        Xq = Xq.reshape(B, 3 * block * block, Nblocks)
        Xq = F.fold(Xq, (H, W), kernel_size=block, stride=block).to(dtype)
    
        #Xq = torch.clamp(Xq, 0.0, 255.0).to(dtype)
        residual_pca = img - Xq
        content_pca = Xq
    
        return residual_pca, content_pca, rgb_residual
    def entropy_from_logits(self, logits):
        p = torch.softmax(logits, dim=1)
        return -(p * torch.log(p + 1e-6)).sum(dim=1)  # [B]
    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
    def forward(self, x):
        # n,c,w,h = x.shape
        # if -1*w%2 != 0: x = x[:,:,:w%2*-1,:      ]
        # if -1*h%2 != 0: x = x[:,:,:      ,:h%2*-1]
        # factor = 0.5
        # x_half = F.interpolate(x, scale_factor=factor, mode='nearest', recompute_scale_factor=True)
        # x_re   = F.interpolate(x_half, scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
        # NPR  = x - x_re
        # n,c,w,h = x.shape
        # if w%2 == 1 : x = x[:,:,:-1,:]
        # if h%2 == 1 : x = x[:,:,:,:-1]
        #NPR  = x - self.interpolate(x, 0.5)
        q = 0.006 * 4
        input = x
        #NPR,n1 = self.pca_block_round_residual_fast_accel3(input, q, block=4)
        NPR, n1, Zone = self.pca_block_round_residual_fast_accel_1stRGB_2ndPCA(input, q, block=4)
        #为0.1的时候效果不好
        #为0.01时效果明显变好，但是stargan的效果又很差,最大能到(8 Mean      ) acc: 93.5; ap: 96.5
        #试了当X255时，pca4和pca8相比，效果都不好，但是pca4要效果更好一些
        #必须要进行【0-1】归一化，才能保证适用性，以上没有标准化
        #print(NPR)
        #exit()
        #print(NPR*5)
        #exit()
        #NPR = self.pid_residual_round(x,( 0.0175, 0.0175, 0.0175))
        #x = self.conv1(NPR*197)
        x1 = self.conv1(NPR*3.464/q)
        #x1 = self.conv1(NPR)
        #x = self.conv1(NPR)
        #print(NPR*3.464/q)
        #exit()
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.layer1(x1)
        x1 = self.layer2(x1)

        x1 = self.avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
       
        NPR2 = self.pid_residual_round(input,( 0.004, 0.004, 0.004))
        #NPR2 = self.pid_residual_round(input*255,(1, 1, 1))
        #x = self.conv1(NPR*197)
        x2 = self.conv1_2(NPR2*3.464/0.004)
        #x2 = self.conv1_2(NPR2)
        #NPR2 =  Zone
        #x2 = self.conv1_2(NPR2*3.464/q)
        #x2 = self.conv1_2(NPR2)
        #x2 = self.conv1_2(NPR2)
        #x = self.conv1(NPR)
        #print(NPR*3.464/q)
        #exit()
        x2 = self.bn1_2(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        x2 = self.layer1_2(x2)
        x2 = self.layer2_2(x2)

        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        
       
        
        out1 = self.fc1(x1)
        out2 = self.fc1_2(x2)
        
        #x1_fuse = x1.detach()
        #x2_fuse = x2.detach()
        #feat = torch.cat([x1_fuse, x2_fuse], dim=1)     # [B, C1+C2]
        #g = torch.sigmoid(self.gate(feat))    # [B,1]
        
        #out = g * out1.detach() + (1 - g) * out2.detach()       # [B, num_classes]
        #output = torch.cat([out1, out2, out, g], dim=1)
        
        #H1 = self.entropy_from_logits(out1)  # out1
        #H2 = self.entropy_from_logits(out2)  # out2
        #delta = 0.0  # 可调：0~0.05，越大越偏向 PID
        #use_pca = (H1 + delta < H2)        # [B] True/False
        
        #out = torch.where(use_pca[:,None], out1, out2)
        #output = torch.cat([out1, out2, out, g], dim=1)
        
        
        feat = torch.cat([x1.detach(), x2.detach()], dim=1)   # [B,1024]
        g = torch.sigmoid(self.gate(feat))                    # [B,1] in (0,1)
        
        # 融合在 logit 层：更像“真正综合”，也更不容易被尺度带偏
        out = g * out1.detach() + (1 - g) * out2.detach()     # [B,1]
        
        # 可选：safeguard（强烈推荐），避免 GAN 时被 PID 拉崩
        # 用 PID 的置信度（margin/entropy 等价，这里用 p 距离 0.5）
        '''
        p1 = torch.sigmoid(out1.detach())
        conf1 = (p1 - 0.5).abs()   # 越大越自信
        p2 = torch.sigmoid(out2.detach())
        conf2 = (p2 - 0.5).abs()   # 越大越自信
        thr = 0.45                 # 需要你在 SD 验证集上调：0.10~0.25
        use_pca = (conf2 < thr)    # PID 不自信 → 用 PCA
        #out = torch.where(use_pca, out1.detach(), out)
        '''
        c1 = out1.abs()
        c2 = out2.abs()
        
        gap = (c2 - c1).abs()
        hard = gap > 15   # T 用 SD_val 统计分位数定，比如 80%分位
        
        out_hard = torch.where(c2 > c1, out2, out1)   # 谁更自信用谁
        #out_soft = self.fc_fuse(torch.cat([x1.detach(), x2.detach()], 1))  # 只在尴尬区用
        x = torch.cat([x1.detach(), x2.detach()], dim=1)   # [B, 2C] 
        #print(x)
        x = self.fc_comb(x)
        
        #delta = -13.0   # 基于你 BigGAN 的统计，这是非常合理的
        delta = 0
        use_pid =  (c2 > c1 + delta)
        out = torch.where(use_pid, out2.detach() , out1.detach() )
        output = torch.cat([out1, out2, out, g], dim=1)	
        
        #print("PCA logit mean/std:", out1.mean().item(), out1.std().item())
        #print("PID logit mean/std:", out2.mean().item(), out2.std().item())
        #print("PCA conf mean/std:", conf1.mean().item(), conf1.std().item())
        #print("PID conf mean/std:", conf2.mean().item(), conf2.std().item())
        #x = self.fc1(x2)
        return output


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
