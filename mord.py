# -*- coding: utf-8 -*-
"""MORD branches.

Each branch (BCPR or PiD) is a separate model with its own checkpoint. The sub-modules
are taken from a freshly initialised two-branch ResNet (networks/resnet.py), so the
initialisation and the parameter names are those of the two-branch model.

MORD = probability averaging of the two branch outputs.
"""
import torch
import torch.nn as nn

from networks.resnet import ResNet, resnet50

BRANCH_MODULES = {
    "bcpr": ("conv1", "bn1", "layer1", "layer2", "fc1"),
    "pid": ("conv1_2", "bn1_2", "layer1_2", "layer2_2", "fc1_2"),
}
Q_BCPR = 0.006 * 4      # BCPR coefficient step q = 0.024
BLOCK_BCPR = 4          # BCPR block size k
Q_PID = 0.004           # PiD quantization step


class SingleBranch(nn.Module):
    def __init__(self, branch):
        super().__init__()
        assert branch in BRANCH_MODULES, branch
        self.branch = branch
        full = resnet50(num_classes=1)
        for name in BRANCH_MODULES[branch]:
            setattr(self, name, getattr(full, name))
        self.relu, self.maxpool, self.avgpool = full.relu, full.maxpool, full.avgpool

    def forward(self, x):
        """Returns [B, 1] logits. Input: RGB in [0, 1], no normalization."""
        if self.branch == "bcpr":
            r, _, _ = ResNet.pca_block_round_residual_fast_accel_1stRGB_2ndPCA(None, x, Q_BCPR, block=BLOCK_BCPR)
            h = self.conv1(r * 3.464 / Q_BCPR)
            h = self.maxpool(self.relu(self.bn1(h)))
            h = self.layer2(self.layer1(h))
            return self.fc1(self.avgpool(h).view(h.size(0), -1))
        r = ResNet.pid_residual_round(None, x, (Q_PID, Q_PID, Q_PID))
        h = self.conv1_2(r * 3.464 / Q_PID)
        h = self.maxpool(self.relu(self.bn1_2(h)))
        h = self.layer2_2(self.layer1_2(h))
        return self.fc1_2(self.avgpool(h).view(h.size(0), -1))


def save_branch(model, path, **meta):
    torch.save({"model": model.state_dict(), "branch": model.branch, **meta}, path)


def load_branch(path, branch, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    prefixes = tuple(n + "." for n in BRANCH_MODULES[branch])
    m = SingleBranch(branch)
    m.load_state_dict({k: v for k, v in sd.items() if k.startswith(prefixes)}, strict=True)
    return m.to(device).eval()


def seed_all(seed, deterministic=True):
    import os
    import random
    import numpy as np
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
