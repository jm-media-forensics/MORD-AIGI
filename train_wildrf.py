# -*- coding: utf-8 -*-
"""WildRF setting of Table 2.

Every image is resized to 256x256 for training, validation and test. Training uses a
random crop 224 and flip; validation and test use the whole 256x256 image.

  cache   resize every train / val / test image once and store it as lossless PNG
  train   train one branch: Adam lr 4e-4, batch 64, lr x0.9 every 10 epochs, 100 epochs,
          a checkpoint after every epoch
  select  validation Acc / AP of BCPR, PiD and MORD for every epoch; the checkpoint with
          the best MORD AP is used

  python train_wildrf.py cache --src /data/WildRF --cache /data/wildrf256
  python train_wildrf.py train --cache /data/wildrf256 --branch bcpr --seed 1 --out checkpoints/wildrf_bcpr_s1
  python train_wildrf.py train --cache /data/wildrf256 --branch pid  --seed 1 --out checkpoints/wildrf_pid_s1
  python train_wildrf.py select --cache /data/wildrf256 --bcpr checkpoints/wildrf_bcpr_s1 --pid checkpoints/wildrf_pid_s1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from mord import SingleBranch, load_branch, save_branch, seed_all

SPLITS = ["train/wild", "val", "test/facebook", "test/reddit", "test/twitter"]
RESIZE = T.Resize((256, 256))


def items(root):
    return [(os.path.join(root, c, f), l) for l, c in ((0, "0_real"), (1, "1_fake"))
            for f in sorted(os.listdir(os.path.join(root, c)))]


def cache(a):
    n = 0
    for s in SPLITS:
        for p, _ in items(os.path.join(a.src, s)):
            q = os.path.join(a.cache, s, os.path.basename(os.path.dirname(p)),
                             os.path.splitext(os.path.basename(p))[0] + ".png")
            if os.path.exists(q):
                continue
            os.makedirs(os.path.dirname(q), exist_ok=True)
            RESIZE(Image.open(p).convert("RGB")).save(q)
            n += 1
    print(f"cached {n} images to {a.cache}")


class DS(Dataset):
    def __init__(self, root, train):
        self.items = items(root)
        self.tf = T.Compose([T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor()]) if train else T.ToTensor()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, l = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), l


def train(a):
    seed_all(a.seed, deterministic=False)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.txt"), "a")
    dl = DataLoader(DS(os.path.join(a.cache, "train/wild"), True), batch_size=64, shuffle=True, num_workers=8)
    dev = "cuda"
    m = SingleBranch(a.branch).to(dev).train()
    opt = torch.optim.Adam(m.parameters(), lr=a.lr, betas=(0.9, 0.999))
    bce = nn.BCEWithLogitsLoss()
    for ep in range(a.epochs):
        t0, ls = time.time(), []
        for x, y in dl:
            loss = bce(m(x.to(dev))[:, 0], y.to(dev).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            ls.append(loss.item())
        if ep % 10 == 0 and ep != 0:
            for g in opt.param_groups:
                g["lr"] *= 0.9
        save_branch(m, os.path.join(a.out, f"model_epoch_{ep}.pth"), seed=a.seed, epoch=ep)
        line = f"{time.strftime('%F %T')} epoch {ep} loss {np.mean(ls):.4f} {time.time() - t0:.0f}s"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
    json.dump({"branch": a.branch, "seed": a.seed, "epochs": a.epochs, "lr": a.lr},
              open(os.path.join(a.out, "done.json"), "w"))


@torch.no_grad()
def logits(path, branch, X):
    m = load_branch(path, branch, "cuda")
    return torch.cat([m(x.cuda())[:, 0].cpu() for x in X]).numpy()


def select(a):
    X, Y = [], []
    for x, y in DataLoader(DS(os.path.join(a.cache, "val"), False), batch_size=64, num_workers=4):
        X.append(x)
        Y.append(y)
    y = torch.cat(Y).numpy()
    sig = lambda z: 1 / (1 + np.exp(-z))
    rows = []
    for ep in range(a.epochs):
        lb = logits(os.path.join(a.bcpr, f"model_epoch_{ep}.pth"), "bcpr", X)
        lp = logits(os.path.join(a.pid, f"model_epoch_{ep}.pth"), "pid", X)
        pm = 0.5 * sig(lb) + 0.5 * sig(lp)
        rows.append(dict(ep=ep, bcpr=np.mean((lb > 0) == y), pid=np.mean((lp > 0) == y),
                         mord=np.mean((pm > 0.5) == y), mord_ap=average_precision_score(y, pm)))
    for r in rows:
        print(f"epoch {r['ep']:3d}  val Acc BCPR/PiD/MORD {r['bcpr']*100:.2f} {r['pid']*100:.2f} {r['mord']*100:.2f}"
              f"  MORD AP {r['mord_ap']*100:.2f}")
    best = max(rows, key=lambda r: r["mord_ap"])
    print(f"SELECTED (best val MORD AP): epoch {best['ep']}  AP {best['mord_ap']*100:.2f}  Acc {best['mord']*100:.2f}")


if __name__ == "__main__":
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("cmd", choices=["cache", "train", "select"])
    pa.add_argument("--src", help="WildRF root with train/wild, val, test/<platform>")
    pa.add_argument("--cache", required=True, help="folder of the resized copies")
    pa.add_argument("--branch", choices=["bcpr", "pid"])
    pa.add_argument("--seed", type=int, default=1)
    pa.add_argument("--out")
    pa.add_argument("--bcpr")
    pa.add_argument("--pid")
    pa.add_argument("--lr", type=float, default=4e-4)
    pa.add_argument("--epochs", type=int, default=100)
    args = pa.parse_args()
    {"cache": cache, "train": train, "select": select}[args.cmd](args)
