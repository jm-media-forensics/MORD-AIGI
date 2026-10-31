# -*- coding: utf-8 -*-
"""Train one MORD branch without degradation augmentation.

Settings of Table 2: UFD / Self-Synthesis (ProGAN, 4 classes) and GenImage (SDv1.4 subset).
Training images: tile if small, random crop 224, horizontal flip.

  python train_clean.py --branch bcpr --seed 1 --name progan_bcpr_s1 \
      --dataroot /data/progan --classes car,cat,chair,horse --lr 0.0004 --niter 50
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from data import clean_train_loader
from mord import SingleBranch, save_branch, seed_all


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--branch", choices=("bcpr", "pid"), required=True)
    pa.add_argument("--seed", type=int, required=True)
    pa.add_argument("--name", required=True)
    pa.add_argument("--dataroot", required=True, help="folder that contains train/<class>/{0_real,1_fake}")
    pa.add_argument("--classes", default="", help="comma-separated class folders under train/")
    pa.add_argument("--checkpoints_dir", default="./checkpoints")
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--lr", type=float, default=0.0004)
    pa.add_argument("--niter", type=int, default=50)
    pa.add_argument("--delr_freq", type=int, default=10, help="lr x0.9 every delr_freq epochs")
    pa.add_argument("--crop", type=int, default=224)
    pa.add_argument("--num_workers", type=int, default=8)
    pa.add_argument("--gpu", type=int, default=0)
    a = pa.parse_args()

    seed_all(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(a.checkpoints_dir, a.name)
    os.makedirs(save_dir, exist_ok=True)
    classes = [c for c in a.classes.split(",") if c] or None
    loader = clean_train_loader(os.path.join(a.dataroot, "train"), classes, a.batch_size, a.crop, a.num_workers)
    model = SingleBranch(a.branch).to(dev).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999))
    bce = nn.BCEWithLogitsLoss()

    with open(os.path.join(save_dir, "log.txt"), "a", encoding="utf-8") as log:
        def say(msg):
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        say(f"branch={a.branch} seed={a.seed} images={len(loader.dataset)} batches/epoch={len(loader)}")
        # epoch indices -1 .. niter-1, as in the training runs reported in the paper
        for epoch in range(-1, a.niter):
            start, losses = time.time(), []
            for x, y in loader:
                x, y = x.to(dev), y.to(dev).float()
                loss = bce(model(x)[:, 0], y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if epoch % a.delr_freq == 0 and epoch != 0:
                for group in optimizer.param_groups:
                    group["lr"] *= 0.9
            say(f"epoch={epoch} loss={np.mean(losses):.5f} lr={optimizer.param_groups[0]['lr']:.6g} "
                f"seconds={time.time() - start:.0f}")

        save_branch(model, os.path.join(save_dir, "model_last.pth"), seed=a.seed, epoch=a.niter - 1)
        json.dump({"branch": a.branch, "seed": a.seed, "train_images": len(loader.dataset)},
                  open(os.path.join(save_dir, "done.json"), "w"))
        say(f"saved {os.path.join(save_dir, 'model_last.pth')}")


if __name__ == "__main__":
    main()
