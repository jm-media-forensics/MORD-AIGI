# -*- coding: utf-8 -*-
"""Train one MORD branch with paired degradation augmentation.

Settings of Table 2: post-processing robustness (--aug_preset mild) and the format-matched
control (--aug_preset aggressive, data from make_format_matched.py).

Every training image gives a clean view and an augmented view with the same resize to
256x256, random crop 224 and flip. Loss: BCE(clean) + 0.5 * BCE(augmented).
Adam, gradient-norm clipping 1.0, lr x0.9 every 10 epochs; the final weights are used.

  python train_robust.py --branch pid --seed 1 --name pid_mild_s1 \
      --dataroot /data/genimage_sdv14 --classes stable_diffusion_v_1_4 --aug_preset mild \
      --val_dir /data/genimage_sdv14/val/stable_diffusion_v_1_4
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from data import Augmentor, TestDataset, gather, paired_train_loader
from mord import SingleBranch, save_branch, seed_all


@torch.no_grad()
def val_acc(model, loader, dev):
    model.eval()
    ok = n = 0
    for x, y in loader:
        ok += ((model(x.to(dev))[:, 0].cpu() > 0).long() == y).sum().item()
        n += len(y)
    model.train()
    return ok / max(n, 1)


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--branch", choices=("bcpr", "pid"), required=True)
    pa.add_argument("--seed", type=int, required=True)
    pa.add_argument("--name", required=True)
    pa.add_argument("--dataroot", required=True, help="folder that contains train/<class>/{0_real,1_fake}")
    pa.add_argument("--classes", default="", help="comma-separated class folders under train/")
    pa.add_argument("--aug_preset", choices=("mild", "aggressive"), required=True)
    pa.add_argument("--aug_weight", type=float, default=0.5)
    pa.add_argument("--val_dir", default=None, help="validation accuracy is logged only, never used for selection")
    pa.add_argument("--checkpoints_dir", default="./checkpoints")
    pa.add_argument("--batch_size", type=int, default=64, help="pairs per mini-batch")
    pa.add_argument("--lr", type=float, default=0.0004)
    pa.add_argument("--niter", type=int, default=50)
    pa.add_argument("--delr_freq", type=int, default=10)
    pa.add_argument("--crop", type=int, default=224)
    pa.add_argument("--save_every", type=int, default=10)
    pa.add_argument("--num_workers", type=int, default=8)
    pa.add_argument("--gpu", type=int, default=0)
    a = pa.parse_args()

    seed_all(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(a.checkpoints_dir, a.name)
    os.makedirs(save_dir, exist_ok=True)
    log = open(os.path.join(save_dir, "log.txt"), "a")

    def say(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    augmentor = Augmentor(a.aug_preset)
    classes = [c for c in a.classes.split(",") if c] or None
    loader = paired_train_loader(os.path.join(a.dataroot, "train"), classes, augmentor,
                                 a.batch_size, a.crop, a.num_workers)
    val_loader = None
    if a.val_dir:
        val_loader = torch.utils.data.DataLoader(TestDataset(gather(a.val_dir), "robust"),
                                                 batch_size=128, num_workers=4)
    say(f"branch={a.branch} seed={a.seed} {augmentor} images={len(loader.dataset)} batches/epoch={len(loader)}")

    model = SingleBranch(a.branch).to(dev).train()
    optim = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999))
    bce = nn.BCEWithLogitsLoss()

    def save(tag, epoch):
        save_branch(model, os.path.join(save_dir, f"model_{tag}.pth"), seed=a.seed, epoch=epoch)

    # epoch indices -1 .. niter-1, as in the training runs reported in the paper
    for epoch in range(-1, a.niter):
        t0, losses = time.time(), []
        for x_clean, x_aug, y in loader:
            x_clean, x_aug, y = x_clean.to(dev), x_aug.to(dev), y.to(dev).float()
            real, fake = y < 0.5, y >= 0.5
            optim.zero_grad()
            loss = bce(model(x_clean)[:, 0], y)
            out_aug = model(x_aug)[:, 0]
            if real.any():
                loss = loss + a.aug_weight * bce(out_aug[real], y[real])
            if fake.any():
                loss = loss + a.aug_weight * bce(out_aug[fake], y[fake])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            losses.append(loss.item())
        if epoch % a.delr_freq == 0 and epoch != 0:
            for g in optim.param_groups:
                g["lr"] *= 0.9
        msg = f"epoch {epoch:3d} loss {np.mean(losses):.4f} lr {optim.param_groups[0]['lr']:.2e} {time.time() - t0:.0f}s"
        if val_loader is not None and (epoch % 5 == 4 or epoch == a.niter - 1):
            msg += f" val_acc {val_acc(model, val_loader, dev) * 100:.2f}"
        say(msg)
        if a.save_every and epoch >= 0 and epoch % a.save_every == a.save_every - 1:
            save(f"epoch_{epoch}", epoch)
    save("last", a.niter - 1)
    json.dump({"branch": a.branch, "seed": a.seed}, open(os.path.join(save_dir, "done.json"), "w"))
    say(f"saved {os.path.join(save_dir, 'model_last.pth')}")


if __name__ == "__main__":
    main()
