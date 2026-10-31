# -*- coding: utf-8 -*-
"""Test BCPR, PiD and MORD (probability averaging of the two branches).

  --protocol clean   one result per subset folder (UFD, Self-Synthesis, GenImage);
                     tile if small, center crop 256, no resize; mAcc / mAP = unweighted mean
  --protocol wildrf  as clean, but the whole image is resized to 256x256 (WildRF platforms)
  --protocol robust  post-processing evaluation: the first --cap images per class of every
                     generator folder are pooled; one result per condition, plus the mean
                     over the conditions other than clean; resize to 256x256

  python test.py --protocol clean --root /data/ufd --subsets progan,cyclegan,biggan \
      --bcpr checkpoints/progan_bcpr_s1/model_last.pth --pid checkpoints/progan_pid_s1/model_last.pth
  python test.py --protocol robust --root /data/genimage/test --cap 800 \
      --bcpr checkpoints/bcpr_mild_s1/model_last.pth --pid checkpoints/pid_mild_s1/model_last.pth
  python test.py --protocol robust --root /data/format_matched/test --conditions resize_0.5+jpeg_75,blur_1.0+jpeg_75 ...
"""
import argparse
import csv
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from data import TestDataset, gather
from mord import load_branch

CONDITIONS = ["clean", "jpeg_95", "jpeg_85", "jpeg_75", "blur_0.5", "blur_1.0", "blur_2.0",
              "noise_0.01", "noise_0.02", "noise_0.05", "resize_0.5", "resize_0.7"]


def sig(x):
    return 1 / (1 + np.exp(-x.astype(np.float64)))


@torch.no_grad()
def run(models, items, protocol, condition, a, dev):
    ld = DataLoader(TestDataset(items, protocol, condition), batch_size=a.batch_size, num_workers=a.num_workers)
    LB, LP, Y = [], [], []
    for x, y in ld:
        x = x.to(dev)
        LB.append(models[0](x)[:, 0].cpu().numpy())
        LP.append(models[1](x)[:, 0].cpu().numpy())
        Y.append(y.numpy())
    lb, lp, y = np.concatenate(LB), np.concatenate(LP), np.concatenate(Y).astype(int)
    out = {}
    # at threshold 0.5, probability averaging decides like lb + lp > 0
    for name, pred, prob in (("BCPR", lb > 0, sig(lb)), ("PiD", lp > 0, sig(lp)),
                             ("MORD", lb + lp > 0, 0.5 * sig(lb) + 0.5 * sig(lp))):
        ap = average_precision_score(y, prob) if 0 < y.sum() < len(y) else np.nan
        out[name] = (np.mean(pred.astype(int) == y) * 100, ap * 100)
    return out


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--bcpr", required=True, help="BCPR branch checkpoint")
    pa.add_argument("--pid", required=True, help="PiD branch checkpoint")
    pa.add_argument("--protocol", choices=("clean", "wildrf", "robust"), required=True)
    pa.add_argument("--root", required=True)
    pa.add_argument("--subsets", default="", help="comma-separated sub-folders of --root (default: all)")
    pa.add_argument("--conditions", default=",".join(CONDITIONS), help="robust protocol only")
    pa.add_argument("--cap", type=int, default=None, help="images per class per folder (robust: 800)")
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--num_workers", type=int, default=8)
    pa.add_argument("--out", default=None, help="optional CSV file")
    a = pa.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = (load_branch(a.bcpr, "bcpr", dev), load_branch(a.pid, "pid", dev))
    subsets = [s for s in a.subsets.split(",") if s] or \
        sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)))

    rows = []
    if a.protocol == "robust":
        items = [it for s in subsets for it in gather(os.path.join(a.root, s), a.cap)]
        print(f"pool: {len(items)} images from {len(subsets)} folders")
        for cond in a.conditions.split(","):
            rows.append((cond, run(models, items, "robust", cond, a, dev)))
            print(cond, rows[-1][1], flush=True)
        perturbed = [r for c, r in rows if c != "clean"]
        if perturbed:
            rows.append((f"mean of {len(perturbed)} perturbed",
                         {k: tuple(np.mean([r[k][i] for r in perturbed]) for i in (0, 1)) for k in perturbed[0]}))
    else:
        for s in subsets:
            rows.append((s, run(models, gather(os.path.join(a.root, s), a.cap), a.protocol, "clean", a, dev)))
            print(s, rows[-1][1], flush=True)
        rows.append(("mean", {k: tuple(np.mean([r[k][i] for _, r in rows]) for i in (0, 1)) for k in rows[0][1]}))

    keys = ("BCPR", "PiD", "MORD")
    header = ["", *[f"{k} {m}" for k in keys for m in ("Acc", "AP")]]
    table = [[name, *[f"{r[k][i]:.2f}" for k in keys for i in (0, 1)]] for name, r in rows]
    width = max(len(t[0]) for t in table + [header])
    print("\n" + "  ".join([header[0].ljust(width), *[h.rjust(9) for h in header[1:]]]))
    for t in table:
        print("  ".join([t[0].ljust(width), *[v.rjust(9) for v in t[1:]]]))
    if a.out:
        with open(a.out, "w", newline="") as f:
            csv.writer(f).writerows([header, *table])


if __name__ == "__main__":
    main()
