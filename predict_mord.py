# -*- coding: utf-8 -*-
"""
predict_mord.py
Run MORD inference on a single image OR every image under a folder
(recursively). No 0_real/1_fake label structure required.

Can also evaluate robustness: apply one or more in-the-wild perturbations
(JPEG / blur / noise / resize / WebP) to every input and report how the
prediction changes, vs. the clean image.

Reuses the exact model-loading, preprocessing and perturbation operators
of eval_mord_all.py so results match the evaluation pipeline
(Resize/CenterCrop handling identical, no normalization).

Usage:
  # clean only, single image
  python predict_mord.py --model_path ckpt.pth --input img.jpg

  # clean + all perturbations on one image
  python predict_mord.py --model_path ckpt.pth --input img.jpg --perturbations all

  # specific perturbations on a folder, save CSV
  python predict_mord.py --model_path ckpt.pth --input ./folder \
        --perturbations clean,jpeg_75,blur_2.0,noise_0.05 --output_csv preds.csv
"""
import os
import csv
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Reuse the evaluation pipeline's helpers (same dir).
from eval_mord_all import (
    load_model, translate_duplicate, sigmoid, fusion_scores,
    apply_perturbation, PERTURBATIONS, IMG_EXT,
)


# ---------------------------------------------------------------------------
# Collect image paths
# ---------------------------------------------------------------------------
def gather_images(path):
    if os.path.isfile(path):
        if os.path.splitext(path)[1].lower() in IMG_EXT:
            return [path]
        raise ValueError(f"Not a supported image file: {path}")
    if os.path.isdir(path):
        out = []
        for dp, _dirs, files in os.walk(path):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in IMG_EXT:
                    out.append(os.path.join(dp, f))
        return sorted(out)
    raise FileNotFoundError(f"Path not found: {path}")


# ---------------------------------------------------------------------------
# Dataset that yields (tensor, idx) — no labels, with optional perturbation
# ---------------------------------------------------------------------------
class ImageListDataset(Dataset):
    def __init__(self, paths, perturbation="clean", crop=256, resize=False):
        self.paths = paths
        self.perturbation = perturbation
        steps = [T.Lambda(lambda img: translate_duplicate(img, crop))]
        if resize:
            steps.append(T.Resize((crop, crop)))   # optional, to match a resize pipeline
        steps += [T.CenterCrop(crop), T.ToTensor()]   # no Normalize
        self.tf = T.Compose(steps)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
            img = apply_perturbation(img, self.perturbation)
        except Exception:
            img = Image.new("RGB", (256, 256))
        return self.tf(img), idx


# ---------------------------------------------------------------------------
@torch.no_grad()
def run_perturbation(model, paths, pert, device, opt):
    ds = ImageListDataset(paths, perturbation=pert, crop=opt.crop, resize=opt.resize)
    loader = DataLoader(ds, batch_size=opt.batch_size, num_workers=opt.num_workers,
                        shuffle=False, pin_memory=True)
    l0s, l1s, idxs = [], [], []
    for imgs, idx in loader:
        out = model(imgs.to(device, non_blocking=True))
        l0s.append(out[:, 0].cpu().numpy())
        l1s.append(out[:, 1].cpu().numpy())
        idxs.append(idx.numpy())
    l0 = np.concatenate(l0s); l1 = np.concatenate(l1s); order = np.concatenate(idxs)
    # restore original order
    inv = np.argsort(order)
    return l0[inv], l1[inv]


# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_path", required=True)
    pa.add_argument("--input", required=True,
                    help="a single image file OR a folder (searched recursively)")
    pa.add_argument("--perturbations", default="clean",
                    help="'clean', 'all', or a comma list e.g. "
                         "'clean,jpeg_75,blur_2.0,noise_0.05'. "
                         f"Available: {','.join(PERTURBATIONS)}")
    pa.add_argument("--fusion", default="Routing",
                    choices=["BCPR", "PiD", "NP", "BMA", "Routing"],
                    help="which score to use for the final decision")
    pa.add_argument("--threshold", type=float, default=0.5,
                    help="decision threshold on the fused probability (fake if >thr)")
    pa.add_argument("--crop", type=int, default=256)
    pa.add_argument("--resize", action="store_true",
                    help="add Resize((crop,crop)) before CenterCrop "
                         "(to match a resize-based training pipeline)")
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--num_workers", type=int, default=4)
    pa.add_argument("--model_impl", default="auto",
                    choices=["auto", "current"])
    pa.add_argument("--output_csv", default="",
                    help="optional path to write per-image, per-perturbation predictions")
    pa.add_argument("--quiet", action="store_true",
                    help="suppress per-image lines (single-image mode always prints)")
    opt = pa.parse_args()

    # resolve perturbation list
    if opt.perturbations.strip().lower() == "all":
        perts = list(PERTURBATIONS)
    else:
        perts = [p.strip() for p in opt.perturbations.split(",") if p.strip()]
    bad = [p for p in perts if p != "clean" and p not in PERTURBATIONS]
    if bad:
        raise ValueError(f"unknown perturbation(s): {bad}. Available: {PERTURBATIONS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {opt.model_path}")
    model = load_model(opt.model_path, device, model_impl=opt.model_impl)
    model.eval()

    paths = gather_images(opt.input)
    print(f"Found {len(paths)} image(s) under: {opt.input}")
    print(f"Perturbations: {perts}")
    print(f"Resize-before-crop: {opt.resize} | crop: {opt.crop} | fusion: {opt.fusion}\n")
    if not paths:
        return

    single = len(paths) == 1
    all_rows = []
    summary = []  # (pert, n, n_fake, fake_pct, mean_p)

    for pert in perts:
        l0, l1 = run_perturbation(model, paths, pert, device, opt)
        sc = fusion_scores(l0, l1)
        fused = sc[opt.fusion]
        dec = (fused > opt.threshold).astype(int)

        for i in range(len(paths)):
            all_rows.append({
                "path": paths[i],
                "perturbation": pert,
                "p_BCPR": float(sc["BCPR"][i]),
                "p_PiD": float(sc["PiD"][i]),
                "p_NP": float(sc["NP"][i]),
                "p_BMA": float(sc["BMA"][i]),
                "p_Routing": float(sc["Routing"][i]),
                f"p_fused({opt.fusion})": float(fused[i]),
                "pred": "FAKE" if dec[i] == 1 else "REAL",
            })

        n_fake = int(dec.sum())
        summary.append((pert, len(paths), n_fake,
                        100.0 * n_fake / len(paths), float(fused.mean())))

    # ---- single image: show per-perturbation table -----------------------
    if single:
        print(f"{'perturbation':>14}  {'pred':>5}  {'p_'+opt.fusion:>12}  "
              f"{'p_BCPR':>8}  {'p_PiD':>8}")
        print("-" * 60)
        for r in all_rows:
            print(f"{r['perturbation']:>14}  {r['pred']:>5}  "
                  f"{r[f'p_fused({opt.fusion})']:>12.4f}  "
                  f"{r['p_BCPR']:>8.4f}  {r['p_PiD']:>8.4f}")
    # ---- folder: per-perturbation aggregate -------------------------------
    else:
        if not opt.quiet:
            print(f"{'perturbation':>14}  {'N':>6}  {'#FAKE':>6}  "
                  f"{'FAKE%':>7}  {'mean p':>8}")
            print("-" * 52)
            for pert, n, nf, pct, mp in summary:
                print(f"{pert:>14}  {n:>6}  {nf:>6}  {pct:>6.1f}%  {mp:>8.4f}")

    print("\n" + "=" * 52)
    print(f"  Images: {len(paths)} | fusion: {opt.fusion} | threshold: {opt.threshold}")
    print("=" * 52)

    if opt.output_csv:
        with open(opt.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nPer-image predictions written to: {opt.output_csv}")


if __name__ == "__main__":
    main()
