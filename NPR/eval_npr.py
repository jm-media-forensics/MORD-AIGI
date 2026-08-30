# -*- coding: utf-8 -*-
"""
eval_npr.py
Full robustness evaluation of the trained NPR detector.

Applies clean + a suite of in-the-wild perturbations (JPEG / Gaussian blur /
Gaussian noise / resize / WebP) to a labeled test set, computes pooled
Acc / AP / AUC for the NPR model, and writes a Markdown report.

Test set: --test_dir may contain real/fake (or 0_real/1_fake) directly, OR
generator subfolders each with real/fake (images are pooled across all
generators, matching the paper's "mean over all generators" protocol).

Preprocessing (matches NPR training's Resize framing):
    perturbation -> Resize(256,256) -> ToTensor   (no normalization)
Use --crop_mode centercrop to test native-pixel center crops instead.

Usage:
  python eval_npr.py --model_path ./checkpoints/npr_sdv4_paired_mild/model_epoch_last.pth \
    --test_dir /home/ubuntu/workspace/NPR-DeepfakeDetection-main2/datasets/dataset_out/test \
    --output_md npr_robustness.md
"""
import os
import io
import gc
import math
import argparse
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile, ImageFilter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

from networks.resnet import resnet50   # NPR model

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REAL_NAMES = {"real", "0", "0_real", "nature"}
FAKE_NAMES = {"fake", "1", "1_fake", "ai"}

PERTURBATIONS = [
    "clean",
    "jpeg_95", "jpeg_85", "jpeg_75",
    "blur_0.5", "blur_1.0", "blur_2.0",
    "noise_0.01", "noise_0.02", "noise_0.05",
    "resize_0.5", "resize_0.7",
    "webp_95", "webp_85",
]


# --- perturbation operators -------------------------------------------------
def _jpeg(img, q):
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="JPEG", quality=int(q))
    buf.seek(0); return Image.open(buf).convert("RGB")

def _blur(img, s):
    return img.filter(ImageFilter.GaussianBlur(radius=float(s)))

def _resize(img, f):
    w, h = img.size
    nw, nh = max(1, round(w*f)), max(1, round(h*f))
    return img.resize((nw, nh), Image.BICUBIC).resize((w, h), Image.BICUBIC)

def _noise(img, s):
    a = np.asarray(img.convert("RGB"), np.float32) + np.random.normal(0, s*255.0, (img.size[1], img.size[0], 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def _webp(img, q):
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="WEBP", quality=int(q))
    buf.seek(0); return Image.open(buf).convert("RGB")

def apply_perturbation(img, name):
    if name == "clean":
        return img
    kind, val = name.split("_", 1)
    val = float(val) if "." in val else int(val)
    return {"jpeg": _jpeg, "blur": _blur, "resize": _resize,
            "noise": _noise, "webp": _webp}[kind](img, val)


def translate_duplicate(img, crop):
    if min(img.size) >= crop:
        return img
    w, h = img.size
    nw, nh = w*math.ceil(crop/w), h*math.ceil(crop/h)
    t = Image.new("RGB", (nw, nh))
    for x in range(0, nw, w):
        for y in range(0, nh, h):
            t.paste(img, (x, y))
    return t


# --- data -------------------------------------------------------------------
def gather_labeled(root, max_per_class=None):
    reals, fakes = [], []
    for dp, _d, files in os.walk(root):
        parts = [p.lower() for p in os.path.normpath(dp).split(os.sep)]
        lbl = None
        for p in reversed(parts):
            if p in FAKE_NAMES: lbl = 1; break
            if p in REAL_NAMES: lbl = 0; break
        if lbl is None:
            continue
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                (fakes if lbl == 1 else reals).append(os.path.join(dp, f))
    if max_per_class:
        reals = reals[:max_per_class]; fakes = fakes[:max_per_class]
    return [(p, 0) for p in reals] + [(p, 1) for p in fakes], len(reals), len(fakes)


class EvalDataset(Dataset):
    def __init__(self, samples, perturbation, crop, crop_mode, max_load_side):
        self.samples = samples
        self.perturbation = perturbation
        self.crop = crop
        self.crop_mode = crop_mode
        self.max_load_side = max_load_side
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize((crop, crop))
        self.center = T.CenterCrop(crop)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        try:
            img = Image.open(path)
            try:
                img.draft("RGB", (self.max_load_side, self.max_load_side))
            except Exception:
                pass
            img = img.convert("RGB")
            if max(img.size) > self.max_load_side:
                img.thumbnail((self.max_load_side, self.max_load_side), Image.BICUBIC)
            img = apply_perturbation(img, self.perturbation)
            if self.crop_mode == "resize":
                img = self.resize(img)
            else:  # centercrop
                img = self.center(translate_duplicate(img, self.crop))
        except Exception:
            img = Image.new("RGB", (self.crop, self.crop))
        return self.to_tensor(img), lbl


# --- model ------------------------------------------------------------------
def load_npr(path, device):
    model = resnet50(num_classes=1)
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:4]}...")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:4]}...")
    return model.to(device).eval()


@torch.no_grad()
def score(model, loader, device):
    ys, ps = [], []
    for imgs, labels in loader:
        out = model(imgs.to(device, non_blocking=True))
        ps.append(torch.sigmoid(out[:, 0]).cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def metrics(y, p):
    acc = accuracy_score(y, p > 0.5)
    try: ap = average_precision_score(y, p)
    except Exception: ap = float("nan")
    try: auc = roc_auc_score(y, p)
    except Exception: auc = float("nan")
    return acc, ap, auc


def gather_labeled_cap(samples, max_per_class):
    r = [x for x in samples if x[1] == 0][:max_per_class]
    fk = [x for x in samples if x[1] == 1][:max_per_class]
    return r + fk


def list_generators(test_dir):
    """Immediate subdirs of test_dir that contain real/fake -> {name: (samples,nr,nf)}."""
    gens = {}
    for d in sorted(os.listdir(test_dir)):
        full = os.path.join(test_dir, d)
        if os.path.isdir(full):
            s, nr, nf = gather_labeled(full)
            if s:
                gens[d] = (s, nr, nf)
    return gens


def eval_set_yp(model, samples, pert, opt, device):
    """Return (y, p) arrays; frees the loader/cuda cache afterwards."""
    ds = EvalDataset(samples, pert, opt.crop, opt.crop_mode, opt.max_load_side)
    loader = DataLoader(ds, batch_size=opt.batch_size, num_workers=opt.num_workers,
                        shuffle=False, pin_memory=False)
    y, p = score(model, loader, device)
    del loader, ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return y, p


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_path", required=True)
    pa.add_argument("--test_dir", required=True,
                    help="folder with real/fake, or generator subfolders each with real/fake")
    pa.add_argument("--crop", type=int, default=256)
    pa.add_argument("--crop_mode", default="resize", choices=["resize", "centercrop"])
    pa.add_argument("--perturbations", default="all",
                    help="'all', 'clean', or comma list")
    pa.add_argument("--batch_size", type=int, default=8)
    pa.add_argument("--num_workers", type=int, default=0)
    pa.add_argument("--max_load_side", type=int, default=1536)
    pa.add_argument("--max_per_class", type=int, default=None)
    pa.add_argument("--output_md", default="npr_paper_numbers.md")
    opt = pa.parse_args()

    if opt.perturbations.strip().lower() == "all":
        perts = list(PERTURBATIONS)
    else:
        perts = [p.strip() for p in opt.perturbations.split(",") if p.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nModel : {opt.model_path}")
    model = load_npr(opt.model_path, device)

    # generators (each evaluated in its own batch; nothing pooled in memory)
    gens = list_generators(opt.test_dir)
    if not gens:  # test_dir directly holds real/fake
        s, nr, nf = gather_labeled(opt.test_dir, opt.max_per_class)
        gens = {os.path.basename(os.path.normpath(opt.test_dir)) or "test": (s, nr, nf)}
    elif opt.max_per_class:
        gens = {g: (gather_labeled_cap(s, opt.max_per_class), nr, nf)
                for g, (s, nr, nf) in gens.items()}
    print(f"Test  : {opt.test_dir} | generators: {list(gens)}")
    print(f"Pipeline: perturbation -> {opt.crop_mode}({opt.crop}) -> ToTensor "
          f"| bs={opt.batch_size} workers={opt.num_workers}\n")
    if not gens:
        print("No real/fake images found."); return

    cross = {}                                   # clean cross-gen metrics
    rob_y = {pert: [] for pert in perts}         # accumulate only scalar arrays
    rob_p = {pert: [] for pert in perts}

    # ---- process ONE generator at a time (bounded memory) ----
    for g, (s, _nr, _nf) in gens.items():
        print(f"-- generator: {g}  ({len(s)} imgs) --")
        # clean cross-gen metric for this generator
        y0, p0 = eval_set_yp(model, s, "clean", opt, device)
        cross[g] = metrics(y0, p0)
        print(f"   clean: acc {cross[g][0]*100:.2f}  ap {cross[g][1]*100:.2f}  "
              f"auc {cross[g][2]*100:.2f}")
        # robustness: every perturbation on THIS generator, keep only y/p scalars
        for pert in perts:
            if pert == "clean":
                y, p = y0, p0
            else:
                y, p = eval_set_yp(model, s, pert, opt, device)
            rob_y[pert].append(y)
            rob_p[pert].append(p)
        del y0, p0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- pooled robustness metrics from the accumulated scalars ----
    print("\n== Robustness (pooled over all generators) ==")
    print(f"{'Perturbation':>16} | {'Acc':>6} {'AP':>6} {'AUC':>6}")
    print("-" * 44)
    rob = {}
    for pert in perts:
        y = np.concatenate(rob_y[pert]); p = np.concatenate(rob_p[pert])
        rob[pert] = metrics(y, p)
        print(f"{pert:>16} | {rob[pert][0]*100:>6.2f} {rob[pert][1]*100:>6.2f} "
              f"{rob[pert][2]*100:>6.2f}")

    # ---- markdown (mirrors the MORD paper-numbers layout) ----
    with open(opt.output_md, "w", encoding="utf-8") as f:
        f.write("# NPR Paper Numbers\n\n")
        f.write(f"Model: `{opt.model_path}`\n")
        f.write(f"Preprocessing: perturbation -> {opt.crop_mode}({opt.crop}) -> ToTensor "
                f"(no normalization). NPR is single-branch (one score).\n\n")

        f.write("## Cross-Generator Generalization (clean)\n\n")
        f.write("| Generator | Acc | AP | AUC |\n|---|---:|---:|---:|\n")
        for g, (acc, ap, auc) in cross.items():
            f.write(f"| {g} | {acc*100:.2f} | {ap*100:.2f} | {auc*100:.2f} |\n")
        mA = np.mean([v[0] for v in cross.values()]) * 100
        mP = np.mean([v[1] for v in cross.values()]) * 100
        mU = np.mean([v[2] for v in cross.values()]) * 100
        f.write(f"| **Mean** | **{mA:.2f}** | **{mP:.2f}** | **{mU:.2f}** |\n\n")

        f.write("## Robustness (AUC)\n\n")
        f.write("| Perturbation | NPR AUC |\n|---|---:|\n")
        for pert in perts:
            f.write(f"| {pert} | {rob[pert][2]*100:.2f} |\n")

        f.write("\n## Robustness (Acc)\n\n")
        f.write("| Perturbation | NPR Acc |\n|---|---:|\n")
        for pert in perts:
            f.write(f"| {pert} | {rob[pert][0]*100:.2f} |\n")

        f.write("\n## Robustness (AP)\n\n")
        f.write("| Perturbation | NPR AP |\n|---|---:|\n")
        for pert in perts:
            f.write(f"| {pert} | {rob[pert][1]*100:.2f} |\n")

    print(f"\nSaved Markdown -> {opt.output_md}")


if __name__ == "__main__":
    main()
