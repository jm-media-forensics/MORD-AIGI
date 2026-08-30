# -*- coding: utf-8 -*-
"""
eval_univfd.py
Full robustness evaluation of UnivFD (Ojha et al., CVPR 2023).

UnivFD = frozen CLIP ViT-L/14 image encoder (768-d) + a single Linear(768 -> 1).
Preprocessing (CLIP standard):
    perturbation -> Resize(224, BICUBIC) -> CenterCrop(224) -> ToTensor
                 -> Normalize(CLIP_MEAN, CLIP_STD)
This is intentionally different from MORD/NPR (no normalization, 256 input);
each method must be evaluated under its own native preprocessing.

Output Markdown mirrors NPR/MORD paper-number layout:
    Cross-Generator Generalization (clean)
    Robustness (AUC) / (Acc) / (AP)

Requires:
    pip install git+https://github.com/openai/CLIP.git
    pip install ftfy regex tqdm
A UnivFD pretrained checkpoint (download from their releases or train it
yourself via the official repo).

Usage:
  python eval_univfd.py --model_path /path/to/univfd_ckpt.pth \
    --test_dir /home/.../dataset_out/test --output_md univfd_paper_numbers.md
"""
import os
import io
import gc
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageFile, ImageFilter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Lazy CLIP import with helpful error
try:
    import clip
except ImportError as e:
    raise ImportError(
        "openai-clip is required: pip install git+https://github.com/openai/CLIP.git"
    ) from e

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_FEAT_DIM = {"ViT-L/14": 768, "RN50": 1024, "ViT-B/32": 512, "ViT-B/16": 512}

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


# --- perturbation operators (identical to NPR/MORD eval) --------------------
def _jpeg(img, q):
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="JPEG", quality=int(q))
    buf.seek(0); return Image.open(buf).convert("RGB")

def _blur(img, s):
    return img.filter(ImageFilter.GaussianBlur(radius=float(s)))

def _resize(img, f):
    w, h = img.size
    nw, nh = max(1, round(w * f)), max(1, round(h * f))
    return img.resize((nw, nh), Image.BICUBIC).resize((w, h), Image.BICUBIC)

def _noise(img, s):
    a = np.asarray(img.convert("RGB"), np.float32) + np.random.normal(
        0, s * 255.0, (img.size[1], img.size[0], 3))
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


# --- UnivFD model -----------------------------------------------------------
class UnivFDModel(nn.Module):
    """Frozen CLIP image encoder + Linear(d, 1) head, as in Ojha et al."""
    def __init__(self, backbone="ViT-L/14"):
        super().__init__()
        self.backbone = backbone
        self.model, _ = clip.load(backbone, device="cpu")
        d = CLIP_FEAT_DIM[backbone]
        self.fc = nn.Linear(d, 1)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            feats = self.model.encode_image(x).float()
        return self.fc(feats)        # [B,1]


def load_univfd(path, backbone, device):
    """Load UnivFD weights. Handles three checkpoint formats:
      (1) Full state_dict with CLIP encoder + fc head
      (2) Linear-head-only state_dict {'fc.weight','fc.bias'} (most common)
      (3) Bare linear-head state_dict {'weight','bias'}
    The CLIP encoder is always reloaded from openai-clip's pretrained weights.
    """
    model = UnivFDModel(backbone=backbone)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    keys = set(sd.keys())
    # case (3): bare fc submodule -> remap
    if keys == {"weight", "bias"}:
        print("[info] detected bare fc state_dict ('weight','bias') -> remapping to 'fc.*'")
        sd = {"fc.weight": sd["weight"], "fc.bias": sd["bias"]}
    # case (2): only fc.* keys -> load into model.fc only
    if set(sd.keys()) <= {"fc.weight", "fc.bias"}:
        print(f"[info] linear-head-only checkpoint (keys: {list(sd)}) -> loading into model.fc")
        fc_sd = {k.replace("fc.", ""): v for k, v in sd.items()}
        model.fc.load_state_dict(fc_sd, strict=True)
        print(f"  fc.weight shape: {tuple(model.fc.weight.shape)}  "
              f"(checkpoint expects {tuple(sd['fc.weight'].shape)})")
        return model.to(device).eval()

    # case (1): full state_dict
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:4]}...")
    if unexpected:
        print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:4]}...")
    if not any("fc" in k for k in sd.keys()):
        print("[warn] no 'fc.*' found in checkpoint; the linear head may be uninitialized.")
    return model.to(device).eval()


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


def list_generators(test_dir):
    gens = {}
    for d in sorted(os.listdir(test_dir)):
        full = os.path.join(test_dir, d)
        if os.path.isdir(full):
            s, nr, nf = gather_labeled(full)
            if s:
                gens[d] = (s, nr, nf)
    return gens


class EvalDataset(Dataset):
    """perturbation -> CLIP standard preprocessing -> tensor."""
    def __init__(self, samples, perturbation, max_load_side):
        self.samples = samples
        self.perturbation = perturbation
        self.max_load_side = max_load_side
        # CLIP preprocessing (Resize 224 BICUBIC, CenterCrop 224, ToTensor, Normalize)
        self.tf = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

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
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.tf(img), lbl


# --- eval helpers -----------------------------------------------------------
@torch.no_grad()
def score(model, loader, device):
    ys, ps = [], []
    use_amp = (device.type == "cuda")
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(imgs)
        ps.append(torch.sigmoid(out[:, 0].float()).cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def metrics(y, p):
    acc = accuracy_score(y, p > 0.5)
    try: ap = average_precision_score(y, p)
    except Exception: ap = float("nan")
    try: auc = roc_auc_score(y, p)
    except Exception: auc = float("nan")
    return acc, ap, auc


def eval_set_yp(model, samples, pert, opt, device):
    ds = EvalDataset(samples, pert, opt.max_load_side)
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
    pa.add_argument("--model_path", required=True,
                    help="UnivFD checkpoint (state_dict including CLIP encoder + fc head)")
    pa.add_argument("--backbone", default="ViT-L/14",
                    choices=list(CLIP_FEAT_DIM), help="CLIP variant used in training")
    pa.add_argument("--test_dir", required=True,
                    help="folder with real/fake, or generator subfolders each with real/fake")
    pa.add_argument("--perturbations", default="all",
                    help="'all', 'clean', or comma list")
    pa.add_argument("--batch_size", type=int, default=8)
    pa.add_argument("--num_workers", type=int, default=0)
    pa.add_argument("--max_load_side", type=int, default=1536)
    pa.add_argument("--max_per_class", type=int, default=None)
    pa.add_argument("--output_md", default="univfd_paper_numbers.md")
    opt = pa.parse_args()

    if opt.perturbations.strip().lower() == "all":
        perts = list(PERTURBATIONS)
    else:
        perts = [p.strip() for p in opt.perturbations.split(",") if p.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}\nBackbone: CLIP {opt.backbone}\nCheckpt : {opt.model_path}")
    model = load_univfd(opt.model_path, opt.backbone, device)

    # generators
    gens = list_generators(opt.test_dir)
    if not gens:
        s, nr, nf = gather_labeled(opt.test_dir, opt.max_per_class)
        gens = {os.path.basename(os.path.normpath(opt.test_dir)) or "test": (s, nr, nf)}
    if opt.max_per_class:
        gens = {g: ([x for x in s if x[1] == 0][:opt.max_per_class]
                    + [x for x in s if x[1] == 1][:opt.max_per_class],
                    min(nr, opt.max_per_class), min(nf, opt.max_per_class))
                for g, (s, nr, nf) in gens.items()}

    print(f"Test    : {opt.test_dir} | generators: {list(gens)}")
    print(f"Pipeline: perturbation -> Resize(224)->CenterCrop(224)->ToTensor->CLIP-Normalize\n")

    # ---- 1. cross-generator (clean) ----
    print("== Cross-generator (clean) ==")
    print(f"{'Generator':>22} | {'Acc':>6} {'AP':>6} {'AUC':>6}")
    print("-" * 50)
    cross = {}
    for g, (s, _nr, _nf) in gens.items():
        y, p = eval_set_yp(model, s, "clean", opt, device)
        acc, ap, auc = metrics(y, p)
        cross[g] = (acc, ap, auc)
        print(f"{g:>22} | {acc*100:>6.2f} {ap*100:>6.2f} {auc*100:>6.2f}")

    # ---- 2. robustness (per-generator accumulation, then pooled) ----
    print("\n== Robustness (pooled over all generators, processed per-folder) ==")
    rob_y = {pert: [] for pert in perts}
    rob_p = {pert: [] for pert in perts}
    for g, (s, _nr, _nf) in gens.items():
        print(f"  -- {g} ({len(s)} imgs) --")
        for pert in perts:
            y, p = eval_set_yp(model, s, pert, opt, device)
            rob_y[pert].append(y); rob_p[pert].append(p)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'Perturbation':>16} | {'Acc':>6} {'AP':>6} {'AUC':>6}")
    print("-" * 44)
    rob = {}
    for pert in perts:
        y = np.concatenate(rob_y[pert]); p = np.concatenate(rob_p[pert])
        rob[pert] = metrics(y, p)
        acc, ap, auc = rob[pert]
        print(f"{pert:>16} | {acc*100:>6.2f} {ap*100:>6.2f} {auc*100:>6.2f}")

    # ---- markdown (mirrors NPR/MORD paper-numbers layout) ----
    with open(opt.output_md, "w", encoding="utf-8") as f:
        f.write("# UnivFD Paper Numbers\n\n")
        f.write(f"Backbone: CLIP {opt.backbone}\n")
        f.write(f"Model   : `{opt.model_path}`\n")
        f.write("Preprocessing: perturbation -> Resize(224, BICUBIC) -> CenterCrop(224) "
                "-> ToTensor -> Normalize(CLIP_MEAN, CLIP_STD). UnivFD is single-branch "
                "(one score).\n\n")

        f.write("## Cross-Generator Generalization (clean)\n\n")
        f.write("| Generator | Acc | AP | AUC |\n|---|---:|---:|---:|\n")
        for g, (acc, ap, auc) in cross.items():
            f.write(f"| {g} | {acc*100:.2f} | {ap*100:.2f} | {auc*100:.2f} |\n")
        mA = np.mean([v[0] for v in cross.values()]) * 100
        mP = np.mean([v[1] for v in cross.values()]) * 100
        mU = np.mean([v[2] for v in cross.values()]) * 100
        f.write(f"| **Mean** | **{mA:.2f}** | **{mP:.2f}** | **{mU:.2f}** |\n\n")

        f.write("## Robustness (AUC)\n\n")
        f.write("| Perturbation | UnivFD AUC |\n|---|---:|\n")
        for pert in perts:
            f.write(f"| {pert} | {rob[pert][2]*100:.2f} |\n")

        f.write("\n## Robustness (Acc)\n\n")
        f.write("| Perturbation | UnivFD Acc |\n|---|---:|\n")
        for pert in perts:
            f.write(f"| {pert} | {rob[pert][0]*100:.2f} |\n")

        f.write("\n## Robustness (AP)\n\n")
        f.write("| Perturbation | UnivFD AP |\n|---|---:|\n")
        for pert in perts:
            f.write(f"| {pert} | {rob[pert][1]*100:.2f} |\n")

    print(f"\nSaved Markdown -> {opt.output_md}")


if __name__ == "__main__":
    main()
