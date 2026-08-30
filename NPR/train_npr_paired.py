# -*- coding: utf-8 -*-
"""
train_npr_paired.py
Train the NPR detector with the SAME *paired* robust-augmentation protocol and
the SAME command-line interface as MORD's train_robust_paired.py.

Example (mirrors the MORD command):
  python train_npr_paired.py --name npr_sdv4_paired_mild \
    --dataroot /home/ubuntu/workspace/NPR-DeepfakeDetection-main2/datasets/dataset_out/ \
    --classes stable_diffusion_v_1_4 \
    --batch_size 32 --lr 0.0004 --niter 50 --cropSize 224 --aug_preset mild

Paired protocol (identical to MORD datasets_paired.py):
  per image -> tile-if-small ; aug_view = augment(image) ;
  Resize(256,256) on BOTH ; SyncRandomCrop(cropSize) + SyncRandomFlip ;
  ToTensor (NO normalization)  -> returns (clean, aug, label)
Loss = BCE(model(clean), y) + BCE(model(aug), y).

NPR residual NPR = x - interpolate(x, 0.5) is inside the model.
"""
import os
import io
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile, ImageFilter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, accuracy_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

from networks.resnet import resnet50   # NPR model

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REAL_NAMES = {"real", "0", "0_real", "nature"}
FAKE_NAMES = {"fake", "1", "1_fake", "ai"}


# ---------------------------------------------------------------------------
# Augmentation presets (identical to MORD robust_augment.py)
# ---------------------------------------------------------------------------
PRESETS = {
    "clean":   dict(jpg_prob=0.0, blur_prob=0.0, resize_prob=0.0, noise_prob=0.0),
    "mild":    dict(jpg_prob=0.3, jpg_qual=(70, 100), blur_prob=0.3, blur_sig=(0.0, 1.0),
                    resize_prob=0.0, noise_prob=0.0),
    "cnnspot": dict(jpg_prob=0.5, jpg_qual=(30, 100), blur_prob=0.5, blur_sig=(0.0, 3.0),
                    resize_prob=0.0, noise_prob=0.0),
    "standard": dict(jpg_prob=0.5, jpg_qual=(50, 100), blur_prob=0.5, blur_sig=(0.0, 2.0),
                     resize_prob=0.3, resize_scale=(0.5, 1.5), noise_prob=0.0),
    "aggressive": dict(jpg_prob=0.7, jpg_qual=(30, 100), blur_prob=0.5, blur_sig=(0.0, 3.0),
                       resize_prob=0.5, resize_scale=(0.4, 1.6),
                       noise_prob=0.2, noise_sig=(0.0, 0.04)),
}


def _jpeg(img, q):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _blur(img, sigma):
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def _resize(img, factor):
    w, h = img.size
    nw, nh = max(1, round(w * factor)), max(1, round(h * factor))
    return img.resize((nw, nh), Image.BICUBIC).resize((w, h), Image.BICUBIC)


def _noise(img, sigma):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr = arr + np.random.normal(0.0, sigma * 255.0, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class RobustAugment:
    """Applies blur/resize/noise then JPEG (JPEG last), each by its probability."""
    def __init__(self, cfg):
        self.c = cfg

    def __call__(self, img):
        c = self.c
        if c.get("blur_prob", 0) and random.random() < c["blur_prob"]:
            img = _blur(img, random.uniform(*c["blur_sig"]))
        if c.get("resize_prob", 0) and random.random() < c["resize_prob"]:
            img = _resize(img, random.uniform(*c["resize_scale"]))
        if c.get("noise_prob", 0) and random.random() < c["noise_prob"]:
            img = _noise(img, random.uniform(*c["noise_sig"]))
        if c.get("jpg_prob", 0) and random.random() < c["jpg_prob"]:
            img = _jpeg(img, random.randint(*c["jpg_qual"]))
        return img


def translate_duplicate(img, crop_size):
    if min(img.size) >= crop_size:
        return img
    w, h = img.size
    nw = w * math.ceil(crop_size / w)
    nh = h * math.ceil(crop_size / h)
    tiled = Image.new("RGB", (nw, nh))
    for x in range(0, nw, w):
        for y in range(0, nh, h):
            tiled.paste(img, (x, y))
    return tiled


class _SyncRandomCrop:
    def __init__(self, size):
        self.size = size; self._cached = None

    def __call__(self, img):
        if self._cached is None:
            self._cached = T.RandomCrop.get_params(img, (self.size, self.size))
        i, j, h, w = self._cached
        return TF.crop(img, i, j, h, w)


class _SyncRandomFlip:
    def __init__(self):
        self._cached = None

    def __call__(self, img):
        if self._cached is None:
            self._cached = random.random() < 0.5
        return TF.hflip(img) if self._cached else img


# ---------------------------------------------------------------------------
def gather_labeled(root, max_per_class=None):
    reals, fakes = [], []
    for dp, _dirs, files in os.walk(root):
        parts = [p.lower() for p in os.path.normpath(dp).split(os.sep)]
        lbl = None
        for p in reversed(parts):
            if p in FAKE_NAMES:
                lbl = 1; break
            if p in REAL_NAMES:
                lbl = 0; break
        if lbl is None:
            continue
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                (fakes if lbl == 1 else reals).append(os.path.join(dp, f))
    if max_per_class:
        reals = reals[:max_per_class]
        fakes = fakes[:max_per_class]
    return [(p, 0) for p in reals] + [(p, 1) for p in fakes], len(reals), len(fakes)


class PairedNPRDataset(Dataset):
    def __init__(self, root, crop, augmentor, max_per_class=None):
        self.samples, self.n_real, self.n_fake = gather_labeled(root, max_per_class)
        self.crop = crop
        self.augmentor = augmentor
        self.resize = T.Resize((256, 256))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        try:
            pil = Image.open(path).convert("RGB")
            pil = translate_duplicate(pil, self.crop)
            pil_aug = self.augmentor(pil)
            pil = self.resize(pil)
            pil_aug = self.resize(pil_aug)
            crop = _SyncRandomCrop(self.crop)
            flip = _SyncRandomFlip()
            pil = flip(crop(pil))
            pil_aug = flip(crop(pil_aug))
            return self.to_tensor(pil), self.to_tensor(pil_aug), lbl
        except Exception:
            z = torch.zeros(3, self.crop, self.crop)
            return z, z.clone(), lbl


class ValNPRDataset(Dataset):
    def __init__(self, root, max_per_class=None):
        self.samples, self.n_real, self.n_fake = gather_labeled(root, max_per_class)
        self.resize = T.Resize((256, 256))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        try:
            img = self.resize(Image.open(path).convert("RGB"))
        except Exception:
            img = Image.new("RGB", (256, 256))
        return self.to_tensor(img), lbl


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for imgs, labels in loader:
        out = model(imgs.to(device))
        ps.append(torch.sigmoid(out[:, 0]).cpu().numpy())
        ys.append(labels.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    acc = accuracy_score(y, p > 0.5)
    try:
        ap = average_precision_score(y, p)
    except Exception:
        ap = float("nan")
    return acc, ap


def seed_all(seed=100):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    pa = argparse.ArgumentParser()
    # --- MORD train_robust_paired.py compatible interface ---
    pa.add_argument("--name", default="npr_paired_mild")
    pa.add_argument("--dataroot", required=True,
                    help="base data root; train data = {dataroot}/{train_split}/{classes}")
    pa.add_argument("--classes", default="stable_diffusion_v_1_4",
                    help="generator/class subfolder under each split")
    pa.add_argument("--train_split", default="train")
    pa.add_argument("--val_split", default="val")
    pa.add_argument("--batch_size", type=int, default=32)
    pa.add_argument("--lr", type=float, default=0.0004)
    pa.add_argument("--niter", type=int, default=50, help="number of epochs")
    pa.add_argument("--cropSize", type=int, default=224)
    pa.add_argument("--aug_preset", default="mild", choices=list(PRESETS))
    pa.add_argument("--checkpoints_dir", default="./checkpoints")
    pa.add_argument("--num_workers", type=int, default=4)
    pa.add_argument("--save_freq", type=int, default=5)
    pa.add_argument("--max_per_class", type=int, default=None)
    pa.add_argument("--no_val", action="store_true", help="skip validation")
    opt = pa.parse_args()

    seed_all(100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(out_dir, exist_ok=True)

    train_root = os.path.join(opt.dataroot, opt.train_split, opt.classes)
    val_root = os.path.join(opt.dataroot, opt.val_split, opt.classes)
    print(f"Device: {device} | save -> {out_dir}")
    print(f"aug_preset: {opt.aug_preset} -> {PRESETS[opt.aug_preset]}")
    print(f"train_root: {train_root}")

    augmentor = RobustAugment(PRESETS[opt.aug_preset])
    train_ds = PairedNPRDataset(train_root, crop=opt.cropSize, augmentor=augmentor,
                                max_per_class=opt.max_per_class)
    print(f"Train(paired): {len(train_ds)} (real={train_ds.n_real}, fake={train_ds.n_fake})")
    if len(train_ds) == 0:
        raise FileNotFoundError(f"No real/fake images under {train_root}. "
                                f"Check --dataroot/--train_split/--classes.")
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    val_loader = None
    if not opt.no_val and os.path.isdir(val_root):
        val_ds = ValNPRDataset(val_root, max_per_class=opt.max_per_class)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)
            print(f"Val  : {len(val_ds)} (real={val_ds.n_real}, fake={val_ds.n_fake})")

    model = resnet50(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    best_acc = 0.0
    for epoch in range(opt.niter):
        model.train()
        t0 = time.time(); running = 0.0; nb = 0
        for clean, aug, labels in train_loader:
            clean = clean.to(device); aug = aug.to(device)
            labels = labels.float().to(device).view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(clean), labels) + criterion(model(aug), labels)
            loss.backward(); optimizer.step()
            running += loss.item(); nb += 1
        msg = f"[epoch {epoch:2d}] loss={running/max(nb,1):.4f} ({time.time()-t0:.0f}s)"

        if val_loader is not None and (epoch % opt.save_freq == 0 or epoch == opt.niter - 1):
            acc, ap = validate(model, val_loader, device)
            msg += f" | val acc={acc*100:.2f} ap={ap*100:.2f}"
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pth"))
        print(msg)

        if epoch % opt.save_freq == 0 or epoch == opt.niter - 1:
            torch.save(model.state_dict(), os.path.join(out_dir, f"model_epoch_{epoch}.pth"))
        torch.save(model.state_dict(), os.path.join(out_dir, "model_epoch_last.pth"))

    print(f"Done. best val acc={best_acc*100:.2f}. checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
