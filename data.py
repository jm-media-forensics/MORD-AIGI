# -*- coding: utf-8 -*-
"""Data loading, training augmentation and test-time perturbations.

Folder layout: every data folder contains 0_real/ and 1_fake/.
Images are decoded with PIL into 8-bit RGB and scaled to [0, 1]; no normalization.
"""
import io
import math
import os
from io import BytesIO
from random import choice, random, uniform

import cv2
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from PIL import Image, ImageFile, ImageFilter
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def translate_duplicate(img, crop_size):
    """Tile images smaller than crop_size."""
    if min(img.size) >= crop_size:
        return img
    w, h = img.size
    new_w, new_h = w * math.ceil(crop_size / w), h * math.ceil(crop_size / h)
    tiled = Image.new("RGB", (new_w, new_h))
    for i in range(0, new_w, w):
        for j in range(0, new_h, h):
            tiled.paste(img, (i, j))
    return tiled


def class_roots(root, classes):
    """root/<class>/{0_real,1_fake} for every class, or root itself."""
    classes = classes or sorted(os.listdir(root))
    if "0_real" in classes and "1_fake" in classes:
        return [root]
    return [os.path.join(root, c) for c in classes]


# ---------------------------------------------------------------------------
# Clean training (UFD / Self-Synthesis and GenImage settings)
# ---------------------------------------------------------------------------
def clean_train_loader(root, classes, batch_size, crop=224, num_workers=8):
    tf = T.Compose([T.Lambda(lambda im: translate_duplicate(im, crop)),
                    T.RandomCrop(crop), T.RandomHorizontalFlip(), T.ToTensor()])
    roots = class_roots(root, classes)
    dsets = [datasets.ImageFolder(r, tf) for r in roots]
    dset = dsets[0] if roots == [root] else torch.utils.data.ConcatDataset(dsets)
    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Degradation augmentation for paired training
# ---------------------------------------------------------------------------
def _pil_jpeg(arr, quality):
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
    out = np.array(Image.open(buf))
    buf.close()
    return out


def _cv2_jpeg(arr, quality):
    ok, enc = cv2.imencode(".jpg", arr[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return arr
    return cv2.imdecode(enc, 1)[:, :, ::-1]


_JPEG_FNS = {"pil": _pil_jpeg, "cv2": _cv2_jpeg}
_INTERP_DOWN = [cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]
_INTERP_UP = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def _gaussian_blur(arr, sigma):
    if sigma <= 0:
        return arr
    out = arr.copy()
    for c in range(out.shape[2]):
        gaussian_filter(out[:, :, c], output=out[:, :, c], sigma=sigma)
    return out


def _random_resize(arr, scale):
    h, w = arr.shape[:2]
    down = cv2.resize(arr, (max(8, int(round(w * scale))), max(8, int(round(h * scale)))),
                      interpolation=choice(_INTERP_DOWN))
    return cv2.resize(down, (w, h), interpolation=choice(_INTERP_UP))


def _gaussian_noise(arr, sigma):
    if sigma <= 0:
        return arr
    noise = np.random.normal(0.0, sigma * 255.0, arr.shape).astype(np.float32)
    return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


PRESETS = {
    "mild": dict(jpg_prob=0.3, jpg_qual_range=(70, 100), jpg_methods=("pil",),
                 blur_prob=0.3, blur_sig_range=(0.0, 1.0),
                 resize_prob=0.0, resize_scale_range=(0.5, 1.5),
                 noise_prob=0.0, noise_sigma_range=(0.0, 0.05)),
    "aggressive": dict(jpg_prob=0.7, jpg_qual_range=(30, 100), jpg_methods=("pil", "cv2"),
                       blur_prob=0.5, blur_sig_range=(0.0, 3.0),
                       resize_prob=0.5, resize_scale_range=(0.4, 1.6),
                       noise_prob=0.2, noise_sigma_range=(0.0, 0.04)),
}


class Augmentor:
    """Blur, noise, resize and JPEG, in this order, each with its own probability."""

    def __init__(self, preset):
        self.__dict__.update(PRESETS[preset])
        self.preset = preset

    def __repr__(self):
        return f"Augmentor({self.preset})"

    def __call__(self, pil_img):
        arr = np.array(pil_img.convert("RGB"))
        if random() < self.blur_prob:
            arr = _gaussian_blur(arr, uniform(*self.blur_sig_range))
        if random() < self.noise_prob:
            arr = _gaussian_noise(arr, uniform(*self.noise_sigma_range))
        if random() < self.resize_prob:
            arr = _random_resize(arr, uniform(*self.resize_scale_range))
        if random() < self.jpg_prob:
            q = int(round(uniform(*self.jpg_qual_range)))
            arr = _JPEG_FNS[choice(self.jpg_methods)](arr, q)
        return Image.fromarray(arr)


class PairedImageFolder(Dataset):
    """Returns (clean view, augmented view, label). Both views share resize, crop and flip."""

    def __init__(self, root, augmentor, crop=224, load_size=256):
        self.base = datasets.ImageFolder(root)
        self.augmentor, self.crop = augmentor, crop
        self.resize = T.Resize((load_size, load_size))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        pil, label = self.base[idx]
        pil = translate_duplicate(pil.convert("RGB"), self.crop)
        pil_aug = self.augmentor(pil)
        pil, pil_aug = self.resize(pil), self.resize(pil_aug)
        i, j, h, w = T.RandomCrop.get_params(pil, (self.crop, self.crop))
        flip = random() < 0.5
        views = []
        for im in (pil, pil_aug):
            im = T.functional.crop(im, i, j, h, w)
            views.append(self.to_tensor(T.functional.hflip(im) if flip else im))
        return views[0], views[1], label


def paired_train_loader(root, classes, augmentor, batch_size, crop=224, num_workers=8):
    roots = class_roots(root, classes)
    dsets = [PairedImageFolder(r, augmentor, crop) for r in roots]
    dset = dsets[0] if roots == [root] else torch.utils.data.ConcatDataset(dsets)
    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Test-time perturbations
# ---------------------------------------------------------------------------
def _jpeg(img, q):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _resize(img, factor):
    w, h = img.size
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    return img.resize((nw, nh), Image.BICUBIC).resize((w, h), Image.BICUBIC)


def _noise(img, sigma):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr = arr + np.random.normal(0.0, sigma * 255.0, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def perturb(img, condition):
    """condition: clean, jpeg_<QF>, blur_<sigma>, noise_<sigma>, resize_<factor>,
    or a chain joined by '+', applied left to right (e.g. resize_0.5+jpeg_75)."""
    for part in condition.split("+"):
        if part == "clean":
            continue
        kind, val = part.split("_", 1)
        val = float(val) if "." in val else int(val)
        if kind == "jpeg":
            img = _jpeg(img, val)
        elif kind == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=val))
        elif kind == "resize":
            img = _resize(img, val)
        elif kind == "noise":
            img = _noise(img, val)
        else:
            raise ValueError(f"unknown condition: {part}")
    return img


# ---------------------------------------------------------------------------
# Test datasets
# ---------------------------------------------------------------------------
TEST_TRANSFORMS = {
    # UFD, Self-Synthesis, GenImage: tile if small, center crop 256, no resize
    "clean": lambda: T.Compose([T.Lambda(lambda im: translate_duplicate(im, 256)), T.CenterCrop(256), T.ToTensor()]),
    # post-processing and format-matched evaluation: tile if small, resize to 256x256
    "robust": lambda: T.Compose([T.Lambda(lambda im: translate_duplicate(im, 256)), T.Resize((256, 256)),
                                 T.CenterCrop(256), T.ToTensor()]),
    # WildRF: whole image resized to 256x256
    "wildrf": lambda: T.Compose([T.Resize((256, 256)), T.ToTensor()]),
}


def gather(root, cap=None):
    """Images under root, labelled by the nearest 0_real / 1_fake folder."""
    reals, fakes = [], []
    for dp, _dirs, fs in os.walk(root):
        parts = [p.lower() for p in os.path.normpath(dp).split(os.sep)]
        lbl = next((0 if p in ("0_real", "real") else 1 for p in reversed(parts)
                    if p in ("0_real", "real", "1_fake", "fake")), None)
        if lbl is None:
            continue
        for f in sorted(fs):
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                (fakes if lbl else reals).append(os.path.join(dp, f))
    if cap:
        reals, fakes = reals[:cap], fakes[:cap]
    return [(p, 0) for p in reals] + [(p, 1) for p in fakes]


class TestDataset(Dataset):
    def __init__(self, items, protocol, condition="clean"):
        self.items, self.condition = items, condition
        self.tf = TEST_TRANSFORMS[protocol]()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        try:
            img = perturb(Image.open(path).convert("RGB"), self.condition)
        except Exception as e:
            print(f"[warn] {path}: {e}")
            img = Image.new("RGB", (256, 256))
        return self.tf(img), y
