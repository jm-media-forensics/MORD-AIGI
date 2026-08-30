# -*- coding: utf-8 -*-
"""Robust-training dataset wrapper.

Mirrors the dataset construction in `data/datasets.py` but injects the
`RobustAugmentor` into the train-time transform pipeline. The original
`data/datasets.py` is left untouched so the clean-training pipeline keeps
its exact behaviour.

Public entry points:
    binary_dataset_robust(opt, root)
    dataset_folder_robust(opt, root)
"""

import math
from random import choice

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image, ImageFile

from .robust_augment import build_augmentor, RobustAugmentor

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# Small helpers reused from data/datasets.py to keep behaviour identical
# ---------------------------------------------------------------------------
def _translate_duplicate(img, cropSize):
    """Tile small images up to cropSize, identical to data.datasets."""
    if min(img.size) < cropSize:
        w, h = img.size
        new_w = w * math.ceil(cropSize / w)
        new_h = h * math.ceil(cropSize / h)
        new_img = Image.new("RGB", (new_w, new_h))
        for i in range(0, new_w, w):
            for j in range(0, new_h, h):
                new_img.paste(img, (i, j))
        return new_img
    return img


class _RandomResizeChoice:
    def __init__(self, sizes):
        self.sizes = sizes

    def __call__(self, img):
        s = choice(self.sizes)
        return transforms.Resize((s, s))(img)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def binary_dataset_robust(opt, root):
    """ImageFolder + robust augmentation pipeline for training.

    At test time (opt.isTrain == False) NO robust augmentation is applied —
    test-time perturbations are administered by the evaluation script via
    a separate evaluator (so train/test perturbations are not coupled).
    """
    cropSize = opt.cropSize
    loadSize = getattr(opt, "loadSize", 256)

    if opt.isTrain:
        crop_func = transforms.RandomCrop(cropSize)
    elif getattr(opt, "no_crop", False):
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(cropSize)

    if opt.isTrain and not getattr(opt, "no_flip", False):
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)

    # Resize policy (unchanged from your data/datasets.py)
    rz_func = transforms.Resize((loadSize, loadSize))

    # --- Build the augmentor (training only) --------------------------------
    if opt.isTrain:
        preset = getattr(opt, "aug_preset", "standard")
        overrides = {}
        # CLI-level fine overrides (all optional; fall back to preset)
        if getattr(opt, "aug_jpg_prob", None) is not None:
            overrides["jpg_prob"] = opt.aug_jpg_prob
        if getattr(opt, "aug_jpg_qual", None) is not None:
            overrides["jpg_qual_range"] = tuple(opt.aug_jpg_qual)
        if getattr(opt, "aug_blur_prob", None) is not None:
            overrides["blur_prob"] = opt.aug_blur_prob
        if getattr(opt, "aug_blur_sig", None) is not None:
            overrides["blur_sig_range"] = tuple(opt.aug_blur_sig)
        if getattr(opt, "aug_resize_prob", None) is not None:
            overrides["resize_prob"] = opt.aug_resize_prob
        if getattr(opt, "aug_resize_scale", None) is not None:
            overrides["resize_scale_range"] = tuple(opt.aug_resize_scale)
        if getattr(opt, "aug_noise_prob", None) is not None:
            overrides["noise_prob"] = opt.aug_noise_prob
        if getattr(opt, "aug_noise_sigma", None) is not None:
            overrides["noise_sigma_range"] = tuple(opt.aug_noise_sigma)
        if getattr(opt, "aug_order_randomize", False):
            overrides["order_randomize"] = True

        augmentor: RobustAugmentor = build_augmentor(preset, **overrides)
        print(f"[robust-train] augmentor: {augmentor}")
        aug_lambda = transforms.Lambda(lambda img: augmentor(img))
    else:
        aug_lambda = transforms.Lambda(lambda img: img)

    # Compose. Important ordering:
    #   1) tile small images so crop is valid
    #   2) apply robust augmentation BEFORE crop, so JPEG sees full-resolution
    #      compression artifacts that span 8x8 blocks at the original scale
    #   3) resize-or-keep
    #   4) crop
    #   5) flip
    #   6) to tensor
    pipeline = transforms.Compose([
        transforms.Lambda(lambda img: _translate_duplicate(img, cropSize)),
        aug_lambda,
        rz_func,
        crop_func,
        flip_func,
        transforms.ToTensor(),
    ])

    return datasets.ImageFolder(root, pipeline)


def dataset_folder_robust(opt, root):
    """Mode-aware wrapper, matching the API of data.datasets.dataset_folder."""
    if opt.mode == "binary":
        return binary_dataset_robust(opt, root)
    raise ValueError("Only mode='binary' is supported for robust training.")
