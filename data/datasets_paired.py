# -*- coding: utf-8 -*-
"""Paired (clean + augmented) dataset for robust training.

Each __getitem__ returns:
    (img_clean, img_aug, label)
where `img_aug` is the same source image with random JPEG / blur / resize
applied. Both go through the same downstream crop / flip / tensorize so
spatial alignment is preserved across the pair.

Why pair instead of "mix in batch":
  - Guarantees every batch has exactly the same number of clean and
    augmented samples (no random imbalance).
  - The two halves come from the SAME source images, so the model sees
    "what this image looks like clean" and "what it looks like degraded"
    together — the cleanest possible signal for invariance learning.
  - Avoids the failure mode where heavy augmentation pushes real-class
    residuals into fake-class territory, causing decision-boundary
    collapse (racc → 0).
"""

import math
from random import choice
from copy import deepcopy

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image, ImageFile

from .robust_augment import build_augmentor, RobustAugmentor

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _translate_duplicate(img, cropSize):
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


class _SyncRandomCrop:
    """RandomCrop where two calls in a row use the same parameters."""
    def __init__(self, size):
        self.size = size
        self._cached = None

    def reset(self):
        self._cached = None

    def __call__(self, img):
        if self._cached is None:
            i, j, h, w = transforms.RandomCrop.get_params(img, (self.size, self.size))
            self._cached = (i, j, h, w)
        i, j, h, w = self._cached
        return transforms.functional.crop(img, i, j, h, w)


class _SyncRandomFlip:
    def __init__(self):
        self._cached = None

    def reset(self):
        self._cached = None

    def __call__(self, img):
        if self._cached is None:
            import random as _r
            self._cached = _r.random() < 0.5
        if self._cached:
            return transforms.functional.hflip(img)
        return img


class PairedImageFolder(Dataset):
    """ImageFolder that returns (img_clean, img_aug, label) triplets.

    Spatial transforms (resize/crop/flip) are SYNCHRONIZED between the
    clean and augmented branches so the two views correspond to the
    same spatial region of the original image. Only the photometric
    degradations (JPEG/blur/resize-down-up/noise) differ.
    """

    def __init__(self, opt, root, augmentor: RobustAugmentor):
        self.base = datasets.ImageFolder(root)
        self.opt = opt
        self.augmentor = augmentor
        self.cropSize = opt.cropSize
        self.loadSize = getattr(opt, "loadSize", 256)
        self.no_flip = getattr(opt, "no_flip", False)
        self.resize = transforms.Resize((self.loadSize, self.loadSize))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        pil, label = self.base[idx]
        pil = pil.convert("RGB")
        pil = _translate_duplicate(pil, self.cropSize)

        # Two photometric branches BEFORE spatial alignment
        pil_aug = self.augmentor(pil)

        # Synchronized spatial transforms
        pil = self.resize(pil)
        pil_aug = self.resize(pil_aug)

        crop = _SyncRandomCrop(self.cropSize)
        flip = _SyncRandomFlip()
        pil = flip(crop(pil))
        pil_aug = flip(crop(pil_aug))

        img_clean = self.to_tensor(pil)
        img_aug = self.to_tensor(pil_aug)

        return img_clean, img_aug, label


def get_paired_dataset(opt, augmentor: RobustAugmentor) -> Dataset:
    """Build a ConcatDataset across all sub-classes, mirroring the logic
    of data.get_dataset() but returning PairedImageFolder instances."""
    import os
    classes = os.listdir(opt.dataroot) if opt.classes == [""] else opt.classes
    if "0_real" not in classes or "1_fake" not in classes:
        dset_lst = []
        for cls in classes:
            root = os.path.join(opt.dataroot, cls)
            dset_lst.append(PairedImageFolder(opt, root, augmentor))
        return torch.utils.data.ConcatDataset(dset_lst)
    return PairedImageFolder(opt, opt.dataroot, augmentor)


def create_paired_dataloader(opt, augmentor: RobustAugmentor):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dset = get_paired_dataset(opt, augmentor)
    return torch.utils.data.DataLoader(
        dset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=int(opt.num_threads),
    )
