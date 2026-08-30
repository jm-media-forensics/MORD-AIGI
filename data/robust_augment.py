# -*- coding: utf-8 -*-
"""Robust training augmentations (CNNSpot-style).

This module implements the standard "robust augmentation" protocol used in
AIGC / DeepFake detection papers (Wang et al., CNNSpot 2020 onward):
random JPEG compression with quality sampled in a range, random Gaussian
blur, random resize, and optionally additive Gaussian noise.

The intent is that all robustness perturbations seen at TEST time should
have been seen DURING TRAINING (under varying parameters), but each test
sample's specific perturbation is unknown to the model — i.e. the model
learns invariance via stochastic exposure, not via matched-pair training.

Usage:
    from data.robust_augment import RobustAugmentor
    aug = RobustAugmentor(
        jpg_prob=0.5, jpg_qual_range=(30, 100), jpg_methods=("pil", "cv2"),
        blur_prob=0.5, blur_sig_range=(0.0, 3.0),
        resize_prob=0.3, resize_scale_range=(0.5, 1.5),
        noise_prob=0.0, noise_sigma_range=(0.0, 0.05),
    )
    pil_img_aug = aug(pil_img)
"""

from io import BytesIO
from random import random, uniform, choice

import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# JPEG implementations (two libraries so the model sees both decoders)
# ---------------------------------------------------------------------------
def _pil_jpeg(arr: np.ndarray, quality: int) -> np.ndarray:
    """Encode/decode through PIL libjpeg."""
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
    out = np.array(Image.open(buf))
    buf.close()
    return out


def _cv2_jpeg(arr: np.ndarray, quality: int) -> np.ndarray:
    """Encode/decode through OpenCV libjpeg-turbo."""
    bgr = arr[:, :, ::-1]
    enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode(".jpg", bgr, enc_param)
    if not ok:
        return arr
    dec = cv2.imdecode(enc, 1)
    return dec[:, :, ::-1]


_JPEG_FNS = {"pil": _pil_jpeg, "cv2": _cv2_jpeg}


# ---------------------------------------------------------------------------
# Gaussian blur (channel-wise, in place to avoid extra allocations)
# ---------------------------------------------------------------------------
def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    out = arr.copy()
    for c in range(out.shape[2]):
        gaussian_filter(out[:, :, c], output=out[:, :, c], sigma=sigma)
    return out


# ---------------------------------------------------------------------------
# Random resize (down then up to original size, simulating platform resizing)
# ---------------------------------------------------------------------------
_INTERP_DOWN = [cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]
_INTERP_UP = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def _random_resize(arr: np.ndarray, scale: float) -> np.ndarray:
    """Resize to scale, then back to original size with possibly different
    interpolation. Simulates social-platform downscale-upscale pipelines."""
    h, w = arr.shape[:2]
    new_h = max(8, int(round(h * scale)))
    new_w = max(8, int(round(w * scale)))
    down = cv2.resize(arr, (new_w, new_h), interpolation=choice(_INTERP_DOWN))
    up = cv2.resize(down, (w, h), interpolation=choice(_INTERP_UP))
    return up


# ---------------------------------------------------------------------------
# Additive Gaussian noise (sigma is in [0,1] image space)
# ---------------------------------------------------------------------------
def _gaussian_noise(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    noise = np.random.normal(0.0, sigma * 255.0, arr.shape).astype(np.float32)
    out = arr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main composite augmentor (callable on PIL.Image, returns PIL.Image)
# ---------------------------------------------------------------------------
class RobustAugmentor:
    """Composite stochastic augmentation matching the CNNSpot-style protocol.

    Each transform is applied independently with its own probability. All
    parameters are sampled per-call from their ranges so the model sees a
    continuous distribution of degradations rather than fixed bins.

    Parameters
    ----------
    jpg_prob          : float, probability of applying JPEG compression.
    jpg_qual_range    : tuple(int,int), uniform-int sampling range for QF.
    jpg_methods       : tuple of str in {"pil","cv2"}, library randomly chosen.
    blur_prob         : float, probability of applying Gaussian blur.
    blur_sig_range    : tuple(float,float), uniform-float sampling range for sigma.
    resize_prob       : float, probability of random down-up resize.
    resize_scale_range: tuple(float,float), uniform-float sampling range for scale.
    noise_prob        : float, probability of additive Gaussian noise.
    noise_sigma_range : tuple(float,float), sigma in normalized image space.
    order_randomize   : bool, if True permute the order of operations per call.
    """

    def __init__(
        self,
        jpg_prob=0.5,
        jpg_qual_range=(30, 100),
        jpg_methods=("pil", "cv2"),
        blur_prob=0.5,
        blur_sig_range=(0.0, 3.0),
        resize_prob=0.3,
        resize_scale_range=(0.5, 1.5),
        noise_prob=0.0,
        noise_sigma_range=(0.0, 0.05),
        order_randomize=False,
    ):
        self.jpg_prob = jpg_prob
        self.jpg_qual_range = tuple(jpg_qual_range)
        self.jpg_methods = tuple(jpg_methods)
        self.blur_prob = blur_prob
        self.blur_sig_range = tuple(blur_sig_range)
        self.resize_prob = resize_prob
        self.resize_scale_range = tuple(resize_scale_range)
        self.noise_prob = noise_prob
        self.noise_sigma_range = tuple(noise_sigma_range)
        self.order_randomize = order_randomize

    def __repr__(self):
        return (
            "RobustAugmentor("
            f"jpg={self.jpg_prob}@{self.jpg_qual_range}/{self.jpg_methods}, "
            f"blur={self.blur_prob}@{self.blur_sig_range}, "
            f"resize={self.resize_prob}@{self.resize_scale_range}, "
            f"noise={self.noise_prob}@{self.noise_sigma_range})"
        )

    # --- individual operators -------------------------------------------------
    def _maybe_blur(self, arr):
        if random() < self.blur_prob:
            sig = uniform(*self.blur_sig_range)
            return _gaussian_blur(arr, sig)
        return arr

    def _maybe_jpeg(self, arr):
        if random() < self.jpg_prob:
            lo, hi = self.jpg_qual_range
            q = int(round(uniform(lo, hi)))
            method = choice(self.jpg_methods)
            return _JPEG_FNS[method](arr, q)
        return arr

    def _maybe_resize(self, arr):
        if random() < self.resize_prob:
            s = uniform(*self.resize_scale_range)
            return _random_resize(arr, s)
        return arr

    def _maybe_noise(self, arr):
        if random() < self.noise_prob:
            sig = uniform(*self.noise_sigma_range)
            return _gaussian_noise(arr, sig)
        return arr

    # --- entry point ----------------------------------------------------------
    def __call__(self, pil_img: Image.Image) -> Image.Image:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img)

        ops = [self._maybe_blur, self._maybe_noise,
               self._maybe_resize, self._maybe_jpeg]
        if self.order_randomize:
            # Note: JPEG is conventionally applied LAST since it represents
            # the final encoding step in real distribution pipelines.
            jpeg_op = ops[-1]
            other = ops[:-1]
            # shuffle the non-jpeg ops, keep jpeg last
            from random import shuffle
            shuffle(other)
            ops = other + [jpeg_op]

        for op in ops:
            arr = op(arr)

        return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------
PRESETS = {
    # CNNSpot original — JPEG + blur only
    "cnnspot": dict(
        jpg_prob=0.5, jpg_qual_range=(30, 100), jpg_methods=("pil", "cv2"),
        blur_prob=0.5, blur_sig_range=(0.0, 3.0),
        resize_prob=0.0, noise_prob=0.0,
    ),
    # Standard robust — adds resize
    "standard": dict(
        jpg_prob=0.5, jpg_qual_range=(50, 100), jpg_methods=("pil", "cv2"),
        blur_prob=0.5, blur_sig_range=(0.0, 2.0),
        resize_prob=0.3, resize_scale_range=(0.5, 1.5),
        noise_prob=0.0,
    ),
    # Aggressive — wider ranges + noise, for "in-the-wild" deployment
    "aggressive": dict(
        jpg_prob=0.7, jpg_qual_range=(30, 100), jpg_methods=("pil", "cv2"),
        blur_prob=0.5, blur_sig_range=(0.0, 3.0),
        resize_prob=0.5, resize_scale_range=(0.4, 1.6),
        noise_prob=0.2, noise_sigma_range=(0.0, 0.04),
    ),
    # Mild — closer to clean, useful as a control / ablation
    "mild": dict(
        jpg_prob=0.3, jpg_qual_range=(70, 100), jpg_methods=("pil",),
        blur_prob=0.3, blur_sig_range=(0.0, 1.0),
        resize_prob=0.0, noise_prob=0.0,
    ),
}


def build_augmentor(preset: str = "standard", **overrides) -> RobustAugmentor:
    """Build an augmentor from a preset name, with optional overrides.

    Example
    -------
    >>> aug = build_augmentor("standard", noise_prob=0.1)
    """
    if preset not in PRESETS:
        raise KeyError(
            f"Unknown preset '{preset}'. Available: {list(PRESETS)}"
        )
    cfg = dict(PRESETS[preset])
    cfg.update(overrides)
    return RobustAugmentor(**cfg)
