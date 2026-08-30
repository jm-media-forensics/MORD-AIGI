# -*- coding: utf-8 -*-
"""Paired robust training — prevents the racc-collapse failure mode.

Failure mode this script fixes
-------------------------------
With residual-based detectors and naive robust augmentation, the model
tends to learn the degenerate rule "weak residual ⇒ fake", because
heavily JPEG-compressed REAL images have residuals that look like FAKE
ones. The result: real-class accuracy collapses (racc → 0..20%) while
fake-class accuracy stays high. The model is "confident wrong" rather
than "uncertain".

How paired training fixes it
----------------------------
For every training image x with label y, we feed BOTH x_clean and x_aug
through the model in the SAME step, with the SAME label. The loss is:

    L = BCE(model(x_clean), y) + alpha * BCE(model(x_aug), y)
        + beta * consistency(feat_clean, feat_aug)            (optional)

The clean-loss term anchors the decision boundary in the un-degraded
distribution; the aug-loss term pulls degraded views toward the same
label. Because every batch always contains the clean version of every
sample, the model cannot "forget" what real residuals look like.

Example
-------
    python train_robust_paired.py \
        --name mord_sdv4_mild \
        --dataroot ./datasets/GenImage \
        --classes sdv4 \
        --batch_size 24 --lr 0.0002 --niter 50 --cropSize 224 \
        --aug_preset standard \
        --aug_weight 0.5

Note: paired training doubles the forward-pass cost, so consider
reducing --batch_size to ~70% of the value you used in train.py.
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from validate import validate
from data.datasets_paired import create_paired_dataloader
from data.robust_augment import build_augmentor
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger


# ---------------------------------------------------------------------------
# CLI: inherit TrainOptions, add paired & aug knobs
# ---------------------------------------------------------------------------
def _add_paired_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("paired robust training")
    g.add_argument("--aug_preset", type=str, default="standard",
                   choices=["mild", "cnnspot", "standard", "aggressive"])
    g.add_argument("--aug_weight", type=float, default=0.5,
                   help="symmetric weight of the augmented-branch loss "
                        "(used only when --aug_weight_real / --aug_weight_fake "
                        "are not specified)")
    g.add_argument("--aug_weight_real", type=float, default=None,
                   help="aug-branch loss weight for REAL samples (label=0). "
                        "If set, overrides --aug_weight for that class.")
    g.add_argument("--aug_weight_fake", type=float, default=None,
                   help="aug-branch loss weight for FAKE samples (label=1). "
                        "If set, overrides --aug_weight for that class.")
    g.add_argument("--consist_weight", type=float, default=0.0,
                   help="weight of feature-consistency loss (beta), "
                        "0 disables it")
    g.add_argument("--consist_real_only", action="store_true",
                   help="apply consistency loss only on real samples")
    g.add_argument("--aug_jpg_qual", type=int, nargs=2, default=None)
    g.add_argument("--aug_blur_sig", type=float, nargs=2, default=None)
    g.add_argument("--aug_resize_prob", type=float, default=None)
    return parser


class PairedTrainOptions(TrainOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser = _add_paired_args(parser)
        return parser


class PairedTestOptions(TestOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser = _add_paired_args(parser)
        return parser


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_torch(seed=100):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


# ---------------------------------------------------------------------------
# Cross-generator test config (same as train.py / train_robust.py)
# ---------------------------------------------------------------------------
TRAINING_MODE = "wild1"

if TRAINING_MODE == "gan":
    VALS = ["progan", "stylegan", "stylegan2", "biggan", "cyclegan",
            "stargan", "gaugan", "deepfake", "dalle"]
    MULTICLASS = [1, 1, 1, 0, 1, 0, 0, 0, 0]
    SAVE_FREQ = 1
elif TRAINING_MODE == "wild":
    VALS = ["facebook", "reddit", "twitter"]
    MULTICLASS = [0, 0, 0]
    SAVE_FREQ = 1
else:
    VALS = ["ADM", "biggan", "Glide", "Midjourney",
            "stable_diffusion_v_1_4", "stable_diffusion_v_1_5",
            "VQDM", "wukong"]
    MULTICLASS = [0, 0, 0, 0, 0, 0, 0, 0]
    SAVE_FREQ = 1


def get_val_opt():
    val_opt = PairedTrainOptions().parse(print_options=False)
    val_opt.dataroot = f"{val_opt.dataroot}/{val_opt.val_split}/"
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    return val_opt


# ---------------------------------------------------------------------------
# Paired optimization step — replaces Trainer.optimize_parameters() inline
# ---------------------------------------------------------------------------
def paired_optimize_step(model_wrapper, x_clean, x_aug, y,
                         aug_weight=0.5, consist_weight=0.0,
                         aug_weight_real=None, aug_weight_fake=None,
                         consist_real_only=False):
    """One optimization step over a paired batch.

    The current `networks/resnet.py` returns a [B,4] tensor where:
        out[:,0]  = branch-1 (BCPR) logit
        out[:,1]  = branch-2 (PiD) logit
        out[:,2]  = fused logit
        out[:,3]  = gate value

    Class-asymmetric aug weighting
    ------------------------------
    The failure mode "JPEG'd real residuals look like fake" pulls the
    decision boundary toward fake. Lowering the aug weight ONLY for real
    samples (aug_weight_real < aug_weight_fake) is the principled fix:
    the model still learns "degraded fake is fake" with full strength,
    but the conflicting "degraded real is real" signal is dialled down so
    it acts as a weak regularizer rather than a competing objective.
    """
    model = model_wrapper.model
    optimizer = model_wrapper.optimizer
    bce = model_wrapper.loss_fn  # BCEWithLogitsLoss

    # Resolve per-class aug weights
    w_real = aug_weight if aug_weight_real is None else aug_weight_real
    w_fake = aug_weight if aug_weight_fake is None else aug_weight_fake

    real_mask = (y < 0.5)
    fake_mask = ~real_mask

    optimizer.zero_grad()

    out_clean = model(x_clean)             # [B,3]
    out_aug = model(x_aug)                 # [B,3]
    # Model now returns [B,3]: col 0 = BCPR logit, col 1 = PiD logit,
    # col 2 = detached confidence-routed fused logit (no gradient).
    # All trainable losses must therefore be applied to cols 0 and 1.

    # ----- clean-branch loss: standard, full strength -----------------------
    def _sum_bce(out, target):
        # Only cols 0 and 1 have gradient. Col 2 is detached fused logit.
        return bce(out[:, 0], target) + bce(out[:, 1], target)

    loss_clean = _sum_bce(out_clean, y)

    # ----- aug-branch loss: per-class weighting -----------------------------
    # Compute per-class aug losses by masking. If a mask is empty in this
    # batch we skip that class to avoid NaN.
    n_real = int(real_mask.sum().item())
    n_fake = int(fake_mask.sum().item())

    loss_aug_real = torch.zeros((), device=x_clean.device)
    loss_aug_fake = torch.zeros((), device=x_clean.device)

    if n_real > 0 and w_real > 0:
        loss_aug_real = _sum_bce(out_aug[real_mask], y[real_mask])
    if n_fake > 0 and w_fake > 0:
        loss_aug_fake = _sum_bce(out_aug[fake_mask], y[fake_mask])

    loss_aug = w_real * loss_aug_real + w_fake * loss_aug_fake
    loss = loss_clean + loss_aug

    # ----- consistency: aug logit ≈ stop_grad(clean logit) ------------------
    # MUST target cols 0 and 1 (which have gradient). Targeting col 2
    # (the fused logit) was a no-op because that column is detached in
    # the model forward -- previous runs reported a non-zero consist value
    # but no gradient ever reached the encoder.
    if consist_weight > 0:
        with torch.no_grad():
            tgt0 = out_clean[:, 0].detach()
            tgt1 = out_clean[:, 1].detach()
        # Huber (smooth L1) replaces MSE: gradient is bounded for large
        # residuals, preventing the consist term from exploding when
        # clean/aug logits diverge during early training.
        if consist_real_only and n_real > 0:
            loss_consist = (F.smooth_l1_loss(out_aug[real_mask, 0], tgt0[real_mask])
                            + F.smooth_l1_loss(out_aug[real_mask, 1], tgt1[real_mask]))
        else:
            loss_consist = (F.smooth_l1_loss(out_aug[:, 0], tgt0)
                            + F.smooth_l1_loss(out_aug[:, 1], tgt1))
        loss = loss + consist_weight * loss_consist
    else:
        loss_consist = torch.zeros((), device=x_clean.device)

    loss.backward()
    # Gradient clipping: cap global grad-norm at 1.0 to prevent the
    # positive-feedback explosion observed when consist MSE dominates.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "loss_clean": float(loss_clean.item()),
        "loss_aug_real": float(loss_aug_real.item()),
        "loss_aug_fake": float(loss_aug_fake.item()),
        "loss_consist": float(loss_consist.item()),
        "n_real": n_real, "n_fake": n_fake,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    opt = PairedTrainOptions().parse()
    seed_torch(100)

    Testdataroot = os.path.join(opt.dataroot, "test")
    opt.dataroot = f"{opt.dataroot}/{opt.train_split}/"

    Logger(os.path.join(opt.checkpoints_dir, opt.name, "log.log"))
    print("  ".join(list(sys.argv)))

    val_opt = get_val_opt()
    Testopt = PairedTestOptions().parse(print_options=False)

    # --- build augmentor (with optional CLI overrides) ----------------------
    overrides = {}
    if opt.aug_jpg_qual is not None:
        overrides["jpg_qual_range"] = tuple(opt.aug_jpg_qual)
    if opt.aug_blur_sig is not None:
        overrides["blur_sig_range"] = tuple(opt.aug_blur_sig)
    if opt.aug_resize_prob is not None:
        overrides["resize_prob"] = opt.aug_resize_prob
    augmentor = build_augmentor(opt.aug_preset, **overrides)
    print(f"[paired-robust] augmentor: {augmentor}")
    print(f"[paired-robust] aug_weight = {opt.aug_weight}, "
          f"consist_weight = {opt.consist_weight}")

    # --- paired dataloader --------------------------------------------------
    data_loader = create_paired_dataloader(opt, augmentor)
    print(f"[paired-robust] dataset size = {len(data_loader.dataset)} "
          f"| batches per epoch = {len(data_loader)}")

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model_wrapper = Trainer(opt)

    # -----------------------------------------------------------------------
    # Eval routine (CLEAN test data — identical to train.py)
    # -----------------------------------------------------------------------
    def testmodel():
        print("*" * 25)
        accs, aps, raccs, faccs = [], [], [], []
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        for v_id, val in enumerate(VALS):
            Testopt.dataroot = f"{Testdataroot}/{val}"
            Testopt.classes = os.listdir(Testopt.dataroot) if MULTICLASS[v_id] else [""]
            Testopt.no_resize = False
            Testopt.no_crop = False
            Testopt.cropSize = 256
            acc, ap, racc, facc, _, _ = validate(model_wrapper.model, Testopt)
            accs.append(acc); aps.append(ap); raccs.append(racc); faccs.append(facc)
            print(f"({v_id} {val:10}) acc: {acc*100:.1f}; ap: {ap*100:.1f}; "
                  f"racc:{racc*100:.1f}; facc:{facc*100:.1f}")
        print(f"({len(VALS)} {'Mean':10}) acc: {np.array(accs).mean()*100:.1f}; "
              f"ap: {np.array(aps).mean()*100:.1f}; "
              f"racc:{np.array(raccs).mean()*100:.1f}; "
              f"facc:{np.array(faccs).mean()*100:.1f}")
        print("*" * 25)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    model_wrapper.eval()
    model_wrapper.train()
    print(f"cwd: {os.getcwd()}")

    save_freq = SAVE_FREQ
    for epoch in range(opt.last_epoch, opt.niter):
        if epoch > 30:
            save_freq = 1

        for i, batch in enumerate(data_loader):
            img_clean, img_aug, label = batch
            x_clean = img_clean.to(model_wrapper.device)
            x_aug = img_aug.to(model_wrapper.device)
            y = label.to(model_wrapper.device).float()

            model_wrapper.total_steps += 1
            stats = paired_optimize_step(
                model_wrapper, x_clean, x_aug, y,
                aug_weight=opt.aug_weight,
                aug_weight_real=opt.aug_weight_real,
                aug_weight_fake=opt.aug_weight_fake,
                consist_weight=opt.consist_weight,
                consist_real_only=opt.consist_real_only,
            )

            # Mirror Trainer state for compatibility with downstream printing
            model_wrapper.loss = stats["loss"]

            if model_wrapper.total_steps % opt.loss_freq == 0:
                ts = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                print(f"{ts} Train loss: {stats['loss']:.4f} "
                      f"(clean: {stats['loss_clean']:.4f}, "
                      f"aug_R: {stats['loss_aug_real']:.4f}, "
                      f"aug_F: {stats['loss_aug_fake']:.4f}, "
                      f"consist: {stats['loss_consist']:.4f}) "
                      f"[nR={stats['n_real']}, nF={stats['n_fake']}] "
                      f"step {model_wrapper.total_steps} lr {model_wrapper.lr}")
                train_writer.add_scalar("loss", stats["loss"], model_wrapper.total_steps)
                train_writer.add_scalar("loss_clean", stats["loss_clean"], model_wrapper.total_steps)
                train_writer.add_scalar("loss_aug_real", stats["loss_aug_real"], model_wrapper.total_steps)
                train_writer.add_scalar("loss_aug_fake", stats["loss_aug_fake"], model_wrapper.total_steps)

        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                  f"changing lr at the end of epoch {epoch}, "
                  f"iters {model_wrapper.total_steps}")
            model_wrapper.adjust_learning_rate()

        if epoch % save_freq == 0:
            model_wrapper.save_networks(epoch)

        # Validation on CLEAN val split
        if epoch % save_freq == 0:
            model_wrapper.eval()
            acc, ap = validate(model_wrapper.model, val_opt)[:2]
            val_writer.add_scalar("accuracy", acc, model_wrapper.total_steps)
            val_writer.add_scalar("ap", ap, model_wrapper.total_steps)
            print(f"(Val @ epoch {epoch}) acc: {acc}; ap: {ap}")

        if epoch % save_freq == 0:
            model_wrapper.eval()
            testmodel()
        model_wrapper.train()

    model_wrapper.eval()
    testmodel()
    model_wrapper.save_networks("last")


if __name__ == "__main__":
    main()
