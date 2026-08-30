# -*- coding: utf-8 -*-
"""
eval_mord_all.py
Unified MORD evaluation script.

Outputs all placeholder numbers needed for the paper:
  - Trilogy fusion (NP-additive, BMA, Routing) Acc / AP / AUC on SDv4 val
  - Cross-generator generalization on all test subdirs (GenImage)
  - Per-perturbation robustness (14 conditions x 3 branches + 3 fusion)
  - NP-additive failure AUC (reverse-predictor cases)
  - Complementarity stats (Pearson r, Jaccard, CMI)

Data layout assumed:
  {data_root}/
    train/stable_diffusion_v_1_4/{0_real,1_fake}
    val/stable_diffusion_v_1_4/{0_real,1_fake}        <- main eval
    test/{VQDM,ADM,SDv1.5,biggan,Glide,Midjourney,wukong}/{0_real,1_fake}

Usage:
  python eval_mord_all.py \
    --model_path ./checkpoints/mord_sdv4_mild/model_epoch_last.pth \
    --data_root  /mnt/data/genimage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4 \
    --output_dir ./results/mord_paired_mild
"""

import os
import io
import argparse
import json
import csv
import importlib
import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile, ImageFilter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------
def _jpeg(img, q):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def _blur(img, sigma):
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))

def _resize(img, factor):
    w, h = img.size
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    return img.resize((nw, nh), Image.BICUBIC).resize((w, h), Image.BICUBIC)

def _noise(img, sigma):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr = arr + np.random.normal(0.0, sigma * 255.0, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def _webp(img, q):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="WEBP", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def apply_perturbation(img, name):
    if name == "clean":
        return img
    kind, val = name.split("_", 1)
    val = float(val) if "." in val else int(val)
    if kind == "jpeg":   return _jpeg(img, val)
    if kind == "blur":   return _blur(img, val)
    if kind == "resize": return _resize(img, val)
    if kind == "noise":  return _noise(img, val)
    if kind == "webp":   return _webp(img, val)
    raise ValueError(f"unknown perturbation: {name}")


PERTURBATIONS = [
    "clean",
    "jpeg_95", "jpeg_85", "jpeg_75",
    "blur_0.5", "blur_1.0", "blur_2.0",
    "noise_0.01", "noise_0.02", "noise_0.05",
    "resize_0.5", "resize_0.7",
    "webp_95", "webp_85",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REAL_NAMES = {"real", "0", "0_real", "nature"}
FAKE_NAMES = {"fake", "1", "1_fake", "ai"}


def translate_duplicate(img, crop_size):
    if min(img.size) >= crop_size:
        return img

    width, height = img.size
    new_width = width * math.ceil(crop_size / width)
    new_height = height * math.ceil(crop_size / height)
    tiled = Image.new("RGB", (new_width, new_height))
    for x in range(0, new_width, width):
        for y in range(0, new_height, height):
            tiled.paste(img, (x, y))
    return tiled


class LabeledDataset(Dataset):
    def __init__(self, root, perturbation="clean", crop=256, max_per_class=None,
                 samples=None):
        self.perturbation = perturbation
        self.tf = T.Compose([
            T.Lambda(lambda img: translate_duplicate(img, crop)),
            T.Resize((crop, crop)),   # match the training pipeline (Resize -> crop)
            T.CenterCrop(crop),
            T.ToTensor(),
            # no Normalize — consistent with training pipeline
        ])
        # Explicit sample list (used for balanced cross-generator pooling).
        if samples is not None:
            self.samples = list(samples)
            return
        # Recursively walk the tree and label each image by the nearest
        # ancestor folder whose (lowercased) name is a known real/fake tag.
        # This handles both flat layouts ({root}/0_real, {root}/1_fake) and
        # nested ones ({root}/facebook/0_real, {root}/reddit/1_fake, ...).
        reals, fakes = [], []
        for dirpath, _dirs, filenames in os.walk(root):
            parts = [p.lower() for p in os.path.normpath(dirpath).split(os.sep)]
            lbl = None
            for p in reversed(parts):           # nearest tag wins
                if p in FAKE_NAMES:
                    lbl = 1; break
                if p in REAL_NAMES:
                    lbl = 0; break
            if lbl is None:
                continue
            bucket = fakes if lbl == 1 else reals
            for f in sorted(filenames):
                if os.path.splitext(f)[1].lower() in IMG_EXT:
                    bucket.append(os.path.join(dirpath, f))
        if max_per_class:
            reals = reals[:max_per_class]
            fakes = fakes[:max_per_class]
        self.samples = [(p, 0) for p in reals] + [(p, 1) for p in fakes]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = apply_perturbation(img, self.perturbation)
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.tf(img), lbl


# ---------------------------------------------------------------------------
# Logit collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    l0s, l1s, ys = [], [], []
    for imgs, labels in loader:
        out = model(imgs.to(device, non_blocking=True))
        l0s.append(out[:, 0].cpu().numpy())
        l1s.append(out[:, 1].cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(l0s), np.concatenate(l1s), np.concatenate(ys)


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

def fusion_scores(l0, l1):
    s_np  = l0 + l1
    p_bma = 0.5 * sigmoid(l0) + 0.5 * sigmoid(l1)
    use_l1 = np.abs(l1) > np.abs(l0)
    p_rt  = np.where(use_l1, sigmoid(l1), sigmoid(l0))
    return {
        "BCPR":    sigmoid(l0),
        "PiD":     sigmoid(l1),
        "NP":      sigmoid(s_np),
        "BMA":     p_bma,
        "Routing": p_rt,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(scores_dict, y):
    results = {}
    for name, scores in scores_dict.items():
        try:
            auc = roc_auc_score(y, scores)
        except Exception:
            auc = float("nan")
        try:
            ap = average_precision_score(y, scores)
        except Exception:
            ap = float("nan")
        acc = ((scores >= 0.5).astype(int) == y).mean()
        results[name] = {"AUC": auc, "AP": ap, "Acc": acc}
    return results


# ---------------------------------------------------------------------------
# Complementarity statistics  (与 analyze_complementarity.py 对齐)
# ---------------------------------------------------------------------------
def _best_tau(logits, labels, n=200):
    """在当前数据上搜索最优阈值"""
    best_t, best_acc = 0.0, 0.0
    for t in np.linspace(logits.min(), logits.max(), n):
        acc = np.mean((logits > t).astype(int) == labels)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


def complementarity_stats(l0, l1, y):
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mutual_info_score

    # 1. Pearson r — logit 相关性
    r, _ = pearsonr(l0, l1)

    # 2. Jaccard — 错误集合的交集占比（越低说明错误越互补）
    tau0 = _best_tau(l0, y)
    tau1 = _best_tau(l1, y)
    w0 = set(np.where((l0 > tau0).astype(int) != y)[0])
    w1 = set(np.where((l1 > tau1).astype(int) != y)[0])
    union = w0 | w1
    jaccard = len(w0 & w1) / max(len(union), 1)

    # 3. CMI — 线性回归残差法：I(PiD; Y | BCPR)
    #    先用 l0 线性预测 l1，残差代表 l1 中 l0 无法解释的成分
    #    再算残差和标签的互信息
    reg = LinearRegression().fit(l0.reshape(-1, 1), l1)
    res = l1 - reg.predict(l0.reshape(-1, 1))
    n_bins = 30
    edges  = np.percentile(res, np.linspace(0, 100, n_bins + 1))
    edges  = np.unique(edges)
    binned = np.digitize(res, edges[1:-1])
    cmi = mutual_info_score(binned, y)   # nats

    return {"pearson_r": float(r), "jaccard": float(jaccard), "cmi_nats": float(cmi)}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        sd = ckpt
    return {k.replace("module.", ""): v for k, v in sd.items()}


def _select_model_impl(model_impl, sd):
    return "current"


def _build_model(which):
    module_name = "networks.resnet"
    module = importlib.import_module(module_name)
    return module.resnet50(num_classes=1), module_name


def load_model(path, device, model_impl="auto"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = _extract_state_dict(ckpt)
    model_impl = _select_model_impl(model_impl, sd)
    model, module_name = _build_model(model_impl)
    msd = model.state_dict()
    matched = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
    missing = [k for k in msd if k not in matched]
    ignored = [k for k in sd if k not in matched]
    print(f"  Model impl : {model_impl} ({module_name})")
    print(f"  Checkpoint : {len(matched)} keys matched, {len(missing)} missing, {len(ignored)} ignored")
    if missing:
        print(f"  Missing keys (first 5): {missing[:5]}")
    if ignored:
        print(f"  Ignored keys (first 10): {ignored[:10]}")
    model.load_state_dict(matched, strict=False)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_metrics_table(results, title=""):
    if title:
        print(f"\n{'='*60}\n  {title}\n{'='*60}")
    print(f"{'Method':>10}  {'Acc':>6}  {'AP':>6}  {'AUC':>6}")
    print("-" * 38)
    for name, m in results.items():
        print(f"{name:>10}  {m['Acc']*100:>6.2f}  {m['AP']*100:>6.2f}  {m['AUC']*100:>6.2f}")

def fmt(v):
    return f"{v:.4f}" if not (v != v) else "nan"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_path",
                    default="./checkpoints/mord_sdv4_mild/model_epoch_last.pth")
    pa.add_argument("--data_root",
                    default="/mnt/data/genimage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4")
    pa.add_argument("--split", default="val", choices=["val", "test"],
                    help="val: use val/stable_diffusion_v_1_4/; test: use test/stable_diffusion_v_1_4/")
    pa.add_argument("--indist_dir", default="",
                    help="explicit path to the in-distribution clean eval set "
                         "(may contain 0_real/1_fake directly, or nested "
                         "platform/generator subdirs). Overrides the default "
                         "{data_root}/{split}/stable_diffusion_v_1_4.")
    pa.add_argument("--test_dir", default="",
                    help="explicit path to the cross-generator/cross-platform "
                         "test root whose subdirs are each evaluated separately. "
                         "Overrides the default {data_root}/test.")
    pa.add_argument("--crop",        type=int, default=256)
    pa.add_argument("--batch_size",  type=int, default=128)
    pa.add_argument("--num_workers", type=int, default=4)
    pa.add_argument("--max_per_class", type=int, default=None)
    pa.add_argument("--robust_pooled", action="store_true",
                    help="compute the robustness table pooled over ALL test "
                         "generators (matched cross-generator protocol, paper "
                         "Table 5/6). Default is the in-distribution SDv4 split.")
    pa.add_argument("--robust_cap", type=int, default=800,
                    help="images per class PER GENERATOR when --robust_pooled "
                         "is set (balanced pooling).")
    pa.add_argument("--output_dir",  default="./results/mord_all")
    pa.add_argument("--model_impl", default="auto",
                    choices=["auto", "current"],
                    help="which model definition to instantiate before loading the checkpoint")
    opt = pa.parse_args()

    os.makedirs(opt.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {opt.model_path}")

    model = load_model(opt.model_path, device, model_impl=opt.model_impl)

    def make_loader(root, pert="clean"):
        ds = LabeledDataset(root, perturbation=pert, crop=opt.crop,
                            max_per_class=opt.max_per_class)
        loader = DataLoader(ds, batch_size=opt.batch_size,
                            num_workers=opt.num_workers,
                            shuffle=False, pin_memory=True)
        return loader, len(ds)

    sdv4_val = opt.indist_dir or os.path.join(
        opt.data_root, opt.split, "stable_diffusion_v_1_4")
    test_root = opt.test_dir or os.path.join(opt.data_root, "test")

    # If the in-distribution dir does not exist, fall back to the test root
    # (e.g. WildRF has no SDv4 val split). This keeps step 1 working on
    # arbitrary datasets without a fixed val-subdir name.
    if not os.path.isdir(sdv4_val):
        print(f"  [warn] in-distribution dir not found: {sdv4_val}")
        if os.path.isdir(test_root):
            print(f"  [warn] falling back to test_root for step 1: {test_root}")
            sdv4_val = test_root
        else:
            raise FileNotFoundError(
                f"Neither indist_dir ({sdv4_val}) nor test_root ({test_root}) "
                f"exists. Pass --indist_dir / --test_dir explicitly.")

    # -----------------------------------------------------------------------
    # 1. In-distribution clean — main result + complementarity
    # -----------------------------------------------------------------------
    print(f"\n[1/4] In-distribution clean eval: {sdv4_val}")
    loader, n = make_loader(sdv4_val, "clean")
    print(f"  Dataset size: {n}")
    l0, l1, y = collect_logits(model, loader, device)

    sc = fusion_scores(l0, l1)
    metrics_clean = compute_metrics(sc, y)
    print_metrics_table(metrics_clean, "SDv4 val — clean")

    # 互补性统计用 test/（跨生成器混合，错误率更高，估计更稳定）
    print(f"\n  Computing complementarity stats on test set (cross-generator)...")
    comp_l0s, comp_l1s, comp_ys = [], [], []
    if os.path.isdir(test_root):
        for gen in sorted(os.listdir(test_root)):
            gpath = os.path.join(test_root, gen)
            if not os.path.isdir(gpath):
                continue
            ldr, ng = make_loader(gpath, "clean")
            if ng == 0:
                continue
            gl0, gl1, gy = collect_logits(model, ldr, device)
            comp_l0s.append(gl0); comp_l1s.append(gl1); comp_ys.append(gy)
    if comp_l0s:
        cl0 = np.concatenate(comp_l0s)
        cl1 = np.concatenate(comp_l1s)
        cy  = np.concatenate(comp_ys)
        comp = complementarity_stats(cl0, cl1, cy)
        print(f"  Test set size: {len(cy)}")
    else:
        comp = complementarity_stats(l0, l1, y)   # fallback to val
        print(f"  (fallback to val set)")
    print(f"    Pearson r = {comp['pearson_r']:.4f}")
    print(f"    Jaccard   = {comp['jaccard']:.4f}  (错误集合，越低越互补)")
    print(f"    CMI       = {comp['cmi_nats']:.4f} nats")

    # -----------------------------------------------------------------------
    # 2. Cross-generator generalization (test set)
    # -----------------------------------------------------------------------
    print(f"\n[2/4] Cross-generator generalization")
    cross_results = {}
    if os.path.isdir(test_root):
        generators = sorted(os.listdir(test_root))
        print(f"  Found {len(generators)} generators: {generators}")
        print(f"\n  {'Generator':>20}  {'n':>5}  {'BCPR':>5}  {'PiD':>5}  {'NP':>5}  {'BMA':>5}  {'Route':>5}")
        print("  " + "-" * 65)
        for gen in generators:
            gpath = os.path.join(test_root, gen)
            if not os.path.isdir(gpath):
                continue
            ldr, ng = make_loader(gpath, "clean")
            if ng == 0:
                continue
            lg0, lg1, yg = collect_logits(model, ldr, device)
            scg = fusion_scores(lg0, lg1)
            mg = compute_metrics(scg, yg)
            cross_results[gen] = mg
            print(f"  {gen:>20}  {ng:>5}  "
                  f"{mg['BCPR']['AUC']*100:>5.1f}  "
                  f"{mg['PiD']['AUC']*100:>5.1f}  "
                  f"{mg['NP']['AUC']*100:>5.1f}  "
                  f"{mg['BMA']['AUC']*100:>5.1f}  "
                  f"{mg['Routing']['AUC']*100:>5.1f}")
    else:
        print(f"  test_root not found: {test_root}")

    # -----------------------------------------------------------------------
    # 3. Robustness — 14 perturbations on SDv4 val
    # -----------------------------------------------------------------------
    # Build a balanced cross-generator pool if requested (matched protocol,
    # paper Table 5/6): sample --robust_cap images per class PER GENERATOR.
    pooled_items = None
    if opt.robust_pooled:
        if not os.path.isdir(test_root):
            raise FileNotFoundError(f"--robust_pooled needs test_root: {test_root}")
        reals, fakes = [], []
        for gen in sorted(os.listdir(test_root)):
            gp = os.path.join(test_root, gen)
            if not os.path.isdir(gp):
                continue
            for lbl, sub in ((0, "0_real"), (1, "1_fake")):
                d = os.path.join(gp, sub)
                if not os.path.isdir(d):
                    continue
                fs = [f for f in sorted(os.listdir(d))
                      if os.path.splitext(f)[1].lower() in IMG_EXT]
                if opt.robust_cap:
                    fs = fs[:opt.robust_cap]
                (fakes if lbl == 1 else reals).extend(os.path.join(d, f) for f in fs)
        pooled_items = [(p, 0) for p in reals] + [(p, 1) for p in fakes]
        print(f"\n[3/4] Robustness (pooled over all generators, "
              f"{opt.robust_cap}/class/gen; n={len(pooled_items)})")
    else:
        print(f"\n[3/4] Robustness (14 perturbations on SDv4 val)")
    print(f"  {'Perturbation':>14}  {'BCPR':>5}  {'PiD':>5}  {'NP':>5}  {'BMA':>5}  {'Route':>5}  | NP-fail?")
    print("  " + "-" * 68)

    def _robust_loader(pert):
        if pooled_items is not None:
            ds = LabeledDataset(None, perturbation=pert, crop=opt.crop,
                                samples=pooled_items)
            return DataLoader(ds, batch_size=opt.batch_size,
                              num_workers=opt.num_workers, shuffle=False,
                              pin_memory=True)
        return make_loader(sdv4_val, pert)[0]

    rob_results = {}
    for pert in PERTURBATIONS:
        if pert == "clean" and pooled_items is None:
            lp0, lp1, yp = l0, l1, y
        else:
            ldr = _robust_loader(pert)
            lp0, lp1, yp = collect_logits(model, ldr, device)

        sc_p = fusion_scores(lp0, lp1)
        m_p  = compute_metrics(sc_p, yp)
        rob_results[pert] = m_p

        np_fail = "FAIL" if m_p["NP"]["AUC"] < 0.50 else ""
        print(f"  {pert:>14}  "
              f"{m_p['BCPR']['AUC']*100:>5.1f}  "
              f"{m_p['PiD']['AUC']*100:>5.1f}  "
              f"{m_p['NP']['AUC']*100:>5.1f}  "
              f"{m_p['BMA']['AUC']*100:>5.1f}  "
              f"{m_p['Routing']['AUC']*100:>5.1f}  | {np_fail}")

    # -----------------------------------------------------------------------
    # 4. Reverse-predictor failure summary
    # -----------------------------------------------------------------------
    print(f"\n[4/4] NP-additive failure cases (AUC < 0.50)")
    np_fail_rows = [(p, rob_results[p]) for p in PERTURBATIONS
                    if rob_results[p]["NP"]["AUC"] < 0.50]
    if np_fail_rows:
        for p, m in np_fail_rows:
            print(f"  {p}: NP={m['NP']['AUC']:.4f}  BCPR={m['BCPR']['AUC']:.4f}  "
                  f"PiD={m['PiD']['AUC']:.4f}  Routing={m['Routing']['AUC']:.4f}")
    else:
        print("  None — NP-additive did not drop below 0.50")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    methods = ["BCPR", "PiD", "NP", "BMA", "Routing"]
    metrics  = ["Acc", "AP", "AUC"]

    summary = {
        "model": opt.model_path,
        "sdv4_clean": {k: {m: float(v) for m, v in mv.items()}
                       for k, mv in metrics_clean.items()},
        "complementarity": comp,
        "cross_generator": {
            g: {k: {m: float(v) for m, v in mv.items()}
                for k, mv in gr.items()}
            for g, gr in cross_results.items()
        },
        "robustness": {
            p: {k: {m: float(v) for m, v in mv.items()}
                for k, mv in pr.items()}
            for p, pr in rob_results.items()
        },
    }
    with open(os.path.join(opt.output_dir, "mord_all_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Robustness CSV
    with open(os.path.join(opt.output_dir, "robustness_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["perturbation"] + [f"{m}_{mt}" for m in methods for mt in metrics])
        for p, pr in rob_results.items():
            w.writerow([p] + [fmt(pr[m][mt]) for m in methods for mt in metrics])

    # Cross-generator CSV
    with open(os.path.join(opt.output_dir, "crossgen_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generator"] + [f"{m}_{mt}" for m in methods for mt in metrics])
        for g, gr in cross_results.items():
            w.writerow([g] + [fmt(gr[m][mt]) for m in methods for mt in metrics])

    # Markdown for paper copy-paste
    md_path = os.path.join(opt.output_dir, "paper_numbers.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# MORD Paper Numbers\n\n")
        f.write(f"Model: `{opt.model_path}`\n\n")

        f.write("## SDv4 Val — Clean (main table)\n\n")
        f.write("| Method | Acc | AP | AUC |\n|---|---:|---:|---:|\n")
        for name, m in metrics_clean.items():
            f.write(f"| {name} | {m['Acc']*100:.2f} | {m['AP']*100:.2f} | {m['AUC']*100:.2f} |\n")

        f.write("\n## Complementarity Stats\n\n")
        f.write(f"- Pearson r = {comp['pearson_r']:.4f}\n")
        f.write(f"- Jaccard   = {comp['jaccard']:.4f}\n")
        f.write(f"- CMI       = {comp['cmi_nats']:.4f} nats\n")

        f.write("\n## Cross-Generator Generalization (AUC)\n\n")
        f.write("| Generator | BCPR | PiD | NP | BMA | Routing |\n|---|---:|---:|---:|---:|---:|\n")
        for g, gr in cross_results.items():
            f.write(f"| {g} | "
                    f"{gr['BCPR']['AUC']*100:.2f} | "
                    f"{gr['PiD']['AUC']*100:.2f} | "
                    f"{gr['NP']['AUC']*100:.2f} | "
                    f"{gr['BMA']['AUC']*100:.2f} | "
                    f"{gr['Routing']['AUC']*100:.2f} |\n")

        f.write("\n## Robustness (AUC)\n\n")
        f.write("| Perturbation | BCPR | PiD | NP | BMA | Routing |\n|---|---:|---:|---:|---:|---:|\n")
        for p, pr in rob_results.items():
            f.write(f"| {p} | "
                    f"{pr['BCPR']['AUC']*100:.2f} | "
                    f"{pr['PiD']['AUC']*100:.2f} | "
                    f"{pr['NP']['AUC']*100:.2f} | "
                    f"{pr['BMA']['AUC']*100:.2f} | "
                    f"{pr['Routing']['AUC']*100:.2f} |\n")

        f.write("\n## Robustness (Acc)\n\n")
        f.write("| Perturbation | BCPR | PiD | NP | BMA | Routing |\n|---|---:|---:|---:|---:|---:|\n")
        for p, pr in rob_results.items():
            f.write(f"| {p} | "
                    f"{pr['BCPR']['Acc']*100:.2f} | "
                    f"{pr['PiD']['Acc']*100:.2f} | "
                    f"{pr['NP']['Acc']*100:.2f} | "
                    f"{pr['BMA']['Acc']*100:.2f} | "
                    f"{pr['Routing']['Acc']*100:.2f} |\n")

        if np_fail_rows:
            f.write("\n## NP-Additive Failure Cases\n\n")
            for p, m in np_fail_rows:
                f.write(f"- `{p}`: NP={m['NP']['AUC']:.4f}, "
                        f"BCPR={m['BCPR']['AUC']:.4f}, "
                        f"PiD={m['PiD']['AUC']:.4f}, "
                        f"Routing={m['Routing']['AUC']:.4f}\n")

    print(f"\nOutputs saved to: {opt.output_dir}")
    print(f"  mord_all_results.json")
    print(f"  robustness_table.csv")
    print(f"  crossgen_table.csv")
    print(f"  paper_numbers.md")


if __name__ == "__main__":
    main()
