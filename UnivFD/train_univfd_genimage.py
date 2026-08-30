"""
train_univfd_genimage.py  —  Train UnivFD (CLIP FC head) on GenImage / UFD-format data.

IMPORTANT — preprocessing must match eval_univfd.py / official UnivFD:
    Resize(224, BICUBIC) -> CenterCrop(224) -> ToTensor -> Normalize(CLIP_MEAN, CLIP_STD)
We deliberately DO NOT reuse the parent project's MORD data loader: that pipeline
is built for the ResNet two-branch model and (a) resizes to a 256x256 square and
(b) has Normalize commented out. Feeding un-normalized images to CLIP's
encode_image produces garbage features and a train/eval mismatch — the main reason
直接测 GenImage 效果差.

UnivFD = frozen CLIP ViT-L/14 image encoder (768-d) + a single Linear(768 -> 1).
Only the FC head is trained.

Data layout (paths classified by substring, works at any nesting depth):
    <data_root>/
        train/  ... files whose path contains '0_real' -> real, '1_fake' -> fake
        val/    ...
        test/   <generator>/ ...   (optional, one folder per generator)

Hyper-parameters follow Ojha et al., CVPR 2023:
    Optimizer       Adam, lr=1e-3, weight_decay=0
    Batch size      64
    Augmentation    train: RandomCrop(224) + RandomHorizontalFlip
                    val/test: CenterCrop(224)
    LR decay        x0.1 on early-stop trigger (patience=5 on val-acc)

Usage:
    python UnivFD/train_univfd_genimage.py \
        --data_root /data/genimage_ufd \
        --output_dir ./checkpoints/univfd_genimage \
        --batch_size 64 --lr 1e-3 --epochs 100 --patience 5 --num_workers 4
"""

import os
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, average_precision_score

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import clip
except ImportError as e:
    raise ImportError(
        "openai-clip required: pip install git+https://github.com/openai/CLIP.git"
    ) from e

# ── CLIP constants (must match eval_univfd.py) ────────────────────────────────
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
CLIP_DIM  = {"ViT-L/14": 768, "RN50": 1024}
IMG_EXT   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEED = 42


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ── data ──────────────────────────────────────────────────────────────────────
def collect_files(root_dir):
    out = []
    for r, _, fs in os.walk(root_dir):
        for f in fs:
            if Path(f).suffix.lower() in IMG_EXT:
                out.append(os.path.join(r, f))
    return out


def build_list(split_dir):
    """Recursive scan; classify by '0_real'/'1_fake' in path."""
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split dir not found: {split_dir}")
    files = collect_files(split_dir)
    real  = [p for p in files if "0_real" in p]
    fake  = [p for p in files if "1_fake" in p]
    skip  = len(files) - len(real) - len(fake)
    print(f"  {split_dir}: real={len(real)} fake={len(fake)}"
          + (f" skipped={skip}" if skip else ""))
    return real, fake


def clip_transform(is_train):
    if is_train:
        return T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


class RealFakeDataset(Dataset):
    def __init__(self, real_list, fake_list, is_train, max_sample=None):
        if max_sample is not None:
            random.shuffle(real_list); random.shuffle(fake_list)
            real_list = real_list[:max_sample]; fake_list = fake_list[:max_sample]
        self.samples = [(p, 0) for p in real_list] + [(p, 1) for p in fake_list]
        random.shuffle(self.samples)
        self.tf = clip_transform(is_train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.tf(img), label


# ── model ─────────────────────────────────────────────────────────────────────
class UnivFD(nn.Module):
    def __init__(self, backbone="ViT-L/14", device="cpu"):
        super().__init__()
        self.clip_model, _ = clip.load(backbone, device=device)
        for p in self.clip_model.parameters():
            p.requires_grad_(False)
        self.fc = nn.Linear(CLIP_DIM[backbone], 1)

    def forward(self, x):
        with torch.no_grad():
            feats = self.clip_model.encode_image(x).float()
        return self.fc(feats)


class EarlyStopping:
    def __init__(self, patience, delta=1e-4):
        self.patience = patience; self.delta = delta
        self.counter = 0; self.best = None; self.stop = False

    def __call__(self, metric):
        if self.best is None or metric > self.best + self.delta:
            self.best = metric; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ── train / eval ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, log_freq, epoch):
    model.train()
    total = 0.0
    for i, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(imgs).squeeze(1), labels)
        loss.backward(); optimizer.step()
        total += loss.item() * imgs.size(0)
        if log_freq and i % log_freq == 0:
            print(f"  [ep{epoch} it{i}/{len(loader)}] loss={loss.item():.4f}")
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for imgs, labels in loader:
        p = torch.sigmoid(model(imgs.to(device)).squeeze(1)).cpu().numpy()
        ps.extend(p.tolist()); ys.extend(labels.numpy().tolist())
    y = np.array(ys); p = np.array(ps)
    acc   = accuracy_score(y, p > 0.5)
    ap    = average_precision_score(y, p)
    r_acc = accuracy_score(y[y == 0], p[y == 0] > 0.5) if (y == 0).any() else 0.0
    f_acc = accuracy_score(y[y == 1], p[y == 1] > 0.5) if (y == 1).any() else 0.0
    return acc, ap, r_acc, f_acc


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--data_root", required=True,
                    help="Root with train/ val/ (and optional test/) splits")
    pa.add_argument("--train_split", default="train")
    pa.add_argument("--val_split",   default="val")
    pa.add_argument("--test_split",  default="test")
    pa.add_argument("--max_sample",  type=int, default=None)
    pa.add_argument("--output_dir",  default="./checkpoints/univfd_genimage")
    pa.add_argument("--backbone",    default="ViT-L/14", choices=list(CLIP_DIM))
    pa.add_argument("--epochs",      type=int,   default=100)
    pa.add_argument("--batch_size",  type=int,   default=64)
    pa.add_argument("--lr",          type=float, default=1e-3)
    pa.add_argument("--weight_decay",type=float, default=0.0)
    pa.add_argument("--patience",    type=int,   default=5)
    pa.add_argument("--lr_drops",    type=int,   default=1)
    pa.add_argument("--num_workers", type=int,   default=4)
    pa.add_argument("--log_freq",    type=int,   default=200)
    pa.add_argument("--seed",        type=int,   default=SEED)
    opt = pa.parse_args()

    set_seed(opt.seed)
    os.makedirs(opt.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}  Backbone=CLIP {opt.backbone}  Out={opt.output_dir}")

    # data
    print("Building lists...")
    tr_real, tr_fake = build_list(os.path.join(opt.data_root, opt.train_split))
    va_real, va_fake = build_list(os.path.join(opt.data_root, opt.val_split))
    train_ds = RealFakeDataset(tr_real, tr_fake, True,  opt.max_sample)
    val_ds   = RealFakeDataset(va_real, va_fake, False, opt.max_sample)
    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False,
                              num_workers=opt.num_workers, pin_memory=True)
    print(f"Train={len(train_ds)}  Val={len(val_ds)}")

    # optional per-generator test set
    test_root = os.path.join(opt.data_root, opt.test_split)
    test_gens = {}
    if os.path.isdir(test_root):
        for g in sorted(os.listdir(test_root)):
            gdir = os.path.join(test_root, g)
            if os.path.isdir(gdir):
                gr, gf = build_list(gdir)
                if gr or gf:
                    ds = RealFakeDataset(gr, gf, False, opt.max_sample)
                    test_gens[g] = DataLoader(ds, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=opt.num_workers)

    # model / optim
    model = UnivFD(backbone=opt.backbone, device=str(device)).to(device)
    print(f"Trainable params (FC): {sum(p.numel() for p in model.fc.parameters()):,}")
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    early = EarlyStopping(patience=opt.patience)
    lr_drops = 0
    best_acc = 0.0
    best_ckpt = os.path.join(opt.output_dir, "fc_best.pth")
    last_ckpt = os.path.join(opt.output_dir, "fc_last.pth")
    log_csv   = os.path.join(opt.output_dir, "train_log.csv")
    with open(log_csv, "w") as f:
        f.write("epoch,train_loss,val_acc,val_ap,val_r_acc,val_f_acc,lr\n")

    t0 = time.time()
    for epoch in range(1, opt.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(model, train_loader, optimizer, criterion,
                               device, opt.log_freq, epoch)
        acc, ap, r_acc, f_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch:03d}] loss={loss:.4f} val_acc={acc*100:.2f}% "
              f"AP={ap*100:.2f}% r={r_acc*100:.1f}% f={f_acc*100:.1f}% "
              f"lr={lr_now:.2e} t={(time.time()-t0)/60:.1f}m")
        with open(log_csv, "a") as f:
            f.write(f"{epoch},{loss:.6f},{acc:.6f},{ap:.6f},{r_acc:.6f},{f_acc:.6f},{lr_now:.2e}\n")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.fc.state_dict(), best_ckpt)
            print(f"  -> best={best_acc*100:.2f}% saved {best_ckpt}")

        early(acc)
        if early.stop:
            if lr_drops < opt.lr_drops:
                lr_drops += 1
                for g in optimizer.param_groups:
                    g["lr"] *= 0.1
                print(f"  -> LR drop #{lr_drops} -> {optimizer.param_groups[0]['lr']:.2e}")
                early = EarlyStopping(patience=opt.patience)
            else:
                print("  -> Early stopping."); break

    torch.save(model.fc.state_dict(), last_ckpt)

    # final per-generator test
    if test_gens:
        print("\n== Per-generator test (best ckpt) ==")
        model.fc.load_state_dict(torch.load(best_ckpt, map_location=device))
        accs, aps = [], []
        for g, loader in test_gens.items():
            acc, ap, r_acc, f_acc = evaluate(model, loader, device)
            accs.append(acc); aps.append(ap)
            print(f"  {g:22s} acc={acc*100:.1f} ap={ap*100:.1f} "
                  f"r={r_acc*100:.1f} f={f_acc*100:.1f}")
        print(f"  {'Mean':22s} acc={np.mean(accs)*100:.1f} ap={np.mean(aps)*100:.1f}")

    print(f"\nDone. best val_acc={best_acc*100:.2f}%")
    print(f"Best FC: {best_ckpt}\nLast FC: {last_ckpt}\nLog: {log_csv}")


if __name__ == "__main__":
    main()
