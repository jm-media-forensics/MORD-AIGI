# MORD: Multi-Order Statistical Residuals for AI-Generated Image Detection

Reference implementation for **"Multi-Order Statistical Residuals for AI-Generated Image
Detection" (MORD)**. MORD combines a first-order pixel residual branch (**PiD**) with a
second-order **Block-wise Color-Correlation PCA Residual (BCPR)** and fuses the two at the
decision level (probability averaging by default).

## Install

```bash
pip install -r requirements.txt
```
Input RGB is scaled to **[0, 1]** (`ToTensor`) with **no additional normalization** — do
not add a `Normalize` transform. Data layout: each class folder contains `0_real/` and
`1_fake/` images; train on the GenImage **SDv1.4** split, test on the other generators.

## Reproduce the robustness results

**Train:**
```bash
python train_robust_paired.py --name mord_sdv4_mild \
    --dataroot /path/to/GenImage --classes stable_diffusion_v_1_4 \
    --batch_size 32 --lr 0.0004 --niter 50 --cropSize 224 --aug_preset mild
```

**Evaluate** — the paper's robustness table (Table 5/6, Fig 5) is the **matched
cross-generator** protocol: pass `--robust_pooled` to pool each perturbation over all test
generators.
```bash
python eval_mord_all.py --model_path checkpoints/mord_sdv4_mild.pth \
    --data_root /path/to/GenImage --robust_pooled --robust_cap 800 \
    --output_dir results/mord_mild
```
Omit `--robust_pooled` to instead measure in-distribution robustness on the SDv1.4
validation split. The robustness `BMA` column is the default MORD number; per-branch
(`PiD`, `BCPR`) and per-fusion (`NP`/`BMA`/`Route`) Acc and AP are written to
`results/mord_mild/robustness_table.csv`.

A pretrained checkpoint (`checkpoints/mord_sdv4_mild.pth`) is provided as a GitHub Release
asset.

**Baselines** (same protocol):
```bash
python NPR/train_npr_paired.py --name npr_sdv4_mild \
    --dataroot /path/to/GenImage --classes stable_diffusion_v_1_4 \
    --batch_size 32 --lr 0.0004 --niter 50 --cropSize 224 --aug_preset mild

python UnivFD/train_univfd_genimage_mild.py --data_root /path/to/GenImage/sdv4
python UnivFD/eval_univfd.py --model_path <ckpt> --test_dir /path/to/GenImage/test --perturbations all
```
The UnivFD scripts use the CLIP loader from
[UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect).

## Reproduce the cross-generator (GenImage main) results

The clean-protocol training/eval recipe and self-contained code are in
[`main_table_protocol/`](main_table_protocol/PROTOCOL.md).

## Inference on your own images

```bash
python predict_mord.py --model_path checkpoints/mord_sdv4_mild.pth --input <image_or_folder>
```

