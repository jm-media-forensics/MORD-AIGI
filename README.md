# MORD

Training and test code for MORD. MORD has two branches, BCPR and PiD. Each branch is trained separately; at test time their probabilities are averaged.

```bash
pip install -r requirements.txt
```

Every data folder contains `0_real/` and `1_fake/`. Inputs are RGB in [0, 1] without normalization.

## Train

Each command trains one branch. Run it once with `--branch bcpr` and once with `--branch pid`, for seeds 1, 2 and 3.

| Setting | Command |
|---|---|
| UFD / Self-Synthesis | `python train_clean.py --branch bcpr --seed 1 --name progan_bcpr_s1 --dataroot <progan> --classes car,cat,chair,horse --lr 0.0004 --niter 50` |
| GenImage | `python train_clean.py --branch bcpr --seed 1 --name genimage_bcpr_s1 --dataroot <sdv14> --classes stable_diffusion_v_1_4 --lr 0.001 --niter 200` |
| Robustness | `python train_robust.py --branch bcpr --seed 1 --name mild_bcpr_s1 --dataroot <sdv14> --classes stable_diffusion_v_1_4 --aug_preset mild --val_dir <sdv14_val>` |
| Format-matched | `python make_format_matched.py --train <sdv14>/train/stable_diffusion_v_1_4 --val <sdv14_val> --test <genimage_test> --out <fm>`<br>`python train_robust.py --branch bcpr --seed 1 --name fm_bcpr_s1 --dataroot <fm> --classes stable_diffusion_v_1_4 --aug_preset aggressive --val_dir <fm>/val/stable_diffusion_v_1_4` |
| WildRF | `python train_wildrf.py cache --src <wildrf> --cache <wildrf256>`<br>`python train_wildrf.py train --cache <wildrf256> --branch bcpr --seed 1 --out checkpoints/wildrf_bcpr_s1`<br>`python train_wildrf.py select --cache <wildrf256> --bcpr checkpoints/wildrf_bcpr_s1 --pid checkpoints/wildrf_pid_s1` |

`<progan>` and `<sdv14>` contain `train/<class>/{0_real,1_fake}`. The SDv1.4 training subset has 5,400 real and 5,400 fake images.

## Test

```bash
# UFD, Self-Synthesis, GenImage (one result per sub-folder)
python test.py --protocol clean --root <test_root> --bcpr <bcpr.pth> --pid <pid.pth>

# Robustness and format-matched control (12 conditions, 800 + 800 images per generator)
python test.py --protocol robust --root <genimage_test or fm/test> --cap 800 --bcpr <bcpr.pth> --pid <pid.pth>

# Composed conditions
python test.py --protocol robust --root <fm>/test --cap 800 --conditions resize_0.5+jpeg_75,blur_1.0+jpeg_75 --bcpr <bcpr.pth> --pid <pid.pth>

# WildRF (checkpoints of the selected epoch)
python test.py --protocol wildrf --root <wildrf>/test --bcpr <bcpr_epoch.pth> --pid <pid_epoch.pth>
```

`test.py` reports Acc and AP of BCPR, PiD and MORD.
