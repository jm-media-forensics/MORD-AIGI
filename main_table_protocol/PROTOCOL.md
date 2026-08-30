# GenImage Cross-Generator (Main Table) — Training Protocol

Self-contained code to train the clean cross-generator model. We release the training
recipe (not the checkpoint); train your own with the command below.

**Settings:** Adam, lr `0.001`, LR ×0.9 every 10 epochs, batch `64`, up to `200` epochs,
`loadSize 256` / `cropSize 224`, train on the GenImage **SDv1.4** split, **no** degradation
augmentation. Preprocessing: tile-pad small images → `RandomCrop(224)` + flip (train) /
`CenterCrop(256)` (test), no resize, no normalization.

**Train:**
```bash
python train.py \
    --dataroot /path/to/GenImage/sdv4_root \
    --classes stable_diffusion_v_1_4 \
    --name mord_maintable_clean \
    --batch_size 64 --lr 0.001 --niter 200 --delr_freq 10 \
    --loadSize 256 --cropSize 224
```

**Evaluate:** run `validate.py` on each test generator folder (`0_real/`, `1_fake/`) with
the test preprocessing above; report per-generator Acc/AP and their mean.
