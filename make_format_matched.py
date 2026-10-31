# -*- coding: utf-8 -*-
"""Format-matched copy of the SDv1.4 data (Table 2, format-matched control).

Real images are left unchanged (symlinked). Every fake image is paired with a randomly
drawn real image of the same split and saved as JPEG with that real image's quantization
tables and chroma subsampling. Images are written by PIL without metadata.

Output layout:
  train/stable_diffusion_v_1_4/{0_real,1_fake}
  val/stable_diffusion_v_1_4/{0_real,1_fake}
  test/<generator>/{0_real,1_fake}          (first --test_cap images per class)

  python make_format_matched.py --train /data/genimage_sdv14/train/stable_diffusion_v_1_4 \
      --val /data/genimage/val/stable_diffusion_v_1_4 --test /data/genimage/test --out /data/format_matched
"""
import argparse
import json
import os
import random
from multiprocessing import Pool

from PIL import Image, JpegImagePlugin

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUBSAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


def list_imgs(d, cap=None):
    fs = [f for f in sorted(os.listdir(d)) if os.path.splitext(f)[1].lower() in IMG_EXT]
    return fs[:cap] if cap else fs


def jobs_for(src, dst, cap, rng):
    reals = list_imgs(os.path.join(src, "0_real"), cap)
    fakes = list_imgs(os.path.join(src, "1_fake"), cap)
    jobs = [("real", os.path.join(src, "0_real", f), os.path.join(dst, "0_real", f), None) for f in reals]
    for f in fakes:
        ref = os.path.join(src, "0_real", rng.choice(reals))
        jobs.append(("fake", os.path.join(src, "1_fake", f),
                     os.path.join(dst, "1_fake", os.path.splitext(f)[0] + ".jpg"), ref))
    return jobs, {"real": len(reals), "fake": len(fakes)}


def run_job(job):
    cls, src, dst, ref = job
    try:
        if cls == "real":
            if not os.path.lexists(dst):
                os.symlink(src, dst)
            return "ok"
        r = Image.open(ref)
        qtables, sub = r.quantization, JpegImagePlugin.get_sampling(r)
        Image.open(src).convert("RGB").save(dst, "JPEG", qtables=qtables, subsampling=sub if sub in SUBSAMPLING else 2)
        return "ok"
    except Exception as e:
        return f"{src}: {e}"


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--train", required=True, help="training folder with 0_real/ and 1_fake/")
    pa.add_argument("--val", required=True, help="validation folder with 0_real/ and 1_fake/")
    pa.add_argument("--test", required=True, help="folder with one sub-folder per test generator")
    pa.add_argument("--out", required=True)
    pa.add_argument("--test_cap", type=int, default=800)
    pa.add_argument("--workers", type=int, default=10)
    pa.add_argument("--seed", type=int, default=0)
    a = pa.parse_args()
    rng = random.Random(a.seed)

    plan = [("train/stable_diffusion_v_1_4", a.train, None), ("val/stable_diffusion_v_1_4", a.val, None)]
    plan += [(f"test/{g}", os.path.join(a.test, g), a.test_cap)
             for g in sorted(os.listdir(a.test)) if os.path.isdir(os.path.join(a.test, g))]

    all_jobs, manifest = [], {"seed": a.seed, "splits": {}}
    for rel, src, cap in plan:
        dst = os.path.join(a.out, rel)
        for c in ("0_real", "1_fake"):
            os.makedirs(os.path.join(dst, c), exist_ok=True)
        jobs, summary = jobs_for(src, dst, cap, rng)
        all_jobs += jobs
        manifest["splits"][rel] = {"source": src, **summary}
        print(f"{rel:40s} real {summary['real']:6d}  fake {summary['fake']:6d}", flush=True)

    with Pool(a.workers) as pool:
        res = pool.map(run_job, all_jobs, chunksize=64)
    errors = [r for r in res if r != "ok"]
    manifest.update(n_images=len(all_jobs), n_errors=len(errors), errors=errors[:50],
                    pairing={j[2]: j[3] for j in all_jobs if j[0] == "fake"})
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"done: {len(all_jobs)} images, {len(errors)} errors -> {a.out}")


if __name__ == "__main__":
    main()
