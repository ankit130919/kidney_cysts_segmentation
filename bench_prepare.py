#!/usr/bin/env python3
"""Old vs new organ segmentation: speed, and whether the crop is the same.

Must be a real file with a __main__ guard -- nnU-Net's preprocessing workers re-import
__main__, and from a heredoc that is '<stdin>', which they cannot import. The failure
looks like "Background workers died", which reads as an out-of-memory problem and is not.
"""
import os
import sys
import time

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cysts.common import organ_seg, run_anatomy  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "/tmp/prep_probe/102_26.1_Routine_Abdomen_Pelvis_Pre_Contrast_10mm_Abdomen.nii.gz"
CLI_SEG = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ts_probe_run1"


def main():
    img = nib.load(SRC)
    z = img.header.get_zooms()[:3]
    print(f"input {img.shape} spacing {[round(float(v), 2) for v in z]}", flush=True)

    t = time.time()
    masks_new, _ = organ_seg.segment_organs(SRC)
    t_new = time.time() - t
    print(f"\nNEW  Dataset291 only        {t_new:7.1f} s      (OLD CLI 'total': 65.7 s)",
          flush=True)

    masks_old, _ = organ_seg.segment_organs_cli(SRC, CLI_SEG)
    bo, ko, fo = run_anatomy.organ_box_from_masks(masks_old, img.shape, z)
    bn, kn, fn = run_anatomy.organ_box_from_masks(masks_new, img.shape, z)
    print(f"\norgans OLD {sorted(fo)}")
    print(f"organs NEW {sorted(fn)}")
    print(f"\ncrop box OLD {list(bo[0])} .. {list(bo[1])}")
    print(f"crop box NEW {list(bn[0])} .. {list(bn[1])}")
    d = max(int(abs(a - b)) for a, b in
            zip(list(bo[0]) + list(bo[1]), list(bn[0]) + list(bn[1])))
    print(f"max crop-corner difference  {d} voxels "
          f"({d * float(min(z)):.1f} mm) -- the box is padded 10 mm, so this is the "
          f"number that matters, not per-voxel agreement")
    print("\nper-organ agreement")
    for o in run_anatomy.ORGANS:
        a, b = masks_old.get(o), masks_new.get(o)
        if a is None or b is None:
            print(f"  {o:<14} missing on one side")
            continue
        dice = 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
        print(f"  {o:<14} Dice {dice:.4f}   {a.sum() * np.prod(z) / 1000:7.1f} -> "
              f"{b.sum() * np.prod(z) / 1000:7.1f} ml")

    t = time.time()
    info = run_anatomy.crop_study(SRC, CLI_SEG, "/tmp/c2.nii.gz", "/tmp/k2.nii.gz")
    print(f"\nfull crop_study now         {time.time() - t:7.1f} s      "
          f"(was 65.7 + 18.4 + 24.6 = 108.7 s)")
    print(f"  path={info['segmentation_path']} shape={info['shape']} "
          f"kidney={info['kidney_ml']} ml")


if __name__ == "__main__":
    main()
