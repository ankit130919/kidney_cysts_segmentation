"""Segment the five crop organs with ONE 1.5 mm model instead of five.

WHAT TotalSegmentator DOES. `--roi_subset` filters the OUTPUT, not the computation
(python_api.py:103). A `total` run with a roi_subset performs:

    1x Dataset298 at 6 mm over the whole volume, to build a rough crop
    5x Datasets 291,292,293,294,295 at 1.5 mm ON THAT CROP  ("total" is a 5-model ensemble)

and then discards everything outside the requested names.

Of those five, exactly one holds anything we need:

    291 organs      spleen, kidney_right, kidney_left, liver, colon   <- all five
    292 vertebrae / 293 cardiac / 294 muscles / 295 ribs              <- nothing

SO WHY KEEP THE 6 mm PASS? Because it is what makes the 1.5 mm stage cheap. Measured on a
512x512x433 study: Dataset291 alone, on the FULL volume with no pre-crop, took 70.1 s --
SLOWER than TotalSegmentator's whole six-model pipeline at 65.7 s. The crop shrinks the
field the expensive models run on by far more than four extra models cost. Dropping it is
the obvious optimisation and it is a pessimisation.

This therefore does what TotalSegmentator does, minus the four models whose output is
thrown away: 6 mm pre-crop, then Dataset291 at 1.5 mm on the crop. Same weights, same
spacing, same crop logic -- not a speed/accuracy trade.

FALLBACK. If the TotalSegmentator internals move, this falls back to the CLI and says why.
A changed private API must cost speed, never correctness.
"""
import contextlib
import os

import numpy as np

ORGANS = ("kidney_left", "kidney_right", "liver", "spleen", "colon")
TASK_ORGANS = 291           # 1.5 mm, part1 organs
TASK_ROUGH = 298            # 6 mm, whole body -- used only to crop
SPACING = 1.5
ROUGH_SPACING = 6.0
TRAINER = "nnUNetTrainerNoMirroring"
ROUGH_TRAINER = "nnUNetTrainer_4000epochs_NoMirroring"
CROP_ADDON = [20, 20, 20]   # TotalSegmentator's own default for a roi_subset crop


@contextlib.contextmanager
def _ts_env():
    """Point nnU-Net at TotalSegmentator's weights for the duration of one call.

    nnU-Net resolves a numeric task id through nnUNet_results, and TotalSegmentator sets
    that to its own weights tree while our package sets it to OUR model. Both are right
    and they cannot both hold, so the TS value is installed only around the call --
    otherwise the first organ segmentation silently breaks cyst inference afterwards.
    """
    keys = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        from totalsegmentator.config import setup_nnunet
        setup_nnunet()
        yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _organ_labels():
    from totalsegmentator.map_to_binary import class_map_5_parts
    m = class_map_5_parts["class_map_part_organs"]
    inv = {v: k for k, v in m.items()}
    missing = [o for o in ORGANS if o not in inv]
    if missing:
        raise RuntimeError(f"Dataset{TASK_ORGANS} no longer provides {missing}; "
                           "the part-model split changed upstream")
    return {o: inv[o] for o in ORGANS}


def _total_labels():
    from totalsegmentator.map_to_binary import class_map
    inv = {v: k for k, v in class_map["total"].items()}
    return {o: inv[o] for o in ORGANS if o in inv}


def segment_organs(ct_path, device="cuda", quiet=True):
    """{organ: bool mask} on the input grid. Raises on failure; caller may fall back."""
    import nibabel as nib
    from totalsegmentator.libs import download_pretrained_weights
    from totalsegmentator.nnunet import nnUNet_predict_image

    want = _organ_labels()
    rough_want = _total_labels()
    common = dict(model="3d_fullres", folds=[0], tta=False, multilabel_image=True,
                  crop_path=None, task_name="total", nora_tag="None", preview=False,
                  save_binary=False, nr_threads_resampling=1, nr_threads_saving=1,
                  output_type="nifti", statistics=False, quiet=quiet, verbose=False,
                  test=0, skip_saving=True, device=device, resampling_order=1)
    with _ts_env():
        download_pretrained_weights(TASK_ROUGH)
        download_pretrained_weights(TASK_ORGANS)

        rough = nnUNet_predict_image(ct_path, None, TASK_ROUGH, trainer=ROUGH_TRAINER,
                                     resample=ROUGH_SPACING, crop=None,
                                     crop_addon=[3, 3, 3], **common)
        if isinstance(rough, (tuple, list)):
            rough = rough[0]
        ra = np.asanyarray(rough.dataobj)
        cm = np.zeros(ra.shape, dtype=np.uint8)
        for v in rough_want.values():
            cm[ra == v] = 1
        if not cm.any():
            raise RuntimeError("6 mm pass found none of the crop organs")
        crop_img = nib.Nifti1Image(cm, rough.affine)

        seg = nnUNet_predict_image(ct_path, None, TASK_ORGANS, trainer=TRAINER,
                                   resample=SPACING, crop=crop_img,
                                   crop_addon=CROP_ADDON, **common)
    if isinstance(seg, (tuple, list)):
        seg = seg[0]
    a = np.asanyarray(seg.dataobj)
    out = {o: (a == v) for o, v in want.items()}
    if not any(m.any() for m in out.values()):
        raise RuntimeError(f"Dataset{TASK_ORGANS} returned none of the five organs")
    return out, seg


def segment_organs_cli(ct_path, seg_dir, gpu=None, totalsegmentator=None, home=None):
    """The TotalSegmentator CLI path, kept as the fallback."""
    import subprocess

    import nibabel as nib
    if not os.path.exists(os.path.join(seg_dir, ".done")):
        os.makedirs(seg_dir, exist_ok=True)
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if home:
            env["TOTALSEG_HOME_DIR"] = home
        exe = totalsegmentator or os.environ.get("CYSTS_TOTALSEG", "TotalSegmentator")
        p = subprocess.run([exe, "-i", ct_path, "-o", seg_dir, "-ta", "total",
                            "--roi_subset", *ORGANS],
                           capture_output=True, text=True, env=env)
        if p.returncode != 0:
            raise RuntimeError(f"TotalSegmentator failed: {p.stderr.strip()[-400:]}")
        open(os.path.join(seg_dir, ".done"), "w").close()
    out = {}
    for o in ORGANS:
        f = os.path.join(seg_dir, f"{o}.nii.gz")
        out[o] = (np.asanyarray(nib.load(f).dataobj) > 0) if os.path.exists(f) else None
    return out, None
