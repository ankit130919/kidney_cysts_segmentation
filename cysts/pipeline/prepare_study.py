"""DICOM study -> the cropped volume the cyst model expects.

    choose series -> TotalSegmentator -> crop to organ box + 10 mm

Every stage is skipped if its output exists, so re-running is cheap and a failed study
resumes rather than restarts.

VALIDATE BY DECODING, NOT BY EXISTENCE. A NIfTI that exists is not a NIfTI that decodes.
Truncated .nii.gz files broke three separate runs during development -- each time the
guard was os.path.exists on a gzip, each time the failure surfaced hours later inside
nnU-Net as `EOFError: Compressed file ended before the end-of-stream marker`. Anything
cached here is decoded before it is trusted.
"""
import os

import nibabel as nib
import numpy as np

from ..common import dicom_index, extract_series, paths, run_anatomy


def _map_path(study_id):
    return os.path.join(os.path.dirname(paths.crop_path(study_id)),
                        f"{study_id}_slicemap.json")


def readable(p):
    try:
        np.asanyarray(nib.load(p).dataobj).sum()
        return True
    except Exception:
        return False


def prepare(study_id, study_dir, gpu=None, force=False):
    """Returns a dict describing the prepared study, or raises with a usable message."""
    paths.ensure(paths.NIFTI, paths.SEG, paths.CROPS, paths.KIDNEY)
    crop = paths.crop_path(study_id)
    kidney = paths.kidney_path(study_id)
    for d in (os.path.dirname(crop), os.path.dirname(kidney)):
        os.makedirs(d, exist_ok=True)
    if not force and os.path.exists(crop) and os.path.exists(kidney):
        if readable(crop) and readable(kidney):
            img = nib.load(crop)
            kid = np.asanyarray(nib.load(kidney).dataobj)
            z = img.header.get_zooms()[:3]
            return dict(study_id=study_id, crop=crop, kidney=kidney, cached=True,
                        shape=tuple(int(v) for v in img.shape),
                        slice_map=dicom_index.load(_map_path(study_id)),
                        kidney_ml=round(float((kid > 0).sum()) * float(np.prod(z)) / 1000, 1))
        for p in (crop, kidney):                      # truncated -- rebuild
            if os.path.exists(p):
                os.remove(p)

    work = os.path.join(paths.NIFTI, str(study_id))
    series, tier, n_cands = extract_series.choose(study_dir, work)
    if series is None:
        raise RuntimeError(f"{tier} ({n_cands} candidates)")
    seg = os.path.join(paths.SEG, str(study_id))
    # crop_study segments internally (one model, masks in memory) and only uses seg
    # if it has to fall back to the TotalSegmentator CLI
    info = run_anatomy.crop_study(series, seg, crop, kidney, gpu=gpu)

    # crop slice -> DICOM InstanceNumber, by physical position. Built once and cached;
    # a failure here costs the slice number, not the analysis.
    smap = []
    try:
        import json as _json
        side = series[:-7] + ".json"
        sn = _json.load(open(side)).get("SeriesNumber") if os.path.exists(side) else None
        sdir = dicom_index.find_series_dir(study_dir, sn) if sn is not None else None
        if sdir:
            smap = dicom_index.build_slice_map(nib.load(crop), sdir)
            dicom_index.save(_map_path(study_id), smap)
    except Exception as e:
        print(f"  slice map unavailable for {study_id}: {type(e).__name__}: {e}", flush=True)

    return dict(study_id=study_id, crop=crop, kidney=kidney, cached=False,
                series=os.path.basename(series), series_tier=tier,
                n_candidate_series=n_cands, slice_map=smap, **info)
