"""DICOM study -> NIfTI candidates -> the one series to analyse.

The size filters drop scouts, dose reports and tiny reconstructions before the
preference rule ever sees them; without them a 3-slice localiser can win a tier.
"""
import glob
import json
import os
import subprocess

import nibabel as nib

from . import paths
from .triage_series import pick, TIER_NAME

MIN_INPLANE = 192
MIN_SLICES = 40


def convert(study_dir, out_dir, dcm2niix=None):
    """dcm2niix the study once; subsequent calls reuse the output."""
    os.makedirs(out_dir, exist_ok=True)
    if not glob.glob(f"{out_dir}/*.nii.gz"):
        exe = dcm2niix or os.environ.get("CYSTS_DCM2NIIX", "dcm2niix")
        subprocess.run([exe, "-z", "y", "-f", "%s_%p_%d", "-o", out_dir, study_dir],
                       capture_output=True, text=True)
    return sorted(glob.glob(f"{out_dir}/*.nii.gz"))


def candidates(nifti_files):
    out = []
    for f in nifti_files:
        try:
            img = nib.load(f)
        except Exception:
            continue
        if img.ndim != 3:
            continue
        nx, ny, nz = img.shape
        if nx < MIN_INPLANE or ny < MIN_INPLANE or nz < MIN_SLICES:
            continue
        meta = {}
        side = f[:-7] + ".json"
        if os.path.exists(side):
            try:
                meta = json.load(open(side))
            except Exception:
                pass
        out.append(dict(path=f, name=os.path.basename(f),
                        desc=str(meta.get("SeriesDescription", "")),
                        prot=str(meta.get("ProtocolName", "")),
                        nz=nz, cov=nz * float(img.header.get_zooms()[2])))
    return out


def choose(study_dir, work_dir):
    """(path, tier_name, n_candidates) for one study, or (None, reason, 0)."""
    files = convert(study_dir, work_dir)
    cands = candidates(files)
    if not cands:
        return None, "no usable series (all scouts/dose reports/too small)", 0
    best, t = pick(cands)
    if best is None:
        return None, "no usable series", len(cands)
    return best["path"], TIER_NAME[t], len(cands)
