"""Run the cyst model on a prepared crop and measure what it found.

The model is nnU-Net 3d_fullres, fine-tuned from TotalSegmentator's
Dataset789_kidney_cyst_501subj. Output labels: 1 kidney, 2 kidney_cyst.

MEASUREMENT CHOICES, and why

  MIN_MM3 = 20. Components below this are dropped everywhere in this project. It is a
  floor on what can be measured, not on what is clinically real -- at the 1.5 mm target
  spacing a 20 mm3 object is about six voxels.

  Volume is the per-component voxel count times voxel volume. No surface correction: the
  mask is what the model produced and inflating it would be inventing precision.

  Mean HU is read from the cropped CT the model saw, so it is directly comparable with
  the -10..30 HU expected of cyst fluid. It is reported for REVIEW, never used to filter
  -- see postprocess.py for the measurements behind that.

  Containment is the fraction of the component inside the TotalSegmentator kidney mask.
  Low containment is NOT a quality flag: it tracks lesion size, because a 30 ml cyst
  physically cannot fit inside a kidney and is necessarily exophytic. Measured on 347
  predictions, median containment fell from 1.00 below 0.25 ml to 0.21 above 20 ml.
"""
import os

import nibabel as nib
import numpy as np
from scipy import ndimage

from ..common.geometry import max_diameter_mm, sphericity
from .measure import format_size, inplane_extents_mm, side_from_kidney

CYST, KIDNEY = 2, 1
MIN_MM3 = 20.0
S = np.ones((3, 3, 3))


def components(pred_path, crop_path, kidney_path=None, min_mm3=MIN_MM3):
    """One record per predicted cyst, largest first."""
    # canonical RAS: axis 0 = L-R, axis 1 = P-A, axis 2 = I-S, so in-plane height and
    # width mean the same thing on every study regardless of the source orientation
    pi = nib.as_closest_canonical(nib.load(pred_path))
    pr = np.asanyarray(pi.dataobj)
    zooms = np.asarray(pi.header.get_zooms()[:3], dtype=float)
    voxvol = float(np.prod(zooms))
    ct = np.asanyarray(nib.as_closest_canonical(nib.load(crop_path)).dataobj).astype(np.float32)
    kid = None
    if kidney_path and os.path.exists(kidney_path):
        kid = np.asanyarray(nib.as_closest_canonical(nib.load(kidney_path)).dataobj)

    cy = pr == CYST
    out = []
    if not cy.any():
        return out, dict(pred_kidney_ml=round(float((pr == KIDNEY).sum()) * voxvol / 1000, 1),
                         slice_thickness_mm=round(float(zooms[2]), 2))
    lab, n = ndimage.label(cy, structure=S)
    sizes = np.bincount(lab.ravel())
    dt = (ndimage.distance_transform_edt(kid == 0, sampling=zooms)
          if (kid is not None and (kid > 0).any()) else None)
    for i in range(1, n + 1):
        vol_mm3 = sizes[i] * voxvol
        if vol_mm3 < min_mm3:
            continue
        c = lab == i
        hu = ct[c]
        h_mm, w_mm = inplane_extents_mm(c, zooms)
        rec = dict(
            size_mm=format_size(h_mm, w_mm),
            height_mm=h_mm,
            width_mm=w_mm,
            density_hu=int(round(float(hu.mean()))),
            volume_ml=round(vol_mm3 / 1000.0, 4),
            max_diameter_3d_mm=round(max_diameter_mm(c, zooms), 1),
            mean_hu=round(float(hu.mean()), 1),
            min_hu=round(float(hu.min()), 1),
            max_hu=round(float(hu.max()), 1),
            sphericity=round(sphericity(c, zooms), 3),
            n_voxels=int(sizes[i]),
        )
        zi = np.argwhere(c)[:, 2]
        # the slice to show and to quote is the one the lesion is LARGEST on, not its
        # first or middle slice -- a radiologist measures where it is biggest
        counts = np.bincount(zi, minlength=pr.shape[2])
        rec["slice_z_canonical"] = int(counts.argmax())
        rec["slice_range_canonical"] = [int(zi.min()), int(zi.max())]
        rec["_mask_index"] = int(i)
        if kid is not None:
            inside = float((c & (kid > 0)).sum()) / float(sizes[i])
            rec["kidney_containment"] = round(inside, 3)
            rec["distance_to_kidney_mm"] = (
                0.0 if inside > 0 else (round(float(dt[c].min()), 1) if dt is not None else None))
            rec["side"] = side_from_kidney(c, kid)
        out.append(rec)
    out.sort(key=lambda r: -r["volume_ml"])
    ctx = dict(pred_kidney_ml=round(float((pr == KIDNEY).sum()) * voxvol / 1000, 1),
               slice_thickness_mm=round(float(zooms[2]), 2))
    return out, ctx


def summarise(comps):
    if not comps:
        return dict(cyst_present=False, n_cysts=0, total_volume_ml=0.0,
                    largest_volume_ml=0.0, max_diameter_3d_mm=0.0,
                    largest_size_mm=None, max_density_hu=None, mean_hu=None)
    return dict(cyst_present=True, n_cysts=len(comps),
                total_volume_ml=round(sum(c["volume_ml"] for c in comps), 4),
                largest_volume_ml=comps[0]["volume_ml"],
                max_diameter_3d_mm=max(c["max_diameter_3d_mm"] for c in comps),
                largest_size_mm=comps[0]["size_mm"],
                max_density_hu=max(c["density_hu"] for c in comps),
                mean_hu=round(float(np.mean([c["mean_hu"] for c in comps])), 1))
