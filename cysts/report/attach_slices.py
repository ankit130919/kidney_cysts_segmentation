"""Fill slice_number and slice_image into each reported cyst.

Separate from detect_cysts because it needs things detection does not: the DICOM headers
(for the InstanceNumber) and PIL (for the PNG). Detection stays usable without either.

FAILS SOFT, LOUDLY. If the DICOM series cannot be located the slice number is null and a
`slice_number_unavailable` note is added -- it is never silently replaced with the array
index, which would look like a valid PACS reference and point at the wrong image.
"""
import os

import nibabel as nib
import numpy as np
from scipy import ndimage

from ..common import dicom_index
from . import slice_image

CYST = 2
S = np.ones((3, 3, 3))
CONTEXT_MM = 60.0          # half-width of the view around the lesion


def _box(mask2d, shape, zooms, context_mm=CONTEXT_MM):
    """Crop window around the lesion, in DISPLAY coordinates, or None for the full slice."""
    ys, xs = np.nonzero(mask2d)
    if not len(xs):
        return None
    pad_y = int(round(context_mm / max(float(zooms[1]), 1e-3)))
    pad_x = int(round(context_mm / max(float(zooms[0]), 1e-3)))
    y0 = max(int(ys.min()) - pad_y, 0); y1 = min(int(ys.max()) + pad_y, shape[0])
    x0 = max(int(xs.min()) - pad_x, 0); x1 = min(int(xs.max()) + pad_x, shape[1])
    return (y0, y1, x0, x1)


def attach(comps, crop_path, pred_path, slice_map=None, context_mm=CONTEXT_MM,
           max_dim=slice_image.MAX_DIM):
    """Mutates and returns comps, adding slice_number and slice_image to each."""
    if not comps:
        return comps
    crop_img = nib.load(crop_path)
    ct = np.asanyarray(nib.as_closest_canonical(crop_img).dataobj).astype(np.float32)
    pr = np.asanyarray(nib.as_closest_canonical(nib.load(pred_path)).dataobj)
    zooms = nib.as_closest_canonical(crop_img).header.get_zooms()[:3]
    lab, _ = ndimage.label(pr == CYST, structure=S)

    for c in comps:
        z = c.get("slice_z_canonical")
        idx = c.pop("_mask_index", None)
        if z is None or idx is None:
            continue
        m3 = lab == idx
        m2 = m3[:, :, z]
        disp = slice_image._display(m2)
        box = _box(disp, disp.shape, zooms, context_mm)
        c["slice_image"] = slice_image.render(ct, m3, z, max_dim=max_dim, box=box)
        if slice_map:
            zc = dicom_index.to_canonical_index(crop_img, z)
            c["slice_number"] = (slice_map[zc] if 0 <= zc < len(slice_map) else None)
        else:
            c["slice_number"] = None
            c["slice_number_unavailable"] = (
                "DICOM series not located -- not substituting an array index")
    return comps
