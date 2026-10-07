"""Per-cyst measurements in the form a report quotes them.

SIZE. The response reports `size_mm` as "height x width": the largest ANTERIOR-POSTERIOR
and TRANSVERSE extents of the lesion, measured in-plane and taken over the slices it
occupies. That is what a radiologist writes ("a small approximately 8 x 8 mm sized
cortical cyst"), so it is what can be compared against a report.

It is NOT the 3D maximum diameter, which is larger for any lesion that is not a sphere;
that is reported separately as `max_diameter_3d_mm` for anyone who wants it. Quoting a
Feret diameter where a report says "8 x 8" would read as the model over-measuring.

ORIENTATION. Extents are computed after conversion to canonical RAS, so axis 0 is
left-right (transverse) and axis 1 is posterior-anterior, whatever the source series
orientation was. Without this the two numbers swap on some studies.

SIDE. Taken from the TotalSegmentator kidney mask the lesion overlaps (LEFT=1, RIGHT=2),
not from its position in the volume. The crop box spans liver, spleen and colon as well
as both kidneys, so it is not centred on the midline and a coordinate test mislabels
lesions near the edges.
"""
import numpy as np


def inplane_extents_mm(mask_ras, zooms_ras):
    """(height_mm, width_mm) -- max A-P and max L-R extent over the occupied slices."""
    idx = np.argwhere(mask_ras)
    if not len(idx):
        return 0.0, 0.0
    zx, zy = float(zooms_ras[0]), float(zooms_ras[1])
    h = w = 0.0
    for z in np.unique(idx[:, 2]):
        s = idx[idx[:, 2] == z]
        w = max(w, (s[:, 0].max() - s[:, 0].min() + 1) * zx)   # axis 0 = L-R
        h = max(h, (s[:, 1].max() - s[:, 1].min() + 1) * zy)   # axis 1 = P-A
    return round(h, 1), round(w, 1)


def side_from_kidney(comp_mask, kidney_mask):
    """'left' / 'right' / None, by which TS kidney the lesion sits in."""
    if kidney_mask is None:
        return None
    left = int((comp_mask & (kidney_mask == 1)).sum())
    right = int((comp_mask & (kidney_mask == 2)).sum())
    if left == 0 and right == 0:
        return None
    return "left" if left >= right else "right"


def format_size(height_mm, width_mm):
    return f"{height_mm:.1f} x {width_mm:.1f}"
