"""Voxel geometry: cropping, bounding boxes, and the measurements a report quotes.

Diameter is the true 3D maximum Feret diameter -- the largest distance between any two
voxels of the lesion -- not the bounding-box diagonal and not an in-plane measurement.
A radiologist's "21 x 20 mm" is an in-plane pair, so the two are comparable only for
roughly spherical lesions; for a lobulated cyst the Feret diameter is the larger number
and the honest one.
"""
import nibabel as nib
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist


def crop_to(img, lo, hi, arr=None):
    """Crop a NIfTI to an index box, keeping the affine physically correct."""
    a = np.asanyarray(img.dataobj) if arr is None else arr
    sl = tuple(slice(int(l), int(h) + 1) for l, h in zip(lo, hi))
    aff = img.affine.copy()
    aff[:3, 3] = nib.affines.apply_affine(img.affine, [lo[0], lo[1], lo[2]])
    out = nib.Nifti1Image(a[sl], aff, img.header)
    return out


def pad_box(lo, hi, shape, zooms, addon_mm):
    pad = np.ceil(addon_mm / np.asarray(zooms[:3], dtype=float)).astype(int)
    lo = np.maximum(np.asarray(lo) - pad, 0)
    hi = np.minimum(np.asarray(hi) + pad, np.asarray(shape) - 1)
    return lo, hi


def max_diameter_mm(mask, zooms):
    """3D maximum Feret diameter in mm.

    The hull reduction is not an approximation of the answer -- the farthest pair of
    points always lies on the convex hull -- it is an approximation only when the hull
    itself fails on a degenerate (flat, single-slice) lesion, which is why there is a
    fallback.
    """
    idx = np.argwhere(mask)
    if len(idx) < 2:
        return 0.0
    pts = idx * np.asarray(zooms[:3], dtype=float)
    if len(pts) > 800:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except Exception:
            pts = pts[np.random.RandomState(0).choice(len(pts), 800, replace=False)]
    return float(pdist(pts).max())


def sphericity(mask, zooms):
    """1.0 for a sphere, lower for an elongated or branching object.

    Used as a REVIEW signal, not a filter. Measured on 207 predictions, hydronephrosis
    scored 0.844 against 0.909 for cysts -- a real difference, but one that overlaps far
    too much to threshold on, and it is confounded with lesion size (small lesions are
    voxelated and score lower regardless of what they are).
    """
    idx = np.argwhere(mask)
    if len(idx) < 2:
        return 0.0
    z = np.asarray(zooms[:3], dtype=float)
    vol = float(mask.sum()) * float(np.prod(z))
    ext = (idx.max(0) - idx.min(0) + 1) * z
    r_eq = (3 * vol / (4 * np.pi)) ** (1 / 3)
    return float(2 * r_eq / max(ext.max(), 1e-6))
