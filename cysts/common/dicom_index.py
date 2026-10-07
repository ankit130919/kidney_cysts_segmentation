"""Map a voxel slice of the cropped volume back to its DICOM InstanceNumber.

WHY THIS IS NOT JUST AN ARRAY INDEX. The z printed on an internal review sheet is an index
into a volume that has been (a) converted by dcm2niix, (b) cropped to the organ box, and
(c) reoriented to canonical RAS. All three shift or flip it. A caller who takes that number
to PACS looks at the wrong image -- on the 752-study cohort the crop offset alone was 139
slices on the study measured, and differs per study.

HOW. Position, not ordering. For each slice of the crop the patient-space coordinate is
computed from the affine and matched to the DICOM slice with the nearest projection along
the slice normal. Ordering is never assumed: dcm2niix sorts by position, the archive may
store InstanceNumber running the other way, and the two disagreeing is exactly the bug
that put annotation masks on the wrong slices earlier in this project.

COST. One header read per file of the chosen series (~1 s for a 1000-slice study), done
once at prepare time and cached next to the crop.
"""
import glob
import json
import os

import numpy as np
import pydicom


def _headers(series_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(series_dir, "*"))):
        if os.path.isdir(f):
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            ipp = [float(v) for v in ds.ImagePositionPatient]
            iop = [float(v) for v in ds.ImageOrientationPatient]
        except Exception:
            continue
        n = np.cross(iop[:3], iop[3:])
        out.append(dict(inst=int(getattr(ds, "InstanceNumber", 0) or 0),
                        proj=float(np.dot(ipp, n)),
                        series=str(getattr(ds, "SeriesNumber", "") or ""),
                        normal=n.tolist()))
    return out


def find_series_dir(study_dir, series_number):
    """The sub-directory holding the series dcm2niix numbered `series_number`."""
    want = str(series_number).strip()
    for d in sorted(glob.glob(os.path.join(study_dir, "*"))):
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*")))[:1]:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            except Exception:
                continue
            if str(getattr(ds, "SeriesNumber", "") or "").strip() == want:
                return d
    return None


def build_slice_map(crop_img, series_dir):
    """[InstanceNumber or None] indexed by the crop's own z axis (pre-canonical)."""
    hdrs = [h for h in _headers(series_dir) if h["inst"]]
    if not hdrs:
        return []
    normal = np.asarray(hdrs[0]["normal"], dtype=float)
    projs = np.asarray([h["proj"] for h in hdrs])
    insts = [h["inst"] for h in hdrs]
    A = crop_img.affine
    out = []
    for z in range(crop_img.shape[2]):
        p = A @ np.array([0.0, 0.0, float(z), 1.0])
        d = np.abs(projs - float(np.dot(p[:3], normal)))
        out.append(int(insts[int(d.argmin())]))
    return out


def save(path, slice_map):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(slice_map, open(path, "w"))


def load(path):
    if not os.path.exists(path):
        return []
    try:
        return json.load(open(path))
    except Exception:
        return []


def to_canonical_index(crop_img, z_canon):
    """Canonical-RAS z -> the crop's own z, so a slice map built on the crop still applies.

    Only the sign of the third axis can differ between the two for an axial volume, so this
    is a flip or a no-op -- but which one varies by study (4 of 752 crops flip), and
    guessing is how a number ends up one slice-count away from correct.
    """
    import nibabel as nib
    ax = nib.aff2axcodes(crop_img.affine)
    return (crop_img.shape[2] - 1 - z_canon) if ax[2] == "I" else z_canon
