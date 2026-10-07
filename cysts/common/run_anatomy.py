"""Crop to the kidney region -- the field of view the cyst model was trained on.

The crop reproduces TotalSegmentator's own kidney_cysts recipe (map_tasks_config.py): the
bounding box of kidney_left, kidney_right, liver, spleen and colon, padded 10 mm. It is a
BOX, not a mask -- voxels outside the organs but inside the box keep their values. That is
deliberate in TS and matters here: renal cysts are frequently exophytic and bulge beyond
the kidney contour, so masking to the organ would amputate the part of the lesion that
makes it visible, and zeroing outside would manufacture an edge the network never saw.

Inference MUST use this same crop. The model was fine-tuned on it, and a different field
of view is a different input distribution.

WHAT CHANGED, AND WHAT IT COST. The segmentation itself moved to organ_seg (one 1.5 mm
model instead of five; see that module). Two further costs were measured and removed here:

  organ_box  read five gzipped full-volume masks back off disk to compute a bounding box
             -- 18.4 s of decompression for six integers. Masks now stay in memory.
  crop write gzip level 9 on a ~130 MB volume -- 24.6 s. These are intermediates the next
             stage reads seconds later and deletes; level 1 is plenty.
"""
import os

import nibabel as nib
import numpy as np

from . import organ_seg
from .geometry import crop_to, pad_box

ORGANS = list(organ_seg.ORGANS)
ADDON_MM = 10.0


def organ_box_from_masks(masks, shape, zooms):
    """(lo, hi) box over the five organs + 10 mm, a LEFT=1/RIGHT=2 kidney mask, organs found.

    The kidney sides are kept distinct because the response must say which kidney a cyst
    is in, and deciding that from a volume midpoint is wrong the moment the crop is not
    centred on the spine -- which it is not, since the box spans liver and spleen too.
    Anything reading this mask as "kidney" must test > 0, never == 1.
    """
    m = np.zeros(shape, dtype=bool)
    found = []
    for o in ORGANS:
        a = masks.get(o)
        if a is None or a.shape != shape:
            continue
        if a.any():
            found.append(o)
        m |= a
    if not m.any():
        return None, None, found
    idx = np.argwhere(m)
    lo, hi = pad_box(idx.min(0), idx.max(0), shape, zooms, ADDON_MM)
    kid = np.zeros(shape, dtype=np.uint8)
    for o, v in (("kidney_left", 1), ("kidney_right", 2)):
        a = masks.get(o)
        if a is not None and a.shape == shape:
            kid[a] = v
    return (lo, hi), kid, found


def organ_box(seg_dir, shape, zooms):
    """Backwards-compatible: read the five masks off disk, then box them."""
    masks = {}
    for o in ORGANS:
        f = os.path.join(seg_dir, f"{o}.nii.gz")
        masks[o] = (np.asanyarray(nib.load(f).dataobj) > 0) if os.path.exists(f) else None
    return organ_box_from_masks(masks, shape, zooms)


def segment(ct_path, seg_dir, gpu=None, totalsegmentator=None, home=None):
    """Deprecated. crop_study segments internally and keeps the masks in memory."""
    return organ_seg.segment_organs_cli(ct_path, seg_dir, gpu=gpu,
                                        totalsegmentator=totalsegmentator, home=home)


def crop_study(ct_path, seg_dir, crop_path, kidney_path, gpu=None, device="cuda"):
    """Segment, crop, write the cropped CT and kidney mask. seg_dir is for the fallback."""
    img = nib.load(ct_path)
    zooms = img.header.get_zooms()[:3]
    path_used = "precrop+dataset291"
    try:
        masks, _ = organ_seg.segment_organs(ct_path, device=device)
    except Exception as e:
        print(f"  organ_seg fast path unavailable ({type(e).__name__}: {str(e)[:140]}); "
              "falling back to the TotalSegmentator CLI", flush=True)
        masks, _ = organ_seg.segment_organs_cli(
            ct_path, seg_dir, gpu=gpu, home=os.environ.get("TOTALSEG_HOME_DIR"))
        path_used = "cli_total"

    box, kid, found = organ_box_from_masks(masks, img.shape, zooms)
    if box is None:
        raise RuntimeError(f"found none of {ORGANS} -- not an abdominal series?")
    lo, hi = box
    for p in (crop_path, kidney_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    ci = crop_to(img, lo, hi)
    nib.save(ci, crop_path)
    ck = crop_to(img, lo, hi, arr=kid)
    ck.set_data_dtype(np.uint8)
    nib.save(ck, kidney_path)
    kml = float((np.asanyarray(ck.dataobj) > 0).sum()) * float(np.prod(zooms)) / 1000.0
    return dict(shape=tuple(int(v) for v in ci.shape), organs=found,
                kidney_ml=round(kml, 1), zooms=[float(z) for z in zooms],
                segmentation_path=path_used)
