"""Contact sheet per study: CT | kidney | predicted cyst, one row per cyst-bearing slice.

ORIENTATION. Every volume is converted to canonical RAS before display. This is not
fussiness: the crops arrive as LAS, the training data as LPI and LPS, and the display
transform `[::-1, :].T` with origin='lower' is correct for exactly one of those. Applied
blind it renders LAS anterior-up and LPI upside down, and left-right mirrored in all
cases -- so a left renal cyst appears on the right of the image and anyone reading
laterality off the picture gets it backwards. as_closest_canonical first, then the
transform, gives anterior at top and patient-right on the viewer's left for every source.

ROW SELECTION. Slices carrying the evidence, or the sheet is blank for a negative study:
cyst slices when there is a prediction, else evenly spaced kidney slices -- a blank page
says nothing, and what is worth seeing on a negative is that the kidney is clean.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from ..common import paths

WL, WW = 60, 400
CLASSES = {1: ("kidney", (0.36, 0.55, 0.94)), 2: ("cyst", (0.90, 0.28, 0.31))}
CYST = 2
ALPHA = 0.45


def window(sl):
    lo, hi = WL - WW / 2, WL + WW / 2
    return np.clip((sl - lo) / (hi - lo), 0, 1)


def overlay(gray01, lab):
    rgb = np.repeat(gray01[..., None], 3, axis=-1)
    for l, (_, col) in CLASSES.items():
        m = lab == l
        if m.any():
            for c in range(3):
                rgb[..., c] = np.where(m, (1 - ALPHA) * rgb[..., c] + ALPHA * col[c], rgb[..., c])
    return np.clip(rgb, 0, 1)


def _canon(p):
    return nib.as_closest_canonical(nib.load(p))


def sheet(study_id, out_dir, max_rows=24, dpi=140, panel_in=4.0):
    crop = os.path.join(paths.CROPS, f"{study_id}_0000.nii.gz")
    pred = os.path.join(paths.PRED, f"{study_id}.nii.gz")
    kidp = os.path.join(paths.KIDNEY, f"{study_id}.nii.gz")
    im = _canon(crop)
    ct = np.asanyarray(im.dataobj).astype(np.float32)
    pr = np.asanyarray(_canon(pred).dataobj).astype(np.uint8)
    kid = (np.asanyarray(_canon(kidp).dataobj) > 0).astype(np.uint8) if os.path.exists(kidp) \
        else np.zeros_like(pr)

    pz = np.unique(np.argwhere(pr == CYST)[:, 2]) if (pr == CYST).any() else np.array([], int)
    if pz.size:
        zs = pz[np.linspace(0, pz.size - 1, max_rows).astype(int)] if pz.size > max_rows else pz
        sub = pz.size > max_rows
    else:
        kz = np.unique(np.argwhere(kid > 0)[:, 2])
        if kz.size == 0:
            return None
        n = min(max_rows, 8)
        zs, sub = kz[np.linspace(0, kz.size - 1, n).astype(int)], kz.size > n
    vox_ml = float(np.abs(np.linalg.det(im.affine[:3, :3]))) / 1000.0
    p_ml = float((pr == CYST).sum()) * vox_ml

    n = zs.size
    fig, ax = plt.subplots(n, 3, figsize=(3 * panel_in, panel_in * n),
                           squeeze=False, facecolor="white")
    for r, z in enumerate(zs):
        sl = window(ct[:, :, z])[::-1, :].T
        for c, lab in enumerate((None, kid[::-1, :, z].T, pr[::-1, :, z].T)):
            a = ax[r][c]
            a.imshow(sl if lab is None else overlay(sl, lab), cmap="gray",
                     origin="lower", interpolation="bilinear")
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
        ax[r][0].set_ylabel(f"z={z}", fontsize=9, rotation=0, labelpad=26, va="center")
        if r == 0:
            for c, t in enumerate(("CT", "kidney (TotalSegmentator)", "predicted cyst")):
                ax[r][c].set_title(t, fontsize=12, fontweight="bold")
    note = f"   [{n} rows, evenly sampled]" if sub else ""
    fig.suptitle(f"{study_id}   predicted cyst {p_ml:.2f} ml   {n} slice(s){note}\n"
                 f"blue = kidney, red = cyst   |   anterior up, patient right on the left",
                 fontsize=13, fontweight="bold", y=1.0 - 0.0015 * n)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.012 * min(n, 6) / 6])
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{study_id}.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study-id", help="one study; omit to render every prediction")
    ap.add_argument("--out", default=paths.OVERLAYS)
    ap.add_argument("--max-rows", type=int, default=24)
    a = ap.parse_args()
    ids = [a.study_id] if a.study_id else [
        os.path.basename(p)[:-7] for p in sorted(glob.glob(os.path.join(paths.PRED, "*.nii.gz")))]
    ok = 0
    for s in ids:
        try:
            if sheet(s, a.out, max_rows=a.max_rows):
                ok += 1
        except Exception as e:
            print(f"  {s}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(ids)} sheets -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
