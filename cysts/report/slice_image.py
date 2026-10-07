"""One axial slice as a base64 PNG, with the cyst outlined in red.

WHAT IS DRAWN. The slice the lesion is largest on, windowed WL 60 / WW 400 (abdomen), with
the predicted cyst filled translucent red and outlined solid red. Orientation is canonical
RAS then the same display transform as the review sheets: anterior at the top, patient's
right on the viewer's left. Getting that wrong does not look wrong -- it looks like a cyst
in the other kidney.

WHY PIL AND NOT MATPLOTLIB. Size and speed. matplotlib renders a figure with padding and
antialiasing and lands around 300 KB for one slice; this writes an 8-bit PNG of exactly the
pixels, typically 60-90 KB, and base64 inflates by a third. A response carrying several
cysts is then tens of KB rather than megabytes.
"""
import base64
import io

import numpy as np
from PIL import Image, ImageDraw

WL, WW = 60.0, 400.0
MAX_DIM = 512
FILL_ALPHA = 0.40
RED = (230, 40, 40)


def _window(sl):
    lo, hi = WL - WW / 2, WL + WW / 2
    return np.clip((sl - lo) / (hi - lo), 0, 1)


def _display(arr2d):
    """Match the review-sheet convention: anterior up, patient right on viewer left."""
    return arr2d[::-1, :].T[::-1, :]


def render(ct_canon, mask_canon, z, max_dim=MAX_DIM, box=None):
    """base64 PNG of slice z, cyst in red. `box` crops to (y0, y1, x0, x1) if given."""
    sl = _display(_window(ct_canon[:, :, z].astype(np.float32)))
    m = _display(mask_canon[:, :, z].astype(bool))
    if box:
        y0, y1, x0, x1 = box
        sl, m = sl[y0:y1, x0:x1], m[y0:y1, x0:x1]
    g = (sl * 255).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1).astype(np.float32)
    if m.any():
        for c in range(3):
            rgb[..., c] = np.where(m, (1 - FILL_ALPHA) * rgb[..., c] + FILL_ALPHA * RED[c],
                                   rgb[..., c])
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    if m.any():                                   # 1px outline, so a small cyst is findable
        edge = m & ~_erode(m)
        ys, xs = np.nonzero(edge)
        d = ImageDraw.Draw(img)
        d.point(list(zip(xs.tolist(), ys.tolist())), fill=RED)
    if max(img.size) > max_dim:
        s = max_dim / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _erode(m):
    out = m.copy()
    for ax in (0, 1):
        for sh in (1, -1):
            out &= np.roll(m, sh, axis=ax)
    return out
