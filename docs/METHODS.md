# Methods

## Model

nnU-Net `3d_fullres`, fine-tuned from TotalSegmentator `Dataset789_kidney_cyst_501subj`.
Labels: 1 kidney, 2 kidney_cyst. Target spacing 1.5 mm isotropic, patch 160×112×128.

Training data (`Dataset795_kidney_cropped`, 781 cases):

| Source | Cases | Studies |
|---|---|---|
| Radiologist polygon annotations, 5C | 634 | 353 |
| Verified cyst-free normal kidneys | 147 | 147 |

By report, those 500 studies are 309 cyst-only, 166 normal, 16 both cyst and
hydronephrosis, 4 hydronephrosis-only.

Plans — architecture, spacing and CT normalisation — are inherited verbatim from the
pretrained model. That is a requirement, not a convenience: `-pretrained_weights` needs
an identical network, and recomputing intensity statistics from a changed label space
shifts the normalisation (on an earlier build, foreground mean moved 105 → 35 HU because
the label set changed from kidney+cyst to cyst-only) and silently wastes the warm start.

## Annotation provenance

Polygons were drawn per axial slice in patient millimetres and rasterised against the
DICOM geometry. Three defects were found and fixed during the build, each caught by
checking HU inside the resulting mask rather than by the code exiting cleanly:

- the slice key is `instance_number`, not `instance_name` — the wrong one placed masks on
  pelvic bone at 500–1200 HU
- GDCM fails on some vendor headers; a pydicom fallback builds the volume instead
- study ids parsed as `"1,954"` with thousand separators

A fourth is **still open**: 71 of 631 annotated cases (11.3%) have masks more than 20 mm
from any kidney — up to 241 mm — consistent with an inverted slice axis. See `OPEN.md`.

## Crop

TotalSegmentator `total`, restricted to kidney_left, kidney_right, liver, spleen, colon;
bounding box of their union padded 10 mm. This reproduces TotalSegmentator's own
`kidney_cysts` crop. It is a box, not a mask — exophytic cysts bulge past the renal
contour and masking to the organ would amputate them.

## Measurement

- **Volume**: component voxel count × voxel volume. No surface correction.
- **Diameter**: true 3D maximum Feret diameter via convex hull. Not the bounding-box
  diagonal; comparable to a radiologist's in-plane pair only for round lesions.
- **HU**: read from the cropped CT the model saw.
- **Containment**: fraction inside the TotalSegmentator kidney mask. Tracks lesion size,
  not correctness — median containment falls from 1.00 below 0.25 ml to 0.21 above 20 ml,
  because a large cyst cannot fit inside a kidney.

Components below 20 mm³ are dropped throughout — about six voxels at the target spacing.

## Decision rule

A study is positive when total reported cyst volume reaches 0.5 ml. Chosen by sweep on
752 studies (best F1 0.733). Components more than 15 mm from the kidney are rejected
first.

## What was tried and rejected

| Filter | Result |
|---|---|
| HU window −10..30 | precision +0.017, recall −0.137. Mean HU separates TP from FP at AUC 0.517 |
| Parenchyma-relative HU delta | AUC 0.510 — chance |
| Sphericity ≥ 0.80 | precision 0.634 at recall 0.829 — worse than volume alone |
| Cortical depth | AUC 0.500 |
| Motion/noise artefact scores | flagged 28 TP against 9 FP |
| Kidney-crop cascade | 100% of predictions already inside kidney bbox + 10 mm |

The common thread: every feature that looked discriminative was confounded with lesion
size. Volume is the only filter that survived.
