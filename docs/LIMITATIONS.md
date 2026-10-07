# Limitations

**This is a second reader.** Not validated for autonomous reporting.

## Not assessed

Bosniak grade, renal tumour, **hydronephrosis**, ureter, bladder, adrenal. Reported as
`null`, never `0`. A caller that reads "not assessed" as "absent" will eventually be
wrong about a patient.

## Measured failure modes

Radiologist adjudication of all 95 flagged cases from 752 consecutive studies. Of 42
genuine false positives:

| Cause | n |
|---|---|
| Hyperdense renal lesion called a cyst | 17 |
| Dilated collecting system / hydronephrosis | 13 |
| Normal cortex or parenchyma overcalled | 6 |
| Motion artefact | 5 |
| Calculus / calcification | 1 |

**Hydronephrosis** is a training-data gap: 4 hydronephrosis-only studies against 16 that
have both findings. Severe hydronephrosis produces a false positive 44% of the time
against a 12% baseline. It is invisible to post-processing — a dilated pelvicalyceal
system matches a cyst on density, containment, shape and depth.

## Detection falls off with size

| Cyst volume | Recall |
|---|---|
| > 2 ml | 1.00 |
| 1–2 ml | 0.83 |
| 0.5–1 ml | 0.73 |
| 0.25–0.5 ml | 0.64 |
| **< 0.25 ml** | **0.41** |

Equivalently by craniocaudal extent: 0.96 for lesions spanning >16 slices, **0.30** for
1–2 slices.

## Plain CT is harder

11 of 15 misses were non-contrast series. Cyst-to-parenchyma contrast is ~20 HU
unenhanced against ~83 HU enhanced — four times less signal. In the 752-study cohort only
7% of studies contained a venous-phase series at all, so this is the normal operating
condition, not an edge case.

## Reference-dependence of the metrics

Precision on the same predictions: **0.560** against report text, **0.769** against
radiologist adjudication. 38 of 80 apparent false positives were real cysts the report
omitted. Never quote a metric from this pipeline without naming its reference.

## Segmentation quality

Median Dice 0.75, mean 0.59. The distribution is bimodal: when a cyst is found it is
segmented well (instance Dice 0.82, boundary error 1.8 mm median), and complete misses
pull the mean down. Mean Dice alone describes neither population.

Ground-truth masks were drawn slice by slice in 2D, so they step between slices while the
model predicts smooth 3D surfaces. A median cyst spans 7 slices, so a one-slice boundary
disagreement is a large fraction of the object — small-cyst Dice understates real
performance, and detection is the fairer metric at that size.
