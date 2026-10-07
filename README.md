# cysts

Kidney cyst detection, segmentation and measurement on **abdominal CT**, plain or contrast.

The only learned component is an nnU-Net fine-tuned from TotalSegmentator's
`Dataset789_kidney_cyst_501subj` on 353 radiologist-annotated 5C studies plus 147
verified-normal kidneys. TotalSegmentator itself supplies the anatomy — the pipeline
crops to the kidney region before the cyst model ever runs, because that is the field of
view the model was trained on.

## What it produces

Per cyst: volume, maximum 3D diameter, mean/min/max HU, sphericity, side, slice range,
and containment within the TotalSegmentator kidney mask. Per study: cyst present or not,
count, total and largest volume.

The response reports Bosniak grade, renal tumour, hydronephrosis, ureter and bladder as
`null`, never `0` — they are **not assessed**, which is different from absent.

## Where it stands

Against **radiologist adjudication** of every flagged case on 752 consecutive clinical
studies:

| | Precision | Recall | Specificity | F1 |
|---|---|---|---|---|
| Per study (n=752) | **0.769** | **0.909** | 0.930 | 0.833 |

Against **manual segmentation masks** on the held-out validation split:

| | Precision | Recall | Specificity | F1 |
|---|---|---|---|---|
| Per study (n=152) | 0.982 | 0.886 | 0.931 | 0.932 |
| Per slice (n=70,047) | 0.817 | 0.802 | 0.993 | 0.809 |

Overlap on cyst-bearing studies: median Dice **0.75**, mean 0.59. The gap between those
two is the whole story of this model — see below.

### Read the two tables as one thing

The first time these 752 studies were scored, precision came out at **0.560**. The
reference was the radiology report. Adjudication showed **38 of the 80 "false positives"
were real cysts the report never mentioned** — so precision was 0.769 all along, and the
report was the weak reference, not the model.

This is measurable rather than asserted: the false positives were indistinguishable from
true positives on every image feature tested — mean HU (AUC 0.517), parenchyma-relative
contrast (0.510), kidney containment, cortical depth (0.500), position along the kidney
(0.511). They differed only in being smaller and more often solitary, which is the
profile of an incidental cyst a radiologist does not bother writing down.

**Do not score this model against report text and quote the result as precision.**

## Known failure modes

From radiologist adjudication of the 42 genuine false positives:

| Cause | n |
|---|---|
| Hyperdense renal lesion called a cyst | 17 |
| **Dilated collecting system / hydronephrosis** | **13** |
| Normal cortex or parenchyma overcalled | 6 |
| Motion artefact | 5 |
| Calculus / calcification | 1 |

Hydronephrosis is a training-data gap, not a thresholding problem. The training set holds
**4** studies that are hydronephrosis without a cyst against **16** that have both — a
4:1 signal in the wrong direction. Severe hydronephrosis produces a false positive 44% of
the time against a 12% baseline. No post-hoc filter catches it, because a dilated
pelvicalyceal system matches a cyst on density, containment, shape and depth.

Misses are dominated by small lesions and by unenhanced scans. Detection is 100% above
2 ml and 41% below 0.25 ml; 11 of 15 misses were plain series, where cyst-to-parenchyma
contrast is ~20 HU against ~83 HU with contrast.

## Pipeline

    DICOM -> pick series -> TotalSegmentator -> crop (organ box + 10 mm)
          -> cyst model -> components -> measure -> filter -> JSON

Series selection prefers contrast-venous, then any contrast, then plain, and excludes
chest, spine and reformat series outright. That exclusion matters more than the
preference: the original selector compared slice counts only and chose a chest
reconstruction for 10% of studies.

## Timing

| Stage | Per study |
|---|---|
| dcm2niix + TotalSegmentator crop | **~19 s** |
| Cyst model inference | **~3.8 s** |
| Measurement | <1 s |
| **End to end** | **~23 s** |

Plus a one-off **41.7 s** to load the model. That is why the service holds the predictor
open: 100 studies through one resident process take ~6.5 minutes; one process per study
takes ~75. TotalSegmentator is ~85% of the marginal cost, so optimising the cyst model
buys almost nothing.

## Use

    pip install -e ".[seg]"
    cysts-study --study-id 9473428 --study-dir /path/to/dicoms

    python app.py          # HTTP API, see DEPLOY.md

## Status

**Second reader.** Not validated for autonomous reporting. Every response carries
`limitations`; they are accurate and should not be stripped.
