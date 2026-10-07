# Open issues

## 1. Inverted slice axis in 11% of training masks — OPEN, affects the shipped model

Measuring each annotation mask's distance from the TotalSegmentator kidney in the same
volume: **71 of 631 annotated cases sit more than 20 mm from any kidney**, up to 241 mm.
A renal cyst is inside a kidney; a mask 241 mm away is not a cyst label.

Independently confirmed on the validation split by comparing predictions against the
ground truth as stored and z-flipped: 9 of 62 cyst-positive series score Dice 0.00 as
stored and 0.38–0.86 flipped.

A fix was applied on 23 Sep and masks were rebuilt, but verification on 30 Sep shows the
defect persists. **59 of the affected cases are in the training split of the shipped
model.**

Impact is confined to overlap, not detection: correcting the validation split moves mean
Dice 0.585 → 0.640 while study-level precision and recall are unchanged (0.980 / 0.892).
So the reported performance is sound but slightly pessimistic, and roughly 9% of the
training signal is wasted.

**Fix**: detect per series by testing both orientations against a TotalSegmentator kidney
mask — a cyst must be inside a kidney, which is a far stronger check than the HU-based
one that missed this.

## 2. Hydronephrosis has no corrective training signal

4 hydronephrosis-only studies against 16 with both findings. A dataset adding ~92
hydronephrosis-only hard negatives (cyst channel empty, no new annotation needed) was
built and then reverted before training; the selection logic and its gates are
reconstructible from this repo's history.

## 3. Hyperdense renal lesions

The largest single FP category (17 of 42) and not yet investigated. Likely needs hard
negatives of solid/hyperdense renal lesions.

## 4. Two truncated files reached the shipped training set

`cyst_1204` (train) and `cyst_1209` (validation) were truncated on write yet preprocessed
successfully, so the shipped checkpoint trained on one corrupt image and validated
against another. Both were regenerated on 30 Sep, after training. Negligible at 2/781,
but the validation figure predates the fix.

Truncated `.nii.gz` files broke three separate runs during development, every time
because a guard checked `os.path.exists` on a gzip. `prepare_study.readable()` decodes
instead. Any new ingest path should do the same.

## 5. Nothing authenticates the API

Anyone who can reach the port can submit a study id and read findings.
