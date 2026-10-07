"""Which predictions to report, and which filters were tried and rejected.

ONLY TWO RULES SURVIVED MEASUREMENT.

  1. MINIMUM STUDY VOLUME. Call a study positive when the surviving predicted cyst
     volume reaches a threshold. On 752 clinical studies, against the report:

         threshold   precision  recall   F1
         0.05 ml       0.560     0.872   0.682   (no filter)
         0.30 ml       0.669     0.795   0.727
         0.50 ml       0.715     0.752   0.733   <- best F1
         1.00 ml       0.760     0.650   0.700

     It is a per-STUDY total, not a per-component floor, deliberately: that keeps a
     patient with several small cysts (whom a radiologist would report) and drops the
     solitary speck.

  2. KIDNEY CONTAINMENT, as a hard anatomical check only. Dropping components more than
     15 mm from the kidney removed 3 of 353 predictions -- one at 976 mm and -83.7 HU,
     i.e. fat. Precision 0.560 -> 0.570 at zero cost to recall. Small, free, correct.

REJECTED, each with the measurement that rejected it:

  HU WINDOW. Keeping only -10..30 HU cost 16 true positives to remove 17 false ones:
  precision 0.560 -> 0.577, recall 0.872 -> 0.735. Mean HU separates TP from FP at
  AUC 0.517, and the parenchyma-relative delta at AUC 0.510 -- both chance. The reason
  is physical: a contrast-phase cyst reads 52 HU and is still 83 HU darker than its
  kidney, so an absolute window discards it. 89 of the 95 lesions removed were above
  30 HU and most were real.

  SHAPE. Sphericity >= 0.80 gives precision 0.634 at recall 0.829 -- better than the HU
  rule but worse than volume alone, and it removes large exophytic cysts along with the
  branching collecting systems it is aimed at. Within size bands the separation
  collapses, so the apparent signal is mostly lesion size.

  ARTEFACT SCORES on the cyst's own slices. Motion and noise proxies flagged 28 true
  positives against 9 false ones -- a large cyst changes the slice statistics it is
  being judged by, so the detector fires on the lesion, not on artefact.

WHAT IS NOT FIXABLE HERE. Radiologist adjudication of all 95 FP/FN cases found 13 of 42
genuine false positives were hydronephrosis and 17 were hyperdense lesions. Neither is
separable by any post-hoc rule measured above; both need training negatives.
"""
MIN_STUDY_ML = 0.5
MAX_DISTANCE_MM = 15.0


def apply(comps, min_study_ml=MIN_STUDY_ML, max_distance_mm=MAX_DISTANCE_MM):
    """(reported, rejected, study_positive). Nothing is deleted -- rejects are returned
    with a reason, because a caller reviewing a borderline study needs to see them."""
    reported, rejected = [], []
    for c in comps:
        d = c.get("distance_to_kidney_mm")
        if d is not None and d > max_distance_mm:
            rejected.append({**c, "rejected_because":
                             f"{d:.0f} mm from the kidney -- not an intrarenal lesion"})
        else:
            reported.append(c)
    total = sum(c["volume_ml"] for c in reported)
    positive = total >= min_study_ml
    return reported, rejected, bool(positive)
