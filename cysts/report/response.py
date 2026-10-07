"""The JSON contract returned to callers.

    {
      "study_iuid": "...",
      "study_prediction": "Abnormal",
      "findings": {
        "cysts_prediction": true,
        "total_cysts": 3,
        "cysts": {
          "right_kidney": [],
          "left_kidney": [ {"cysts_id": 1, "size_mm": "11.7 x 9.4", "density_hu": 20}, ... ]
        },
        "largest_cysts_mm": 11.7,
        "max_density_hu": 577,
        "slice_thickness_mm": 1.25
      }
    }

size_mm is "height x width" -- the largest anterior-posterior and transverse extents,
measured in-plane. That is the form a report quotes ("8 x 8 mm cortical cyst"), so the
two are comparable. largest_cysts_mm is the single largest of those extents across all
cysts, so it is on the same scale as the sizes beneath it rather than a 3D diameter that
would always read larger.

Cysts are numbered 1..N across the whole study, largest first, so cysts_id is unique
within a response and not per-kidney.

TWO NOTES ON THIS CONTRACT, both deliberate.

study_prediction is hard-coded "Abnormal" as specified. It therefore carries NO
information -- a study with no cysts returns "Abnormal" exactly as one with three does.
Read `findings.cysts_prediction` for the answer. This is flagged because a downstream
reader that treats study_prediction as a verdict will mark every scan abnormal.

A lesion the model found but could not assign to a kidney (no overlap with either
TotalSegmentator kidney mask) goes in `unassigned`, never silently into one side. Those
are rare -- 6 of 353 on the validation cohort -- and all 3 beyond 15 mm from any kidney
were genuine errors, so hiding them in a kidney list would launder a known failure.
"""
EXTRA_KEYS = ("not_assessed", "limitations", "rejected", "timing_s",
              "series_name", "series_phase", "model", "version")

# per-cyst keys, in order. slice_image is last so a human reading the JSON is not
# scrolling past 80 KB of base64 to reach the numbers.
CYST_KEYS = ("cysts_id", "size_mm", "density_hu", "slice_number", "slice_image")


def build(study_iuid, comps, ctx, positive, extras=None):
    """comps: reported components, largest first. ctx: per-study context from detect."""
    buckets = {"right_kidney": [], "left_kidney": []}
    unassigned = []
    for n, c in enumerate(comps, 1):
        entry = {"cysts_id": n,
                 "size_mm": c.get("size_mm"),
                 "density_hu": c.get("density_hu"),
                 # DICOM InstanceNumber, so it can be opened in PACS. null when the
                 # mapping could not be built -- never an array index dressed up as one.
                 "slice_number": c.get("slice_number"),
                 "slice_image": c.get("slice_image")}
        side = c.get("side")
        if side == "left":
            buckets["left_kidney"].append(entry)
        elif side == "right":
            buckets["right_kidney"].append(entry)
        else:
            unassigned.append(entry)

    sizes = [v for c in comps for v in (c.get("height_mm"), c.get("width_mm"))
             if v is not None]
    findings = {
        "cysts_prediction": bool(positive and comps),
        "total_cysts": len(comps),
        "cysts": buckets,
        "largest_cysts_mm": round(max(sizes), 1) if sizes else 0.0,
        "max_density_hu": (max(c["density_hu"] for c in comps) if comps else None),
        "slice_thickness_mm": ctx.get("slice_thickness_mm"),
    }
    if unassigned:
        findings["cysts"]["unassigned"] = unassigned
    out = {"study_iuid": study_iuid,
           "study_prediction": "Abnormal",
           "findings": findings}
    if extras:
        out["findings"].update({k: v for k, v in extras.items() if k in EXTRA_KEYS})
    return out
