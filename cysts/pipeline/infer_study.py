"""One study in, kidney cysts out.

    prepare (series -> TotalSegmentator -> crop)  ->  cyst model  ->  measure  ->  filter

TIMING, measured: ~19 s preparation, ~3.8 s inference, <1 s measurement. TotalSegmentator
is ~85% of it. Optimising the cyst model is close to pointless.

WHAT THIS REPORTS AND WHAT IT DOES NOT. Every result carries `not_assessed` and
`limitations`, and they are not boilerplate. This model finds CYSTS. It does not grade
them (Bosniak), does not assess renal tumours, and does not report hydronephrosis --
despite hydronephrosis being, by radiologist adjudication, 13 of its 42 genuine false
positives. A caller that cannot tell "we looked and found nothing" from "we did not look"
will eventually read one as the other.
"""
import argparse
import json
import os
import time

from ..common import paths
from ..kidney import detect_cysts, postprocess
from ..report import attach_slices, response
from . import prepare_study, seg_fastpredict

VERSION = "0.1.0"


def analyse(study_id, study_dir, gpu=None, force=False, min_study_ml=None,
            study_iuid=None):
    t0 = time.time()
    prep = prepare_study.prepare(study_id, study_dir, gpu=gpu, force=force)
    t_prep = time.time()

    pdir = paths.pred_dir()
    paths.ensure(pdir)
    stage = paths.stage_dir(study_id)
    os.makedirs(stage, exist_ok=True)
    link = os.path.join(stage, os.path.basename(prep["crop"]))
    if not os.path.exists(link):
        os.symlink(os.path.abspath(prep["crop"]), link)
    seg_fastpredict.predict(stage, pdir)
    pred = paths.pred_path(study_id)
    t_inf = time.time()

    comps, ctx = detect_cysts.components(pred, prep["crop"], prep["kidney"])
    kw = {} if min_study_ml is None else dict(min_study_ml=min_study_ml)
    reported, rejected, positive = postprocess.apply(comps, **kw)
    attach_slices.attach(reported, prep["crop"], pred, slice_map=prep.get("slice_map"))
    summary = detect_cysts.summarise(reported)

    extras = dict(
        model=paths.DATASET, version=VERSION,
        series_name=prep.get("series"), series_phase=prep.get("series_tier"),
        rejected=rejected,
        not_assessed=dict(
            bosniak_grade=None, renal_tumour=None, hydronephrosis=None,
            ureter=None, bladder=None, adrenal=None),
        limitations=[
            "Second reader. Not validated for autonomous reporting.",
            "study_prediction is always 'Abnormal' by contract and carries no "
            "information -- read findings.cysts_prediction.",
            "Cysts only. Hydronephrosis is NOT assessed and is a known confuser: "
            "13 of 42 adjudicated false positives were a dilated collecting system.",
            "17 of 42 adjudicated false positives were hyperdense renal lesions.",
            "Small-cyst recall is limited: 41% below 0.25 ml against 100% above 2 ml.",
            "Plain CT is harder than contrast -- cyst-to-parenchyma contrast is ~20 HU "
            "unenhanced against ~83 HU enhanced; 11 of 15 misses were plain series.",
        ],
        timing_s=dict(prepare=round(t_prep - t0, 1), inference=round(t_inf - t_prep, 1),
                      measure=round(time.time() - t_inf, 1),
                      total=round(time.time() - t0, 1)),
    )
    out = response.build(study_iuid or str(study_id), reported, ctx, positive,
                         extras=extras)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study-id", required=True)
    ap.add_argument("--study-iuid", help="echoed into the response; defaults to --study-id")
    ap.add_argument("--study-dir", required=True, help="directory of DICOM files")
    ap.add_argument("--gpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-study-ml", type=float)
    ap.add_argument("--json", help="write the result here as well as printing it")
    a = ap.parse_args()
    r = analyse(a.study_id, a.study_dir, gpu=a.gpu, force=a.force,
                min_study_ml=a.min_study_ml, study_iuid=a.study_iuid)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        json.dump(r, open(a.json, "w"), indent=2)
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
