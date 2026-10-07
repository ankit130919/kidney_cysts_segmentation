"""Score a directory of predictions against a truth table.

Reads the per-study results an inference run produced and joins them to a reference
column, then prints the confusion matrix and the volume sweep.
"""
import argparse
import glob
import json
import os

import pandas as pd

from ..common import paths
from ..kidney import detect_cysts, postprocess
from .metrics import score, sweep_volume


def collect(pred_dir=None, crops=None, kidney=None, min_study_ml=None):
    pred_dir = pred_dir or paths.PRED
    crops = crops or paths.CROPS
    kidney = kidney or paths.KIDNEY
    rows = []
    for p in sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz"))):
        sid = os.path.basename(p)[:-7]
        crop = os.path.join(crops, f"{sid}_0000.nii.gz")
        kid = os.path.join(kidney, f"{sid}.nii.gz")
        if not os.path.exists(crop):
            continue
        comps, _ = detect_cysts.components(p, crop, kid if os.path.exists(kid) else None)
        kw = {} if min_study_ml is None else dict(min_study_ml=min_study_ml)
        reported, rejected, positive = postprocess.apply(comps, **kw)
        s = detect_cysts.summarise(reported)
        rows.append(dict(study_id=sid, pred_cyst="yes" if positive else "no",
                         n_rejected=len(rejected), **s))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir")
    ap.add_argument("--truth-csv", help="CSV with study_id and the reference column")
    ap.add_argument("--truth-col", default="report_cyst")
    ap.add_argument("--reference", default="(unspecified -- say what the truth column is)")
    ap.add_argument("--out", default=os.path.join(paths.CSV, "cohort_scores.csv"))
    a = ap.parse_args()
    df = collect(a.pred_dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} studies scored -> {a.out}")
    print(f"  predicted positive: {int((df.pred_cyst == 'yes').sum())}")
    if a.truth_csv:
        t = pd.read_csv(a.truth_csv)
        t["study_id"] = t.study_id.astype(str)
        df["study_id"] = df.study_id.astype(str)
        j = df.merge(t[["study_id", a.truth_col]], on="study_id", how="inner")
        print(f"\nn = {len(j)}   reference: {a.reference}")
        for k, v in score(j, a.truth_col, "pred_cyst").items():
            print(f"  {k:<20}{v}")
        print("\nvolume sweep")
        print(sweep_volume(j, a.truth_col, "total_volume_ml").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
