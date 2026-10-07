"""Confusion matrix and the metrics that go with it.

ALWAYS STATE THE REFERENCE. On the same 752 studies and the same predictions, precision
is 0.560 against report text and 0.769 against radiologist adjudication. The model did
not change. A metric quoted without its reference is not a measurement.
"""
import argparse

import numpy as np
import pandas as pd


def confusion(y_true, y_pred):
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    return dict(TP=int((y & p).sum()), FP=int((~y & p).sum()),
                FN=int((y & ~p).sum()), TN=int((~y & ~p).sum()))


def metrics(TP, FP, FN, TN):
    n = TP + FP + FN + TN
    pr = TP / max(TP + FP, 1)
    rc = TP / max(TP + FN, 1)
    sp = TN / max(TN + FP, 1)
    den = np.sqrt(float(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                precision=round(pr, 4), recall=round(rc, 4), specificity=round(sp, 4),
                NPV=round(TN / max(TN + FN, 1), 4),
                F1=round(2 * pr * rc / max(pr + rc, 1e-9), 4),
                accuracy=round((TP + TN) / max(n, 1), 4),
                balanced_accuracy=round((rc + sp) / 2, 4),
                MCC=round((TP * TN - FP * FN) / den if den else 0.0, 4))


def score(df, truth_col, pred_col, positive="yes"):
    y = df[truth_col].astype(str).str.lower().eq(str(positive).lower())
    p = df[pred_col].astype(str).str.lower().eq(str(positive).lower())
    return metrics(**confusion(y, p))


def sweep_volume(df, truth_col, volume_col, thresholds=(0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0),
                 positive="yes"):
    """Precision/recall against a per-study volume threshold. 0.5 ml was best on n=752."""
    y = df[truth_col].astype(str).str.lower().eq(str(positive).lower())
    rows = []
    for t in thresholds:
        rows.append(dict(threshold_ml=t, **metrics(**confusion(y, df[volume_col] > t))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("--truth", default="report_cyst")
    ap.add_argument("--pred", default="pred_cyst")
    ap.add_argument("--volume", help="column of per-study predicted ml, for a sweep")
    ap.add_argument("--reference", required=True,
                    help="what the truth column IS, e.g. 'radiologist adjudication' or "
                         "'report text'. Printed with the metrics; there is no default "
                         "because a metric without its reference is not a measurement.")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    m = score(df, a.truth, a.pred)
    print(f"n = {len(df)}   reference: {a.reference}\n")
    print(f"           pred+   pred-")
    print(f"  actual+ {m['TP']:6d} {m['FN']:6d}")
    print(f"  actual- {m['FP']:6d} {m['TN']:6d}\n")
    for k, v in m.items():
        if k not in ("TP", "FP", "FN", "TN"):
            print(f"  {k:<20}{v}")
    if a.volume:
        print("\nper-study volume threshold sweep")
        print(sweep_volume(df, a.truth, a.volume).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
