"""Load test: random studies from a cohort, through the whole pipeline, timed per stage.

WHAT IT MEASURES. Wall clock for download -> prepare -> inference -> measure, per study,
plus the throughput that implies. Two modes:

    --mode lib    call the pipeline in-process. Measures the PIPELINE.
    --mode http   POST /analyze against a running service. Measures the SERVICE, including
                  queueing, serialisation and HTTP overhead -- which is what a caller
                  actually experiences.

Use http for a capacity answer. The library mode cannot show queue wait, and the service
serialises analyses deliberately, so concurrency only shows up through the socket.

WHY SEQUENTIAL BY DEFAULT. The service runs one analysis at a time on purpose (memory and
a shared GPU). Firing 20 concurrent requests does not measure 20x the work; it measures
19 of them waiting. --concurrency exists to measure that wait, not to speed anything up.

THE FIRST STUDY IS AN OUTLIER AND IS REPORTED SEPARATELY. Loading the model costs 41.7 s
once. Averaging it into a 10-study run inflates the per-study figure by ~4 s and makes the
result depend on the sample size, which is meaningless.

A COLD STUDY IS NOT A FAILURE. Prod returns 503 `restoring_from_archive` for anything on
tape, and that can be most of an old cohort. Those are counted and reported separately so
they do not silently become "slow" or "failed".
"""
import argparse
import json
import os
import random
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


def pick_studies(source, n, seed=0, column=None):
    if source.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(source)
    else:
        df = pd.read_csv(source)
    col = column or next((c for c in df.columns if c.lower() in
                          ("study_iuid", "studyinstanceuid", "study_uid")), None)
    if col is None:
        sys.exit(f"no study_iuid column in {source}; columns are {list(df.columns)}")
    ids = df[col].dropna().astype(str).str.strip()
    ids = [i for i in dict.fromkeys(ids) if i]
    random.Random(seed).shuffle(ids)
    return ids[:n]


def _one_lib(iuid, env, min_study_ml=None):
    from cysts.pipeline import run_study
    from cysts.common import download_dicoms
    t0 = time.time()
    try:
        r = run_study.run(iuid, env=env, min_study_ml=min_study_ml)
        t = r["findings"].get("timing_s", {})
        return dict(study_iuid=iuid, status="ok", total_s=round(time.time() - t0, 1),
                    download_s=t.get("download"), prepare_s=t.get("prepare"),
                    inference_s=t.get("inference"), measure_s=t.get("measure"),
                    cached=r["findings"].get("source", {}).get("cached"),
                    n_cysts=r["findings"].get("total_cysts"),
                    cysts_prediction=r["findings"].get("cysts_prediction"),
                    archive_mb=r["findings"].get("source", {}).get("archive_mb"))
    except download_dicoms.ArchiveRestoring as e:
        return dict(study_iuid=iuid, status="archive_restoring",
                    total_s=round(time.time() - t0, 1), error=str(e))
    except Exception as e:
        return dict(study_iuid=iuid, status="failed", total_s=round(time.time() - t0, 1),
                    error=f"{type(e).__name__}: {str(e)[:200]}")


def _one_http(iuid, env, url, timeout):
    import requests
    t0 = time.time()
    try:
        resp = requests.post(url, data=dict(study_iuid=iuid, env=env), timeout=timeout)
        dt = round(time.time() - t0, 1)
        if resp.status_code == 503:
            return dict(study_iuid=iuid, status="archive_restoring", total_s=dt,
                        retry_after=resp.headers.get("Retry-After"))
        if resp.status_code != 200:
            return dict(study_iuid=iuid, status="failed", total_s=dt,
                        error=f"HTTP {resp.status_code}: {resp.text[:200]}")
        r = resp.json()
        t = r.get("findings", {}).get("timing_s", {})
        return dict(study_iuid=iuid, status="ok", total_s=dt, wall_s=dt,
                    download_s=t.get("download"), prepare_s=t.get("prepare"),
                    inference_s=t.get("inference"), measure_s=t.get("measure"),
                    server_total_s=t.get("total"),
                    n_cysts=r.get("findings", {}).get("total_cysts"),
                    response_kb=round(len(resp.content) / 1024, 1))
    except Exception as e:
        return dict(study_iuid=iuid, status="failed", total_s=round(time.time() - t0, 1),
                    error=f"{type(e).__name__}: {str(e)[:200]}")


def report(rows, elapsed, concurrency):
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    print("\n" + "=" * 78)
    print(f"{'LOAD TEST RESULT':^78}")
    print("=" * 78)
    print(f"  studies attempted      {len(df)}")
    print(f"  completed              {len(ok)}")
    for s in ("archive_restoring", "failed"):
        k = int((df.status == s).sum())
        if k:
            print(f"  {s:<22} {k}")
    if not len(ok):
        print("\n  nothing completed -- see the per-study table")
        return df
    first, rest = ok.iloc[:1], ok.iloc[1:]
    print(f"\n  first study            {first.total_s.iloc[0]:.1f} s  "
          f"(includes ~42 s one-off model load -- excluded from the figures below)")
    pool = rest if len(rest) else ok

    def q(s, p):
        v = sorted(s.dropna())
        return v[min(int(len(v) * p), len(v) - 1)] if v else float("nan")

    print(f"\n  per study, n={len(pool)}")
    print(f"    {'stage':<14}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}")
    for col, lbl in (("download_s", "download"), ("prepare_s", "prepare"),
                     ("inference_s", "inference"), ("measure_s", "measure"),
                     ("total_s", "TOTAL")):
        if col in pool and pool[col].notna().any():
            s = pool[col].dropna()
            print(f"    {lbl:<14}{s.mean():>9.1f}{s.median():>9.1f}"
                  f"{q(s, .95):>9.1f}{s.max():>9.1f}")
    tot = pool.total_s.dropna()
    if len(tot):
        print(f"\n  sustained throughput   {3600 / tot.mean():.0f} studies/hour "
              f"at concurrency {concurrency}")
    print(f"  wall clock             {elapsed:.0f} s for {len(df)} studies "
          f"({len(df) / max(elapsed / 3600, 1e-9):.0f}/hour end to end)")
    if "cached" in ok and ok.cached.notna().any():
        nc = int((~ok.cached.astype(bool)).sum())
        print(f"  freshly downloaded     {nc} of {len(ok)} "
              f"(the rest were already on disk -- download_s ~0)")
    if "n_cysts" in ok:
        print(f"  cysts found            {int(ok.n_cysts.fillna(0).sum())} across "
              f"{int((ok.n_cysts.fillna(0) > 0).sum())} studies")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="xlsx/csv with a study_iuid column")
    ap.add_argument("-n", "--count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env", default="prod")
    ap.add_argument("--mode", choices=("lib", "http"), default="lib")
    ap.add_argument("--url", default="http://localhost:8089/analyze")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--min-study-ml", type=float)
    ap.add_argument("--out", default="load_test_results.csv")
    ap.add_argument("--prefetch", type=int, default=0,
                    help="download this many studies ahead of the GPU (lib mode). "
                         "0 = serial, the honest baseline")
    a = ap.parse_args()

    ids = pick_studies(a.source, a.count, seed=a.seed)
    print(f"{len(ids)} studies sampled from {os.path.basename(a.source)} "
          f"(seed {a.seed}) | mode={a.mode} env={a.env} concurrency={a.concurrency}",
          flush=True)
    fn = ((lambda i: _one_lib(i, a.env, a.min_study_ml)) if a.mode == "lib"
          else (lambda i: _one_http(i, a.env, a.url, a.timeout)))
    rows = []
    t0 = time.time()
    if a.mode == "lib" and a.prefetch > 0:
        # downloads overlap with GPU work; the per-study download_s then reads ~0 because
        # the study was already on disk when its turn came. That is the point, but it
        # means download time disappears from the per-stage table and shows up only in
        # the wall clock -- which is the number that matters.
        from cysts.common.prefetch import Prefetcher
        from cysts.pipeline import infer_study
        pf = Prefetcher(ids, env=a.env, depth=a.prefetch, workers=min(a.prefetch, 3))
        for k, (iuid, dest, info, err) in enumerate(pf, 1):
            if err is not None:
                r = dict(study_iuid=iuid, status="failed", total_s=0.0,
                         error=f"{type(err).__name__}: {str(err)[:200]}")
            else:
                ts = time.time()
                try:
                    res = infer_study.analyse(iuid, dest, study_iuid=iuid,
                                              min_study_ml=a.min_study_ml)
                    t = res["findings"].get("timing_s", {})
                    r = dict(study_iuid=iuid, status="ok",
                             total_s=round(time.time() - ts, 1),
                             download_s=0.0, prepare_s=t.get("prepare"),
                             inference_s=t.get("inference"), measure_s=t.get("measure"),
                             n_cysts=res["findings"].get("total_cysts"),
                             archive_mb=round(info.get("zip_bytes", 0) / 1e6, 1) or None)
                except Exception as e:
                    r = dict(study_iuid=iuid, status="failed",
                             total_s=round(time.time() - ts, 1),
                             error=f"{type(e).__name__}: {str(e)[:200]}")
            rows.append(r)
            print(f"  [{k:3d}/{len(ids)}] {iuid[-28:]:<28} {r['status']:<18}"
                  f"{r.get('total_s', 0):7.1f}s  cysts={r.get('n_cysts')}", flush=True)
    elif a.concurrency <= 1:
        for k, i in enumerate(ids, 1):
            r = fn(i)
            rows.append(r)
            print(f"  [{k:3d}/{len(ids)}] {i[-28:]:<28} {r['status']:<18}"
                  f"{r.get('total_s', 0):7.1f}s  cysts={r.get('n_cysts')}", flush=True)
    else:
        with ThreadPoolExecutor(a.concurrency) as ex:
            for k, r in enumerate(ex.map(fn, ids), 1):
                rows.append(r)
                print(f"  [{k:3d}/{len(ids)}] {r['study_iuid'][-28:]:<28} "
                      f"{r['status']:<18}{r.get('total_s', 0):7.1f}s", flush=True)
    df = report(rows, time.time() - t0, a.concurrency)
    df.to_csv(a.out, index=False)
    print(f"\nper-study detail -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
