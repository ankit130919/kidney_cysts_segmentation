"""study_iuid in, JSON out -- the whole pipeline including the download.

WHY THIS EXISTS SEPARATELY FROM infer_study. infer_study starts from DICOM already on
disk, which is right for re-analysis and for a cohort that is already staged. A caller
passing a study_iuid wants the archive fetch counted too, and the download is the most
variable stage by far: 2 s for a cached study, minutes for a cold one, and a 503 while it
restores from tape. Timing that reports only the compute is not the number anyone needs
for capacity planning.

TIMING CONTRACT. `timing_s` covers every stage and the parts sum to `total`:

    download   archive fetch + unzip         (0 when already on disk -- `cached` says so)
    prepare    dcm2niix, series choice, TotalSegmentator, crop, slice map
    inference  the cyst model
    measure    components, measurements, slice PNGs
    total      wall clock for the call
"""
import argparse
import json
import os
import time

from ..common import download_dicoms, paths
from . import infer_study


def run(study_iuid, env="prod", gpu=None, force=False, min_study_ml=None):
    t0 = time.time()
    dest = os.path.join(paths.DICOMS, study_iuid)
    info = download_dicoms.fetch(study_iuid, dest, env=env)
    t_dl = time.time()

    out = infer_study.analyse(study_iuid, dest, gpu=gpu, force=force,
                              min_study_ml=min_study_ml, study_iuid=study_iuid)
    f = out.setdefault("findings", {})
    t = f.get("timing_s") or {}
    t["download"] = round(t_dl - t0, 1)
    t["total"] = round(time.time() - t0, 1)
    f["timing_s"] = {k: t.get(k) for k in ("download", "prepare", "inference", "measure", "total")}
    f["source"] = dict(env=info["env"], cached=bool(info.get("cached")),
                       n_dicom_files=info.get("n_files"),
                       archive_mb=round(info.get("zip_bytes", 0) / 1e6, 1) or None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--study-iuid", required=True)
    ap.add_argument("--env", default="prod", choices=sorted(download_dicoms.ENVIRONMENTS))
    ap.add_argument("--gpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-study-ml", type=float)
    ap.add_argument("--json", help="also write the response here")
    ap.add_argument("--no-images", action="store_true",
                    help="strip slice_image, so the printed JSON is readable")
    a = ap.parse_args()
    r = run(a.study_iuid, env=a.env, gpu=a.gpu, force=a.force, min_study_ml=a.min_study_ml)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        json.dump(r, open(a.json, "w"), indent=2)
    if a.no_images:
        import copy
        r = copy.deepcopy(r)
        for side in r["findings"]["cysts"].values():
            for c in side:
                if c.get("slice_image"):
                    c["slice_image"] = f"<base64 PNG, {len(c['slice_image'])} chars>"
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
