"""Fetch a study's DICOM from the 5C archives.

ENVIRONMENTS

    env       base URL                                              auth
    prod      https://dcm.5cnetwork.com/download/                   Bearer token
    staging   https://e2e-staging-api.5cnetwork.com/dicom/download/ none
    qa        https://e2e-qa-api.5cnetwork.com/dicom/download/      none
    sandbox   https://e2e-sandbox-api.5cnetwork.com/dicom/download/ none

prod is the default and the only one that needs a token; it comes from
NCCTF_DICOM_TOKEN (or CYSTS_DICOM_TOKEN) in api_secrets.env and is never read from a
request. An unknown env is refused rather than silently falling back to prod -- a caller
who typos "stagin" must not have their study pulled from production.

WHAT THIS DELIBERATELY DOES NOT DO

  No automatic retry on the HTTP layer. A cold study answers 503 `restoring_from_archive`
  with `Retry-After: 900`, and curl's --retry honours that header by SLEEPING FIFTEEN
  MINUTES inside the process. Thirty workers once sat blocked that way. A 503 is returned
  to the caller as a retryable status with the wait time, and the caller decides.

  No long per-connection timeout by default. Measured throughput is ~150 KB/s per
  connection, so a 450 MB study genuinely needs ~50 minutes; DOWNLOAD_TIMEOUT is set
  accordingly but is a cap, not a target.

EXTRACTION. The archive wraps files as DICOM/<study-hash>/<series>/<instance>; the wrapper
is stripped and the series level kept, because everything downstream groups by series.
Entries are checked to stay inside the destination -- a zip can name ../ and a naive
extract writes outside it.
"""
import os
import re
import shutil
import subprocess
import zipfile

ENVIRONMENTS = {
    "prod": dict(url="https://dcm.5cnetwork.com/download/", auth=True,
                 host="dcm.5cnetwork.com"),
    "staging": dict(url="https://e2e-staging-api.5cnetwork.com/dicom/download/", auth=False,
                    host="e2e-staging-api"),
    "qa": dict(url="https://e2e-qa-api.5cnetwork.com/dicom/download/", auth=False,
               host="e2e-qa-api"),
    "sandbox": dict(url="https://e2e-sandbox-api.5cnetwork.com/dicom/download/", auth=False,
                    host="e2e-sandbox-api"),
}
DEFAULT_ENV = "prod"
DOWNLOAD_TIMEOUT = int(os.environ.get("CYSTS_DOWNLOAD_TIMEOUT", "5400"))
USER_AGENT = os.environ.get("CYSTS_USER_AGENT", "Mozilla/5.0")


class ArchiveRestoring(Exception):
    """The study is on cold storage. Retryable, after `retry_after` seconds."""

    def __init__(self, retry_after=900):
        super().__init__(f"study is being restored from archive; retry in {retry_after}s")
        self.retry_after = retry_after


def token():
    for k in ("NCCTF_DICOM_TOKEN", "CYSTS_DICOM_TOKEN"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return ""


def resolve(env):
    e = (env or DEFAULT_ENV).strip().lower()
    if e not in ENVIRONMENTS:
        raise ValueError(f"unknown env {env!r}; expected one of {sorted(ENVIRONMENTS)}")
    cfg = dict(ENVIRONMENTS[e], env=e)
    if cfg["auth"] and not token():
        raise RuntimeError(
            f"env '{e}' needs a bearer token but NCCTF_DICOM_TOKEN is unset. "
            "Put it in api_secrets.env (chmod 600); it is never taken from a request.")
    return cfg


def _curl(url, dest_zip, bearer=None):
    cmd = ["curl", "-sS", "-fSL", "--http1.1",
           "--connect-timeout", "30", "--max-time", str(DOWNLOAD_TIMEOUT),
           "-A", USER_AGENT,
           "-w", "%{http_code} %{size_download} %{time_total}",
           "-o", dest_zip, url]
    if bearer:
        cmd[1:1] = ["-H", f"Authorization: Bearer {bearer}"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        codes = re.findall(r"error:\s*(\d{3})", p.stderr)
        code = codes[-1] if codes else None
        if code == "503":
            raise ArchiveRestoring()
        if p.returncode == 28:
            raise RuntimeError(f"download timed out after {DOWNLOAD_TIMEOUT}s "
                               "(~150 KB/s per connection; a 450 MB study needs ~50 min)")
        raise RuntimeError(f"download failed: "
                           f"{(' '.join(p.stderr.split())[:200]) or f'curl exit {p.returncode}'}")
    code, size, _ = (p.stdout.strip().split() + ["", "", ""])[:3]
    if code == "503":
        raise ArchiveRestoring()
    if code != "200":
        raise RuntimeError(f"archive returned HTTP {code}")
    return int(size or 0)


def _extract(zpath, dest):
    if not zipfile.is_zipfile(zpath):
        with open(zpath, "rb") as fh:
            head = fh.read(80)
        raise RuntimeError(f"response is not a zip: {head!r}")
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"corrupt entry in archive: {bad}")
        files = [i for i in z.infolist() if not i.is_dir()]
        if not files:
            raise RuntimeError("archive is empty")
        dirs = [os.path.dirname(i.filename) for i in files]
        prefix = os.path.commonpath(dirs) if all(dirs) else ""
        if prefix and all(d == prefix for d in dirs):
            prefix = os.path.dirname(prefix)       # keep the series level
        for info in files:
            rel = os.path.relpath(info.filename, prefix) if prefix else info.filename
            out = os.path.normpath(os.path.join(dest, rel))
            if not out.startswith(os.path.abspath(dest) + os.sep) and \
               not out.startswith(dest + os.sep):
                raise RuntimeError(f"archive entry escapes the destination: {info.filename}")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(info) as src, open(out, "wb") as fh:
                shutil.copyfileobj(src, fh)
    return sum(len(fs) for _, _, fs in os.walk(dest))


def fetch(study_iuid, dest_dir, env=DEFAULT_ENV, zip_dir=None):
    """Download and extract one study. Returns a dict; raises on failure."""
    cfg = resolve(env)
    if os.path.isdir(dest_dir) and os.listdir(dest_dir):
        return dict(study_iuid=study_iuid, env=cfg["env"], path=dest_dir, cached=True,
                    n_files=sum(len(fs) for _, _, fs in os.walk(dest_dir)))
    zip_dir = zip_dir or os.path.join(os.path.dirname(dest_dir.rstrip("/")), ".zips")
    os.makedirs(zip_dir, exist_ok=True)
    zpath = os.path.join(zip_dir, f"{study_iuid}.zip")
    try:
        size = _curl(cfg["url"] + study_iuid, zpath,
                     bearer=token() if cfg["auth"] else None)
        n = _extract(zpath, dest_dir)
    finally:
        if os.path.exists(zpath):
            os.remove(zpath)
    return dict(study_iuid=study_iuid, env=cfg["env"], path=dest_dir, cached=False,
                n_files=n, zip_bytes=size)
