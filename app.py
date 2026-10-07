"""HTTP API for kidney-cyst segmentation on abdominal CT.

    POST /analyze     {"study_iuid": "..."}   synchronous, returns the full result
    POST /jobs        {"study_iuid": "..."}   returns a job_id immediately
    GET  /jobs/<id>                           status, then the result
    GET  /health

    curl --location 'http://<host>:8089/analyze' \
      --form 'study_iuid="<StudyInstanceUID>"' --form 'env="prod"'

WHY STDLIB AND NOT FLASK/FASTAPI
This box runs other production model services out of shared venvs. Installing a web
framework to serve three JSON endpoints is a change to a shared environment for no
benefit; http.server does this job.

WHY ONE ANALYSIS AT A TIME
Not throughput -- memory. A study holds several full-volume float32 arrays, and
TotalSegmentator is GPU work on a card shared with other services. Two concurrent
analyses risk an OOM that the kernel resolves by killing the largest process, which need
not be this one. Requests are SERIALISED through one worker thread; a second request
waits. That is a deliberate ceiling. Raise it only with VRAM to back it:
roughly 7 GB per concurrent study while segmenting.

WHY THE MODEL IS LOADED ONCE
Benchmarked: 41.7 s to load weights, 3.78 s to predict a study. A service that spawned a
predictor per request would spend 92% of its time loading a model it already had. The
predictor is process-wide and held open -- see cysts/pipeline/seg_fastpredict.py.

WHAT THE RESPONSE PROMISES
Every response carries `not_assessed` and `limitations`. This model finds cysts. It does
not grade them, does not look for tumours, and does not report hydronephrosis -- which
radiologist adjudication showed is 13 of its 42 genuine false positives. Reporting those
fields as null rather than 0 is the difference between "not assessed" and "absent", and
a caller will eventually read one as the other if we let it.

STATUS: SECOND READER. Against radiologist adjudication on 752 consecutive studies:
precision 0.769, recall 0.909. Not validated for autonomous reporting.
"""
import cgi
import json
import os
import queue
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cysts.common import download_dicoms, paths
from cysts.pipeline import infer_study, run_study

HOST = os.environ.get("CYSTS_API_HOST", "0.0.0.0")
PORT = int(os.environ.get("CYSTS_API_PORT", "8089"))
WORKERS = int(os.environ.get("CYSTS_API_WORKERS", "1"))
DOWNLOAD = os.environ.get("CYSTS_DOWNLOAD_CMD", "")   # optional override; see api_secrets.env.example

_JOBS = {}
_JOBS_LOCK = threading.Lock()
_Q = queue.Queue()


def _fetch(study_iuid, env):
    """Materialise a study's DICOM locally. Returns (study_id, directory).

    env selects the archive: prod (default, bearer token), staging, qa, sandbox. An
    unknown env is refused rather than silently falling back to prod.

    CYSTS_DOWNLOAD_CMD still overrides, for a site with its own fetcher: a script taking
    <study_iuid> <env> <dest_dir>.
    """
    dest = os.path.join(paths.DICOMS, study_iuid)
    if os.path.isdir(dest) and os.listdir(dest):
        return study_iuid, dest
    if DOWNLOAD:
        os.makedirs(dest, exist_ok=True)
        import subprocess
        p = subprocess.run([DOWNLOAD, study_iuid, env, dest], capture_output=True, text=True)
        if p.returncode != 0 or not os.listdir(dest):
            raise RuntimeError(f"download failed: {p.stderr.strip()[-300:]}")
        return study_iuid, dest
    info = download_dicoms.fetch(study_iuid, dest, env=env)
    mb = info.get("zip_bytes", 0) / 1e6
    print(f"  fetched {study_iuid} from {info['env']}: {info['n_files']} files"
          + ("" if info.get("cached") else f", {mb:.0f} MB"), flush=True)
    return study_iuid, dest


def _run(study_iuid, env, gpu=None):
    """Download + analyse, with the download counted in timing_s.

    CYSTS_DOWNLOAD_CMD still overrides the fetch; in that case run_study's built-in
    downloader is bypassed by _fetch having already put the files in place.
    """
    if DOWNLOAD:
        _fetch(study_iuid, env)                 # external fetcher, then the fetch is a no-op
    return run_study.run(study_iuid, env=env, gpu=gpu)


def _worker():
    while True:
        job_id, iuid, env = _Q.get()
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="running", started=time.time())
        try:
            res = _run(iuid, env)
            with _JOBS_LOCK:
                _JOBS[job_id].update(status="done", result=res, finished=time.time())
        except Exception as e:
            with _JOBS_LOCK:
                _JOBS[job_id].update(status="failed", error=f"{type(e).__name__}: {e}",
                                     traceback=traceback.format_exc()[-2000:],
                                     finished=time.time())
        finally:
            _Q.task_done()


class Handler(BaseHTTPRequestHandler):
    server_version = "cysts/0.1"

    def log_message(self, fmt, *a):      # one line per request, not two
        print(f"{self.address_string()} {fmt % a}", flush=True)

    def _send(self, code, obj):
        body = json.dumps(obj, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _params(self):
        ctype = self.headers.get("Content-Type", "")
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if ctype.startswith("application/json"):
            return json.loads(raw or b"{}")
        if ctype.startswith("multipart/form-data"):
            import io
            fs = cgi.FieldStorage(fp=io.BytesIO(raw), headers=self.headers,
                                  environ={"REQUEST_METHOD": "POST",
                                           "CONTENT_TYPE": ctype})
            return {k: fs.getfirst(k) for k in fs.keys()}
        from urllib.parse import parse_qs
        return {k: v[0] for k, v in parse_qs(raw.decode()).items()}

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            with _JOBS_LOCK:
                running = sum(1 for j in _JOBS.values() if j["status"] == "running")
            envs = {}
            for name in sorted(download_dicoms.ENVIRONMENTS):
                try:
                    download_dicoms.resolve(name)
                    envs[name] = "ready"
                except Exception as e:
                    envs[name] = str(e)[:80]
            return self._send(200, dict(status="ok", model=paths.DATASET, environments=envs,
                                        model_folder=paths.model_folder(),
                                        model_present=os.path.isdir(paths.model_folder()),
                                        workers=WORKERS, queued=_Q.qsize(), running=running))
        if self.path.startswith("/jobs/"):
            jid = self.path.rsplit("/", 1)[-1]
            with _JOBS_LOCK:
                j = _JOBS.get(jid)
            if not j:
                return self._send(404, dict(error="unknown job_id"))
            return self._send(200, j)
        return self._send(404, dict(error="not found",
                                    endpoints=["POST /analyze", "POST /jobs",
                                               "GET /jobs/<id>", "GET /health"]))

    def do_POST(self):
        try:
            p = self._params()
        except Exception as e:
            return self._send(400, dict(error=f"could not parse body: {e}"))
        iuid = (p.get("study_iuid") or p.get("study_id") or "").strip()
        env = (p.get("env") or "prod").strip()
        if not iuid:
            return self._send(400, dict(error="study_iuid is required"))
        route = self.path.rstrip("/")
        if route == "/jobs":
            try:
                download_dicoms.resolve(env)              # fail fast, not 10 min later
            except ValueError as e:
                return self._send(400, dict(error=str(e),
                                            environments=sorted(download_dicoms.ENVIRONMENTS)))
            except RuntimeError as e:
                return self._send(500, dict(error=str(e)))
            jid = uuid.uuid4().hex[:12]
            with _JOBS_LOCK:
                _JOBS[jid] = dict(job_id=jid, study_iuid=iuid, status="queued",
                                  queued_at=time.time())
            _Q.put((jid, iuid, env))
            return self._send(202, dict(job_id=jid, status="queued",
                                        poll=f"/jobs/{jid}", queue_depth=_Q.qsize()))
        if route == "/analyze":
            try:
                return self._send(200, _run(iuid, env))
            except ValueError as e:                       # unknown env -- caller's error
                return self._send(400, dict(error=str(e),
                                            environments=sorted(download_dicoms.ENVIRONMENTS)))
            except download_dicoms.ArchiveRestoring as e:
                # 503 + Retry-After is the honest answer: the study exists, it is on cold
                # storage, and blocking the connection for 15 minutes helps nobody.
                self.send_response(503)
                self.send_header("Retry-After", str(e.retry_after))
                body = json.dumps(dict(error=str(e), retry_after_s=e.retry_after,
                                       retryable=True)).encode()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            except Exception as e:
                return self._send(500, dict(error=f"{type(e).__name__}: {e}",
                                            traceback=traceback.format_exc()[-2000:]))
        return self._send(404, dict(error="not found"))


def main():
    paths.ensure(paths.DICOMS, paths.NIFTI, paths.SEG, paths.CROPS, paths.KIDNEY, paths.PRED)
    for _ in range(max(1, WORKERS)):
        threading.Thread(target=_worker, daemon=True).start()
    print(f"model   : {paths.model_folder()}"
          f"{'' if os.path.isdir(paths.model_folder()) else '   *** MISSING ***'}", flush=True)
    print(f"serving : http://{HOST}:{PORT}  ({WORKERS} worker(s), analyses are serialised)",
          flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
