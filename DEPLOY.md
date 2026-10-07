# Deploying the cyst API on a new machine

Clone, install, preflight, start. The numbers below were measured on the development
box, not estimated.

## 1. Hardware

The binding constraint is **VRAM**, not RAM or CPU: ~7 GB per study while
TotalSegmentator runs. The cyst model itself needs ~4 GB and is held resident.

| target | GPU | RAM | CPU | disk |
|---|---|---|---|---|
| ~150 studies/hour | 1 x A100-40GB **dedicated** | 128 GB | 16 | 1 TB |
| ~250 studies/hour | 1 x H100-80GB | 192 GB | 24 | 1 TB |
| minimum viable | 1 x 24 GB (A10 / L4 / 3090) | 64 GB | 8 | 500 GB |

Measured per study, 752-study production run:

    end-to-end  ~23 s      (19 s TotalSegmentator, 3.8 s cyst model)
    VRAM          7 GB     peak, while segmenting
    RAM          14 GB     on a 536-slice study, ~26 MB/slice
    archive     ~370 MB    average DICOM

**Do not put other GPU model services on this box if you can avoid it.** During
development a neighbouring job held 50 GB of an 80 GB card and the crop stage dropped
28 studies to CUDA OOM before the GPU assignment was changed.

## 2. Install

    git clone <this repo> /opt/kidney-cysts
    cd /opt/kidney-cysts

    python3.11 -m venv venv
    ./venv/bin/pip install --upgrade pip
    # torch is a CUDA build -- take it from the right index, not PyPI default
    ./venv/bin/pip install -r deploy/requirements-lock.txt \
        --extra-index-url https://download.pytorch.org/whl/cu126
    ./venv/bin/pip install -e ".[seg]"

The trained weights are **not in git** (~250 MB per checkpoint). Copy the tree in:

    mkdir -p model/nnUNet_results
    rsync -a <source>:/.../nnUNet_results/Dataset795_kidney_cropped model/nnUNet_results/

It must contain `plans.json`, `dataset.json` and `fold_0/checkpoint_final.pth`.
Without `plans.json` the predictor cannot rebuild the network and fails at startup.

Pre-download the TotalSegmentator weights (~3 GB) so the first study does not stall:

    TOTALSEG_HOME_DIR=/opt/kidney-cysts/.totalsegmentator ./venv/bin/TotalSegmentator --help >/dev/null
    # or copy an existing ~/.totalsegmentator

Credentials and configuration:

    cp deploy/api_secrets.env.example api_secrets.env
    chmod 600 api_secrets.env

This repo does not embed archive credentials. Point `CYSTS_DOWNLOAD_CMD` at a script
taking `<study_iuid> <env> <dest_dir>`, or stage studies yourself under
`data/dicoms/<study_iuid>/`.

## 3. Preflight — do not skip

    APP_DIR=/opt/kidney-cysts VENV=/opt/kidney-cysts/venv ./deploy/preflight.sh

It checks the interpreter, packages, CUDA, free VRAM, the model folder and checkpoint,
TotalSegmentator and its weights, dcm2niix, RAM, swap and disk.

**The nvidia-smi check is the one that matters.** When `nvidia-smi` fails, a VRAM gate
that reports "cannot tell" passes, and the only guard against over-subscribing the card
disappears silently. A failure here is usually a driver/library mismatch and needs a
reboot.

The preflight also warns when there is **no swap**. That is not cosmetic: with no swap
the OOM killer chooses by size, and a 25 GB analysis is rarely the largest process on a
shared box — so the kernel takes someone else's service instead of this one.

## 4. Service

    mkdir -p logs
    sed -e "s|@APP_DIR@|/opt/kidney-cysts|g" -e "s|@VENV@|/opt/kidney-cysts/venv|g" \
        deploy/kidney-cysts-api.service > /etc/systemd/system/kidney-cysts-api.service
    systemctl daemon-reload
    systemctl enable --now kidney-cysts-api
    systemctl status kidney-cysts-api

### Workers

Analyses are **serialised** regardless of this setting — see the module docstring in
`app.py`. `CYSTS_API_WORKERS` sets how many requests may be in flight, not how many run
at once. Raising it does not raise throughput; it only lets more callers queue without
blocking on the socket.

To run genuinely concurrent studies you need VRAM for them: `floor(free_VRAM_GB / 7)`.
Run a second process on a second GPU rather than lifting the lock in one process.

## 5. nginx

    cat deploy/nginx-cysts.conf   # paste into your server{} block
    nginx -t && systemctl reload nginx

`proxy_read_timeout` must stay well above one analysis plus its queue wait. The 60 s
default returns 504 for work proceeding normally.

## 6. Verify

    curl -s localhost:8089/health | python3 -m json.tool

`model_present` must be `true`. Then a real study:

    curl --location 'http://localhost:8089/analyze' --max-time 600 \
      --form 'study_iuid="<StudyInstanceUID>"' --form 'env="prod"'

Expect `study_prediction`, `findings.cysts[]` with `volume_ml`, `max_diameter_mm`,
`mean_hu` and `kidney_containment`, plus `not_assessed` and `limitations`.

For anything bulk use the async route — it returns in ~40 ms, so no client timeout
applies:

    curl -s -X POST localhost:8089/jobs -H 'Content-Type: application/json' \
      -d '{"study_iuid":"..."}'      # -> {"job_id": "..."}
    curl -s localhost:8089/jobs/<job_id>

## Capacity

One study ~23 s, serialised: **~150 studies/hour** on a dedicated card. A synchronous
caller must keep at most one study in flight or raise its timeout above the queue wait.

## Known rough edges

- **Nothing authenticates inbound requests.** Anyone who can reach the port can submit a
  study id and read findings. Put it behind your own auth.
- The service trusts `CYSTS_DOWNLOAD_CMD` to validate its own input; a study id goes
  straight to it as an argument.
- Overlay rendering is not exposed by the API. It is batch tooling
  (`cysts-overlay`) and takes ~20 s/study on CPU.
