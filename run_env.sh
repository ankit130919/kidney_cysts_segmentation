#!/usr/bin/env bash
# Environment for running the deployment code on this box.
set -a; . /data/ankit/kidney-cysts-segmentation/api_secrets.env; set +a
export PATH=/data/ankit/TS_liver_kidney_model/.venv/bin:$PATH
export PYTHONPATH=/data/ankit/kidney-cysts-segmentation
export CYSTS_TOTALSEG=/data/ankit/TS_liver_kidney_model/.venv/bin/TotalSegmentator
export CYSTS_DCM2NIIX=/data/ankit/TS_liver_kidney_model/.venv/bin/dcm2niix
export TOTALSEG_HOME_DIR=/data/ankit/TS_liver_kidney_model/.totalsegmentator
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# Service mode: intermediates in RAM, uncompressed. See cysts/common/paths.py.
export CYSTS_SCRATCH=${CYSTS_SCRATCH:-/dev/shm/cysts}
export CYSTS_GZIP_LEVEL=${CYSTS_GZIP_LEVEL:-1}
