#!/usr/bin/env bash
# Check the box can actually serve before systemd starts pretending it can.
#   APP_DIR=/opt/kidney-cysts VENV=/opt/kidney-cysts/venv ./deploy/preflight.sh
set -u
APP_DIR=${APP_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
VENV=${VENV:-$APP_DIR/venv}
PY=$VENV/bin/python
fail=0
ok(){ printf '  \033[32mok\033[0m   %s\n' "$1"; }
bad(){ printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=1; }
warn(){ printf '  warn %s\n' "$1"; }

echo "== interpreter"
[ -x "$PY" ] && ok "$($PY -V 2>&1)" || bad "no interpreter at $PY"

echo "== python packages"
for m in numpy scipy nibabel pydicom SimpleITK matplotlib; do
  $PY -c "import $m" 2>/dev/null && ok "$m" || bad "$m missing"
done
$PY -c "import torch" 2>/dev/null && ok "torch $($PY -c 'import torch;print(torch.__version__)')" || bad "torch missing"
$PY -c "import nnunetv2" 2>/dev/null && ok "nnunetv2" || bad "nnunetv2 missing"

echo "== cuda"
if $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  ok "cuda available ($($PY -c 'import torch;print(torch.cuda.get_device_name(0))'))"
else
  bad "torch cannot see a GPU"
fi

# THE IMPORTANT CHECK. When nvidia-smi fails, a VRAM gate that reports "cannot tell"
# will pass, and the only guard against over-subscribing the card disappears silently.
echo "== nvidia-smi and free VRAM"
if command -v nvidia-smi >/dev/null 2>&1; then
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
  if [ -n "${free:-}" ]; then
    ok "GPU0 free ${free} MiB"
    [ "$free" -lt 8000 ] && warn "under 8 GB free -- one study needs ~7 GB while segmenting"
  else
    bad "nvidia-smi ran but returned nothing (driver/library mismatch? usually needs a reboot)"
  fi
else
  bad "nvidia-smi not found -- the VRAM guard cannot work"
fi

echo "== model"
MD=${CYSTS_MODEL_DIR:-$APP_DIR/model/nnUNet_results}
DS=${CYSTS_DATASET:-Dataset795_kidney_cropped}
TR=${CYSTS_TRAINER:-nnUNetTrainerFinetune1e3_500ep}
CK=${CYSTS_CHECKPOINT:-checkpoint_final.pth}
F="$MD/$DS/${TR}__nnUNetPlans__3d_fullres"
[ -d "$F" ] && ok "model folder $F" || bad "no model folder at $F"
[ -f "$F/fold_0/$CK" ] && ok "checkpoint $CK ($(du -h "$F/fold_0/$CK" 2>/dev/null | cut -f1))" \
  || bad "no checkpoint at $F/fold_0/$CK"
[ -f "$F/plans.json" ] && ok "plans.json" || bad "no plans.json -- predictor cannot build the network"
# nnU-Net resolves the trainer by CLASS NAME and searches only its own package; the
# checkpoint names a custom one. A missing trainer fails at the first prediction, not at
# startup, so check it here rather than discovering it under load.
TRN="$APP_DIR/cysts/trainers/${TR}.py"
[ -f "$TRN" ] && ok "trainer class $TR" \
  || bad "no $TRN -- every prediction will fail with 'Could not find requested nnunet trainer'"

echo "== totalsegmentator"
command -v "${CYSTS_TOTALSEG:-$VENV/bin/TotalSegmentator}" >/dev/null 2>&1 \
  && ok "TotalSegmentator present" || bad "TotalSegmentator not on PATH"
TSH=${TOTALSEG_HOME_DIR:-$HOME/.totalsegmentator}
[ -d "$TSH/nnunet/results" ] && ok "TS weights at $TSH" \
  || warn "no TS weights at $TSH -- the first study will stall ~3 GB downloading them"

echo "== dcm2niix"
command -v "${CYSTS_DCM2NIIX:-dcm2niix}" >/dev/null 2>&1 && ok "dcm2niix" || bad "dcm2niix not on PATH"

echo "== resources"
ram=$(free -g | awk '/^Mem:/{print $2}')
[ "${ram:-0}" -ge 32 ] && ok "RAM ${ram} GB" || warn "RAM ${ram} GB -- a 950-slice study can reach ~25 GB"
swap=$(free -g | awk '/^Swap:/{print $2}')
[ "${swap:-0}" -eq 0 ] && warn "no swap -- an OOM kills by size, possibly not this process"
disk=$(df -BG --output=avail "$APP_DIR" | tail -1 | tr -dc '0-9')
[ "${disk:-0}" -ge 100 ] && ok "disk ${disk} GB free" || warn "disk ${disk} GB free"

echo
[ $fail -eq 0 ] && echo "preflight PASSED" || echo "preflight FAILED -- do not start the service"
exit $fail
