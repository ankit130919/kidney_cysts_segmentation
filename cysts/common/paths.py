"""Where things live.

Large intermediates (nifti/, seg/, crops/) sit at the project root and are SHARED
between runs: TotalSegmentator is ~19 s/study and its output does not depend on which
analysis you are doing, so rebuilding it per run is pure waste.

Results (csv/, overlays/) go under RUN, which defaults to the project root but can be
pointed at a per-run folder so a new run never overwrites an old one's numbers:

    CYSTS_RUN=run_2026_10 cysts-study --study-iuid ...
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

_run = os.environ.get("CYSTS_RUN", "").strip()
RUN = ROOT if not _run else (_run if os.path.isabs(_run) else os.path.join(ROOT, _run))


def _dir(env, *default):
    v = os.environ.get(env, "").strip()
    if not v:
        return os.path.join(ROOT, *default)
    return v if os.path.isabs(v) else os.path.join(ROOT, v)


# shared, expensive to rebuild
DICOMS = _dir("CYSTS_DICOMS", "data", "dicoms")
NIFTI = _dir("CYSTS_NIFTI", "data", "nifti")
SEG = _dir("CYSTS_SEG", "data", "seg")
CROPS = _dir("CYSTS_CROPS", "data", "crops")
KIDNEY = _dir("CYSTS_KIDNEY", "data", "kidney")
PRED = _dir("CYSTS_PRED", "data", "pred")

# per-run results
CSV = os.path.join(RUN, "csv")
OVERLAYS = os.path.join(RUN, "overlays")

# the trained model. Ships outside git -- 250 MB per checkpoint.
MODEL_DIR = _dir("CYSTS_MODEL_DIR", "model", "nnUNet_results")
DATASET = os.environ.get("CYSTS_DATASET", "Dataset795_kidney_cropped")
TRAINER = os.environ.get("CYSTS_TRAINER", "nnUNetTrainerFinetune1e3_500ep")
CONFIG = os.environ.get("CYSTS_CONFIG", "3d_fullres")
CHECKPOINT = os.environ.get("CYSTS_CHECKPOINT", "checkpoint_final.pth")
FOLD = os.environ.get("CYSTS_FOLD", "0")


def model_folder():
    return os.path.join(MODEL_DIR, DATASET, f"{TRAINER}__nnUNetPlans__{CONFIG}")


def ensure(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# nnU-Net looks up a trainer by class name, in its own package only. The checkpoint names
# nnUNetTrainerFinetune1e3_500ep, which lives in this repo, so point nnU-Net at it here --
# at import, before any predictor is built. Without this every prediction fails with
# "Could not find requested nnunet trainer", which is how the first load test ended.
TRAINERS = os.path.join(HERE, os.pardir, "trainers")
os.environ.setdefault("nnUNet_extTrainer", os.path.abspath(TRAINERS))

# nnU-Net also insists these exist even for pure inference.
os.environ.setdefault("nnUNet_raw", os.path.join(ROOT, "data", "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", os.path.join(ROOT, "data", "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", MODEL_DIR)


# ---------------------------------------------------------------- intermediates
# The crop and the prediction exist for seconds: the crop is written, read straight back
# by nnU-Net, and the prediction is written and read back by the measurement stage. On a
# ~130 MB volume that cost 24.6 s of gzip alone, for data nothing outside the request
# will ever read again.
#
# CYSTS_SCRATCH points them at a tmpfs (RAM), and CYSTS_NIFTI_EXT drops the compression.
# Both default OFF, because a batch or evaluation run WANTS them on disk -- re-running the
# 752-study cohort reuses crops and skips ~55 s of GPU per study. This is a service-mode
# setting, not a global one.
SCRATCH = os.environ.get("CYSTS_SCRATCH", "").strip()

# The extension is NOT configurable. nnU-Net selects its input files by the `file_ending`
# recorded in the model's dataset.json (".nii.gz"), so an uncompressed ".nii" crop is
# invisible to predict_from_files -- it reports "0 cases" and returns nothing. Compression
# is cheapened instead, by level, not by removing it.
NIFTI_EXT = ".nii.gz"

# nibabel ALREADY defaults to compression level 1 -- checked, not assumed -- so the 24.6 s
# measured for "crop + write" is data volume and disk I/O, not the compression level. The
# lever that remains is WHERE it is written, which is what CYSTS_SCRATCH does. This knob is
# kept only so a slow-disk box can trade size for time; changing it is unlikely to help.
COMPRESSLEVEL = int(os.environ.get("CYSTS_GZIP_LEVEL", "1"))
try:
    import nibabel.openers as _op
    _op.Opener.default_compresslevel = COMPRESSLEVEL
except Exception:
    pass


def _side(kind, default_dir):
    return os.path.join(SCRATCH, kind) if SCRATCH else default_dir


def crop_path(study_id):
    return os.path.join(_side("crops", CROPS), f"{study_id}_0000{NIFTI_EXT}")


def kidney_path(study_id):
    return os.path.join(_side("kidney", KIDNEY), f"{study_id}{NIFTI_EXT}")


def pred_dir():
    return _side("pred", PRED)


def pred_path(study_id):
    # nnU-Net names its output from the input file, minus _0000, keeping the extension
    return os.path.join(pred_dir(), f"{study_id}{NIFTI_EXT}")


def stage_dir(study_id):
    return os.path.join(_side("crops", CROPS), f".one_{study_id}")
