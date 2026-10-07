"""One predictor, held open, used for every study.

Separate from nnunet_resident because they solve different halves of the same problem:
that module stops the WEIGHTS being reloaded; this one stops the PREDICTOR being rebuilt
and gives the service a single object to call.

SERIALISED BY DESIGN. One study holds several full-volume float32 arrays and
segmentation is GPU work on a card this box shares. Two concurrent predictions is how
you get an OOM that the kernel resolves by killing whichever process is largest -- which
need not be this one. The lock is a deliberate ceiling.
"""
import os
import threading

from ..common import paths

_LOCK = threading.Lock()
_PRED = None


def get(device="cuda", tta=False):
    """The process-wide predictor, built on first use."""
    global _PRED
    with _LOCK:
        if _PRED is not None:
            return _PRED
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from . import nnunet_resident
        nnunet_resident.install()
        folder = paths.model_folder()
        if not os.path.isdir(folder):
            raise RuntimeError(
                f"model not found at {folder}. Set CYSTS_MODEL_DIR, or copy the trained "
                "nnU-Net results tree in -- the checkpoint is ~250 MB and is not in git.")
        p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=bool(tta),
                            device=torch.device(device), verbose=False,
                            verbose_preprocessing=False, allow_tqdm=False)
        p.initialize_from_trained_model_folder(folder, (paths.FOLD,), paths.CHECKPOINT)
        _PRED = p
        return _PRED


def predict(in_dir, out_dir, npp=2, nps=2):
    """Predict every *_0000.nii.gz in in_dir. Serialised: see the module docstring."""
    os.makedirs(out_dir, exist_ok=True)
    p = get()
    with _LOCK:
        p.predict_from_files(in_dir, out_dir, save_probabilities=False,
                             overwrite=False, num_processes_preprocessing=npp,
                             num_processes_segmentation_export=nps)
    return out_dir
