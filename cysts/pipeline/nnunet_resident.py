"""Keep nnU-Net weights on the GPU between studies.

THE NUMBERS THAT JUSTIFY THIS MODULE. Benchmarked on this model by timing one study and
then twenty-one, so the fixed cost separates from the marginal one:

    1 study    45.5 s
    21 studies 121.0 s
    ------------------------------------
    startup     41.7 s   (interpreter, 250 MB checkpoint, cuDNN warm-up)
    per study    3.78 s

A per-study service that spawns `nnUNetv2_predict` per request therefore spends 92% of
its time loading a model it already loaded. 100 studies one at a time is ~75 minutes;
the same 100 through one resident process is ~6.5 minutes.

HOW. nnUNetPredictor.initialize_from_trained_model_folder() torch.load()s the checkpoint,
rebuilds the network and pushes it to the card on every call. This wraps it so the
expensive result is cached by (model folder, folds, checkpoint) and restored afterwards.

Only the attributes the original assigns are cached, read off the nnunetv2 source rather
than guessed -- so a version mismatch surfaces as a missing attribute at import time
rather than as a subtly wrong prediction.
"""
import threading

_LOCK = threading.Lock()
_CACHE = {}

# assigned by nnUNetPredictor.initialize_from_trained_model_folder in nnunetv2 2.x
_ATTRS = ("plans_manager", "configuration_manager", "list_of_parameters", "network",
          "dataset_json", "trainer_name", "allowed_mirroring_axes", "label_manager")


def install():
    """Patch nnUNetPredictor so repeated loads of the same model are free."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    if getattr(nnUNetPredictor, "_cysts_resident", False):
        return
    original = nnUNetPredictor.initialize_from_trained_model_folder

    def cached(self, model_training_output_dir, use_folds, checkpoint_name="checkpoint_final.pth"):
        key = (str(model_training_output_dir), tuple(sorted(map(str, use_folds))), checkpoint_name)
        with _LOCK:
            hit = _CACHE.get(key)
            if hit is None:
                original(self, model_training_output_dir, use_folds, checkpoint_name)
                missing = [a for a in _ATTRS if not hasattr(self, a)]
                if missing:
                    raise RuntimeError(
                        "nnunet_resident is out of date with this nnunetv2: "
                        f"predictor has no {missing}. Re-read "
                        "nnUNetPredictor.initialize_from_trained_model_folder and update _ATTRS.")
                _CACHE[key] = {a: getattr(self, a) for a in _ATTRS}
                return
            for a, v in hit.items():
                setattr(self, a, v)

    nnUNetPredictor.initialize_from_trained_model_folder = cached
    nnUNetPredictor._cysts_resident = True


def cached_models():
    return list(_CACHE)
