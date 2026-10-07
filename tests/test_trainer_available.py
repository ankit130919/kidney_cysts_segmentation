"""The trainer class the checkpoint names must ship with the repo.

The first load test failed all 10 studies on this, after downloading 1.8 GB and running
TotalSegmentator on each -- the failure is at predictor construction, so everything
expensive happens first. A unit test costs nothing and catches it at commit time.
"""
import os

from cysts.common import paths


def test_ext_trainer_env_is_set_on_import():
    assert os.environ.get("nnUNet_extTrainer"), \
        "paths must point nnUNet_extTrainer at cysts/trainers on import"
    assert os.path.isdir(os.environ["nnUNet_extTrainer"])


def test_the_checkpoints_trainer_class_is_present():
    f = os.path.join(os.environ["nnUNet_extTrainer"], f"{paths.TRAINER}.py")
    assert os.path.exists(f), (
        f"checkpoint names {paths.TRAINER} but {f} does not exist; nnU-Net resolves "
        "trainers by class name and searches only its own package")
    src = open(f).read()
    assert f"class {paths.TRAINER}" in src


def test_ts_env_swap_restores_our_model_paths():
    """A TotalSegmentator call must not leave nnUNet_results pointing at TS weights --
    the cyst predictor would then be unfindable for the rest of the process."""
    import os
    from cysts.common import organ_seg
    before = {k: os.environ.get(k) for k in
              ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")}
    try:
        with organ_seg._ts_env():
            pass
    except Exception:
        return                      # TotalSegmentator not installed in this env
    after = {k: os.environ.get(k) for k in before}
    assert after == before, f"env not restored: {before} -> {after}"
