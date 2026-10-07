"""What a caller is promised. Breaking any of this breaks a downstream reader."""
import inspect

from cysts.pipeline import infer_study


def test_not_assessed_fields_are_null_never_zero():
    src = inspect.getsource(infer_study)
    # match the DICT, not the docstring that also mentions it
    assert "not_assessed=dict(" in src
    i = src.index("not_assessed=dict(")
    block = src[i:src.index(")", i) + 1]
    for field in ("hydronephrosis", "renal_tumour", "bosniak_grade", "ureter", "bladder"):
        assert field in block, f"{field} must be declared not-assessed"
    assert "=0" not in block.replace(" ", "")


def test_limitations_mention_the_measured_failure_modes():
    src = inspect.getsource(infer_study)
    low = src.lower()
    for term in ("hydronephrosis", "hyperdense", "plain"):
        assert term in low, f"limitations must name the {term} failure mode"
