"""The JSON contract. Breaking any of this breaks a downstream reader."""
import json

from cysts.report.response import build


def _c(h, w, hu, side, slice_number=101, slice_image="iVBORw0KGgo="):
    return dict(size_mm=f"{h} x {w}", height_mm=h, width_mm=w, density_hu=hu, side=side,
                slice_number=slice_number, slice_image=slice_image)


def test_exact_shape_and_keys():
    r = build("1.2.3", [_c(11.7, 9.4, 20, "left")], dict(slice_thickness_mm=1.25), True)
    assert set(r) >= {"study_iuid", "study_prediction", "findings"}
    f = r["findings"]
    for k in ("cysts_prediction", "total_cysts", "cysts", "largest_cysts_mm",
              "max_density_hu", "slice_thickness_mm"):
        assert k in f, k
    assert set(f["cysts"]) >= {"right_kidney", "left_kidney"}
    c = f["cysts"]["left_kidney"][0]
    assert set(c) == {"cysts_id", "size_mm", "density_hu", "slice_number", "slice_image"}
    assert json.loads(json.dumps(r))        # must serialise


def test_study_prediction_is_always_abnormal_even_with_no_cysts():
    r = build("1.2.3", [], dict(slice_thickness_mm=1.0), False)
    assert r["study_prediction"] == "Abnormal"
    assert r["findings"]["cysts_prediction"] is False
    assert r["findings"]["total_cysts"] == 0
    assert r["findings"]["cysts"] == {"right_kidney": [], "left_kidney": []}


def test_ids_are_unique_across_both_kidneys():
    r = build("1.2.3", [_c(9, 8, 20, "left"), _c(7, 6, 30, "right"), _c(5, 4, 40, "left")],
              dict(slice_thickness_mm=1.0), True)
    ids = [c["cysts_id"] for side in ("left_kidney", "right_kidney")
           for c in r["findings"]["cysts"][side]]
    assert sorted(ids) == [1, 2, 3], "ids must be unique study-wide, not per kidney"


def test_largest_is_an_inplane_extent_not_a_3d_diameter():
    # 11.7 x 9.4 -> 11.7, never sqrt(11.7^2+9.4^2)
    r = build("1.2.3", [_c(11.7, 9.4, 20, "left")], dict(slice_thickness_mm=1.0), True)
    assert r["findings"]["largest_cysts_mm"] == 11.7


def test_unassignable_lesion_is_not_laundered_into_a_kidney():
    r = build("1.2.3", [_c(9, 8, 20, None)], dict(slice_thickness_mm=1.0), True)
    assert r["findings"]["cysts"]["left_kidney"] == []
    assert r["findings"]["cysts"]["right_kidney"] == []
    assert r["findings"]["cysts"]["unassigned"][0]["cysts_id"] == 1


def test_max_density_is_the_max_not_the_mean():
    r = build("1.2.3", [_c(9, 8, 20, "left"), _c(7, 6, 577, "right")],
              dict(slice_thickness_mm=1.0), True)
    assert r["findings"]["max_density_hu"] == 577


def test_slice_number_is_null_rather_than_an_array_index_when_unmapped():
    # a wrong-but-plausible slice number sends a radiologist to the wrong image; null
    # tells them to look it up themselves
    r = build("1.2.3", [_c(9, 8, 20, "left", slice_number=None)],
              dict(slice_thickness_mm=1.0), True)
    assert r["findings"]["cysts"]["left_kidney"][0]["slice_number"] is None


def test_slice_image_is_base64_and_decodes_to_a_png():
    import base64
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 32).decode()
    r = build("1.2.3", [_c(9, 8, 20, "left", slice_image=png)],
              dict(slice_thickness_mm=1.0), True)
    raw = base64.b64decode(r["findings"]["cysts"]["left_kidney"][0]["slice_image"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
