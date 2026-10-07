"""The filters must behave exactly as the measurements that justified them."""
from cysts.kidney import postprocess as pp


def comp(ml, dist=0.0):
    return dict(volume_ml=ml, distance_to_kidney_mm=dist, max_diameter_mm=10.0, mean_hu=12.0)


def test_far_from_kidney_is_rejected_not_deleted():
    rep, rej, pos = pp.apply([comp(5.0, dist=130.0)])
    assert rep == [] and len(rej) == 1
    assert "rejected_because" in rej[0], "a reviewer needs to see WHY it went"


def test_exophytic_cyst_is_kept():
    # a cyst bulging a few mm beyond the renal contour is normal anatomy, not an error
    rep, rej, pos = pp.apply([comp(5.0, dist=3.0)])
    assert len(rep) == 1 and rej == [] and pos


def test_study_threshold_is_a_total_not_a_per_component_floor():
    # three 0.2 ml cysts = 0.6 ml -> positive. A per-component floor would drop all three,
    # which is wrong: a patient with several small cysts is one a radiologist reports.
    rep, rej, pos = pp.apply([comp(0.2), comp(0.2), comp(0.2)], min_study_ml=0.5)
    assert pos and len(rep) == 3


def test_single_speck_is_negative_at_the_default_threshold():
    rep, rej, pos = pp.apply([comp(0.2)], min_study_ml=0.5)
    assert not pos


def test_no_components_is_negative():
    rep, rej, pos = pp.apply([])
    assert rep == [] and not pos
