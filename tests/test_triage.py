"""Series selection. Each case here is a study that actually went wrong."""
from cysts.common.triage_series import pick, tier


def c(name, nz=400, cov=400.0):
    return dict(name=name, desc=name, prot="", nz=nz, cov=cov)


def test_chest_series_never_wins_even_with_more_slices():
    # study 9476631: the old selector chose NCCT_CHEST_LUNG_THIN because it had the most
    # slices, and a 20 mm cortical cyst in the report was missed.
    best, t = pick([c("NCCT_CHEST_LUNG_THIN", nz=900, cov=900),
                    c("Abdomen_Pelvis_Thin_Plain", nz=400, cov=400)])
    assert "Abdomen" in best["name"]


def test_mpr_reformat_is_excluded():
    # study 9481037: a coronal KUB MPR was segmented instead of the axial series.
    best, t = pick([c("KUB_PLAIN_STUDY_B_MPR", nz=900, cov=900),
                    c("Routine_Abdomen_Thin_Plain", nz=400, cov=400)])
    assert "MPR" not in best["name"]


def test_venous_beats_plain():
    best, t = pick([c("Abdomen_Plain_Thin", nz=800, cov=800),
                    c("Abdomen_Venous_Phase_Thin", nz=400, cov=400)])
    assert "Venous" in best["name"] and t == 0


def test_pre_contrast_is_plain_not_contrast():
    # "Pre Contrast" contains the word contrast; it is the plain series.
    assert tier(c("Routine_Abdomen_Pelvis_Pre_Contrast_THIN")) == 2


def test_scouts_and_dose_reports_excluded():
    assert tier(c("Scout")) == 9
    assert tier(c("10001_DoseReport_Dose_Report")) == 9


def test_everything_excluded_still_returns_something():
    # a study of nothing but chest series must not crash the pipeline
    best, t = pick([c("NCCT_CHEST_LUNG_THIN")])
    assert best is not None
