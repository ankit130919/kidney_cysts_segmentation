"""Which series of a study to analyse.

PREFERENCE, in order:
    1. contrast, venous / portal phase
    2. contrast, any other phase
    3. plain / non-contrast
    4. abdominal, phase not stated

NON-ABDOMINAL AND NON-DIAGNOSTIC SERIES ARE EXCLUDED FIRST, and that exclusion is the
part that matters. The original selector compared only slice counts, with no notion of
what body part a series covered, and on a 752-study cohort it chose a chest, thorax or
spine reconstruction for 77 studies (10%) -- 11 of them `NCCT_CHEST_LUNG_THIN` -- simply
because thin chest reconstructions carry more slices than the abdominal ones. One of
those is a confirmed miss with a 20 mm cortical cyst in the report. A coronal KUB MPR
cost another.

Body part cannot be read from the DICOM header here: BodyPartExamined was blank on all
3402 series of that cohort and ContrastBolusAgent absent on every one, so classification
is by SeriesDescription / ProtocolName text.

WHAT THIS RULE DID AND DID NOT BUY. Measured head-to-head on the same 747 studies, it
changed the chosen series for 25% of them and moved overall precision from 0.564 to
0.555 and recall from 0.872 to 0.863 -- i.e. nothing, within noise. It is kept because
it is anatomically correct, not because it improved a metric: only 7% of that cohort
contained a venous-phase series at all, so the preference rarely had anything to act on.
"""
import re

EXCLUDE = re.compile(
    r"scout|topogram|localis|localiz|dose.?report|dosereport|patient.?protocol"
    r"|mpr|reformat|coronal|sagittal|\bcor\b|\bsag\b"
    r"|lung|chest|thorax|hrct|spine|cervic|lumbar|dorsal|bone|osseo", re.I)
ABDOMEN = re.compile(r"abdomen|abd\b|kub|pelvis|liver|renal|kidney|urogram|urograph"
                     r"|triphasic|whole.?body|hepat", re.I)
VENOUS = re.compile(r"venous|portal|pvp", re.I)
CONTRAST = re.compile(r"contrast|cect|\bce\b|\bc\+|post.?contrast|arterial|\bart\b"
                      r"|delay|nephro|excret|angio", re.I)
PLAIN = re.compile(r"plain|pre.?contrast|non.?con|ncct|unenhanced|without|\bnc\b", re.I)

TIER_NAME = {0: "contrast-venous", 1: "contrast-other", 2: "plain",
             3: "abdominal-unlabelled", 9: "excluded"}


def _text(c):
    return f"{c.get('desc', '')} {c.get('prot', '')} {c.get('name', '')}"


def tier(c):
    t = _text(c)
    if EXCLUDE.search(t):
        return 9
    # a plain marker wins over a stray 'contrast' in a protocol name such as
    # "Abdomen Pelvis Pre Contrast" -- that series IS the plain one
    if PLAIN.search(t):
        return 2
    if VENOUS.search(t):
        return 0
    if CONTRAST.search(t):
        return 1
    return 3 if ABDOMEN.search(t) else 9


def pick(cands):
    """Best candidate, or (None, None) if the study has nothing usable.

    Within the winning tier: coverage must reach 85% of the best available (do not take
    a short series just because it is thin), then the most slices wins.
    """
    if not cands:
        return None, None
    scored = [(tier(c), c) for c in cands]
    usable = [(t, c) for t, c in scored if t < 9]
    if not usable:                      # everything excluded -- fall back rather than fail
        usable = [(3, c) for c in cands]
    best_tier = min(t for t, _ in usable)
    pool = [c for t, c in usable if t == best_tier]
    maxcov = max(c["cov"] for c in pool)
    pool = [c for c in pool if c["cov"] >= 0.85 * maxcov]
    return max(pool, key=lambda c: c["nz"]), best_tier
