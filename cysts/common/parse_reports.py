"""Report text -> a reference label, for evaluation only. Never an input to the model.

NEGATION SCOPING, NOT KEYWORD MATCHING. "No renal cyst", "kidneys are unremarkable" and
"free of focal lesion" all contain the keyword and all mean the opposite. The rule: a
negation governs a finding only when it appears BEFORE it, and only until a contrastive
conjunction closes its scope.

ORGAN SPECIFICITY MATTERS HERE. Hepatic and splenic cysts are common in these reports,
and the crop fed to the model deliberately contains liver and spleen -- so a liver cyst
counted as positive manufactures a false negative every time the model correctly ignores
it.

MEASURED AGAINST AN LLM. A local Qwen2.5-14B scoring yes/no on the same 756 reports
agreed with these rules on 98.9% of studies. The 8 disagreements were all rules errors,
and all the same kind: a cyst named in the clinical history or a quoted prior ultrasound
rather than as a finding on this scan. If an LLM is available, prefer it; this is the
fallback.
"""
import re

NEG = re.compile(r"\b(no|not|without|absent|nil|free of|negative for|unremarkable)\b", re.I)
STOP = re.compile(r"\b(but|however|although|though|whereas|while)\b", re.I)
SENT = re.compile(r"(?<=[.;\n])\s+")

RENAL_CYST = re.compile(
    r"(renal|kidney|cortical|parapelvic|peripelvic)[^.;]{0,40}\bcysts?\b"
    r"|\bcysts?\b[^.;]{0,40}(renal|kidney|in (the )?(right|left) kidney)"
    r"|bosniak", re.I)
OTHER_ORGAN = re.compile(r"\b(hepatic|liver|splenic|spleen|pancrea\w*|ovarian|adnexal|"
                         r"mesenteric|renal artery)\b", re.I)
HISTORY = re.compile(r"\b(h/o|history of|k/c/o|usg impression|prior|previous)\b", re.I)


def negated(sentence, start):
    for m in NEG.finditer(sentence):
        if m.start() >= start:
            continue
        stop = STOP.search(sentence, m.end(), start)
        if not stop:
            return True
    return False


def has_renal_cyst(text):
    """(bool, evidence sentence)."""
    for s in SENT.split(str(text)):
        if HISTORY.search(s):
            continue                      # a cyst in the history is not a finding here
        for m in RENAL_CYST.finditer(s):
            if negated(s, m.start()):
                continue
            clause = s[max(0, m.start() - 60):m.end() + 60]
            if OTHER_ORGAN.search(clause) and not re.search(r"\b(renal|kidney)\b", clause, re.I):
                continue
            return True, " ".join(s.split())[:220]
    return False, ""
