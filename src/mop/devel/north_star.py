
from __future__ import annotations

import re

DEVELOPMENTAL_LOOP = (
    "perceive -> remember -> predict -> notice surprise -> choose what to study -> adapt -> "
    "consolidate -> abstract -> transfer -> explain what changed -> choose the next lesson"
)

ENGINEERING_TERMS = {
    "drive": "an engineered objective term (novelty, uncertainty, learning-progress), not a wish",
    "memory": "a data structure / behavioral retention, not subjective memory",
    "self-monitoring": "diagnostics (uncertainty, calibration, failure detection), not self-awareness",
    "curiosity": "a data-selection objective (learning progress), not a feeling",
    "child-like learner": "an engineering metaphor for fast absorption + transfer, nothing more",
    "attention": "a routing/weighting mechanism, not focused awareness",
}

FORBIDDEN_CLAIM_WORDS = (
    "sentient",
    "sentience",
    "conscious",
    "consciousness",
    "self-aware",
    "self-awareness",
    "subjective experience",
    "qualia",
    "personhood",
    "feelings",
    "sapient",
    "sapience",
    "agency",  # 'agentic' is intentionally NOT here: it collides with the benign engineering sense
    "free will",  # ('agentic workflow/run'). 'agency' as a mental-state noun is the claim we guard.
    "self-determination",
)
FORBIDDEN_PREDICATES = (
    "wants",  # 3rd-person agentic forms only: "the system wants". Bare "want"/"desire" are omitted
    "desires",  # because they collide with benign first-person/noun use ("data I want", "design desire").
    "wishes",
    "longs",
    "yearns",
    "craves",
    "suffers",
)
_CLAIM_VERBS = (
    r"(?:is|are|am|was|were|be|being|been|becomes?|became|achiev\w*|attain\w*|gain\w*|"
    r"ha[sd]|have|possess\w*|experienc\w*|develops?|developed|reach\w*|emerg\w*|"
    r"exhibit\w*|display\w*|present|real|undeniabl\w*)"
)
_NEGATORS = (
    "not",
    "never",
    "non-",
    "without",
    "cannot",
    "can't",
    "isn't",
    "aren't",
    "doesn't",
    "don't",
    "won't",
    "n't",
    "neither",
    "nor",
    "avoid",
    "deny",
    "denies",
    "disclaim",
    "lacks",
    "lack ",
    "devoid of",
    "free of",
    "far from",
    "rules out",
    "no claim",
)
_AFFIRMING_DECOYS = (
    "no doubt",
    "without question",
    "without a doubt",
    "no longer",
    "make no mistake",
    "cannot be denied",
    "can not be denied",
    "cannot be questioned",
    "no question",
    "not in doubt",
    "beyond doubt",
    "undeniable",
    "undeniably",
)

_CLAUSE_RE = re.compile(r"[^.;:?!\n]+")
_NOUN_RE = re.compile(
    r"\b(" + "|".join(w.replace(" ", r"[\s-]+") for w in FORBIDDEN_CLAIM_WORDS) + r")\b", re.IGNORECASE
)
_VERB_RE = re.compile(r"\b" + _CLAIM_VERBS + r"\b", re.IGNORECASE)
_PRED_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_PREDICATES) + r")\b", re.IGNORECASE)


def scan_text(text: str) -> list[dict]:
    out: list[dict] = []
    for cm in _CLAUSE_RE.finditer(text):
        clause = cm.group(0)
        noun = _NOUN_RE.search(clause)
        pred = _PRED_RE.search(clause)
        is_claim = (noun is not None and _VERB_RE.search(clause) is not None) or pred is not None
        if not is_claim:
            continue
        probe = clause.lower()
        for decoy in _AFFIRMING_DECOYS:
            probe = probe.replace(decoy, " ")
        if any(neg in probe for neg in _NEGATORS):
            continue  # a genuine disclaimer in this clause
        word = (noun or pred).group(1)  # type: ignore[union-attr]
        line_no = text.count("\n", 0, cm.start()) + 1
        out.append({"line": line_no, "match": clause.strip()[:80], "word": word})
    return out


def assert_no_sentience_claims(text: str, where: str = "report") -> None:
    hits = scan_text(text)
    if hits:
        joined = "; ".join(f"L{h['line']}: {h['match']!r}" for h in hits[:5])
        raise ValueError(f"sentience-claim safety rail tripped in {where}: {joined}")


def safety_rail_note() -> str:
    return (
        "Safety rails (Frontier 35): this is a bounded engineered learner. It does NOT claim "
        "sentience, consciousness, self-awareness, feelings, suffering, wants, agency, or personhood. "
        "'drive'/'curiosity' mean engineered objective terms (novelty, uncertainty, learning "
        "progress); 'memory' means a data structure; 'self-monitoring' means diagnostics. Every "
        "capacity below is a measurement with a null hypothesis, not a philosophical claim."
    )
