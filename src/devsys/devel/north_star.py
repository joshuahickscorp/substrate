"""Sentience-adjacent NORTH STAR and SAFETY RAILS (Frontier 31 + 35). This project does not claim
sentience, consciousness, personhood, feelings, wants, suffering, or agency. It builds a bounded
engineered learner whose every "developmental" capacity is OPERATIONALIZED as a measurable quantity
with a null hypothesis, never a mystical claim. This module is the one place that states the loop,
the allowed engineering vocabulary, and a linter that flags grandiose claims in any report text.

The developmental loop (the north star, recorded in registries and reports):
  perceive -> remember -> predict -> notice surprise -> choose what to study -> adapt -> consolidate
  -> abstract -> transfer -> explain what changed -> choose the next lesson

Each arrow is a measurable capacity (see capacities.py), not a feeling.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

import re

DEVELOPMENTAL_LOOP = (
    "perceive -> remember -> predict -> notice surprise -> choose what to study -> adapt -> "
    "consolidate -> abstract -> transfer -> explain what changed -> choose the next lesson"
)

# the engineering vocabulary we ARE allowed to use, with its strictly operational meaning. These
# words are fine in code and reports; the rail below only flags AFFIRMATIVE sentience claims.
ENGINEERING_TERMS = {
    "drive": "an engineered objective term (novelty, uncertainty, learning-progress), not a wish",
    "memory": "a data structure / behavioral retention, not subjective memory",
    "self-monitoring": "diagnostics (uncertainty, calibration, failure detection), not self-awareness",
    "curiosity": "a data-selection objective (learning progress), not a feeling",
    "child-like learner": "an engineering metaphor for fast absorption + transfer, nothing more",
    "attention": "a routing/weighting mechanism, not focused awareness",
}

# Forbidden NOUNS/adjectives: a sentience/consciousness/agency concept that, reached by a claim verb
# in the same clause (either order), is an affirmative claim. May ONLY appear as a disclaimer.
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
# Forbidden PREDICATE verbs: mentalistic verbs that are a claim ON THEIR OWN (the system "wants" /
# "suffers"). Deliberately excludes engineered selection verbs (choose/select/prefer) used by the
# north-star loop, so "choose what to study" is never mistaken for desire.
FORBIDDEN_PREDICATES = (
    "wants",  # 3rd-person agentic forms only: "the system wants". Bare "want"/"desire" are omitted
    "desires",  # because they collide with benign first-person/noun use ("data I want", "design desire").
    "wishes",
    "longs",
    "yearns",
    "craves",
    "suffers",
)
# claim verbs that, with a forbidden NOUN in the same clause, constitute a claim (order-independent)
_CLAIM_VERBS = (
    r"(?:is|are|am|was|were|be|being|been|becomes?|became|achiev\w*|attain\w*|gain\w*|"
    r"ha[sd]|have|possess\w*|experienc\w*|develops?|developed|reach\w*|emerg\w*|"
    r"exhibit\w*|display\w*|present|real|undeniabl\w*)"
)
# genuine negation/disclaimer markers within a clause (bare 'no ' removed: it collides with no-doubt/
# no-longer/no-mistake, which are handled as affirming decoys below). 'rather than'/'instead of'/
# 'adjacent' removed: they do not negate a same-clause claim.
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
# phrases that LOOK negated but actually AFFIRM (double negation / rhetorical). Stripped before the
# negator check so they cannot mask a claim (e.g. 'no doubt the system is sentient' must still flag).
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

# split on STRONG clause boundaries only (sentence enders + ; : and newline), NOT commas, so a
# negated list ("does not claim sentience, consciousness, wants") stays one clause and passes.
_CLAUSE_RE = re.compile(r"[^.;:?!\n]+")
_NOUN_RE = re.compile(
    r"\b(" + "|".join(w.replace(" ", r"[\s-]+") for w in FORBIDDEN_CLAIM_WORDS) + r")\b", re.IGNORECASE
)
_VERB_RE = re.compile(r"\b" + _CLAIM_VERBS + r"\b", re.IGNORECASE)
_PRED_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_PREDICATES) + r")\b", re.IGNORECASE)


def scan_text(text: str) -> list[dict]:
    """Find AFFIRMATIVE sentience/consciousness/agency claims in `text` (empty == clean).

    Clause-scoped: the text is split on STRONG boundaries (. ; : ? ! newline), and a clause is a CLAIM
    if it co-occurs a forbidden noun with a claim verb (either order) OR a forbidden mentalistic
    predicate verb (wants/suffers/...), UNLESS the SAME clause carries a genuine negator (so disclaimers
    pass). Rhetorical double-negatives (no doubt / without question / cannot be denied) are stripped
    first so they cannot mask a claim. Clause scoping kills the cross-clause negator-poisoning bypass;
    order-independent matching catches noun-first/passive claims. Biased toward flagging (a safety rail)."""
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
    """Raise if `text` contains an affirmative sentience claim. Used to gate generated reports."""
    hits = scan_text(text)
    if hits:
        joined = "; ".join(f"L{h['line']}: {h['match']!r}" for h in hits[:5])
        raise ValueError(f"sentience-claim safety rail tripped in {where}: {joined}")


def safety_rail_note() -> str:
    """The standard disclaimer block stamped into developmental reports (always a DISCLAIMER, so it
    passes its own scanner). States what the measured capacities are and are not."""
    return (
        "Safety rails (Frontier 35): this is a bounded engineered learner. It does NOT claim "
        "sentience, consciousness, self-awareness, feelings, suffering, wants, agency, or personhood. "
        "'drive'/'curiosity' mean engineered objective terms (novelty, uncertainty, learning "
        "progress); 'memory' means a data structure; 'self-monitoring' means diagnostics. Every "
        "capacity below is a measurement with a null hypothesis, not a philosophical claim."
    )
