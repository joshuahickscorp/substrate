"""Report generation from sealed fields only.

Two historical defects: a report read a key that did not exist and returned None, and a synthesis softened a
sealed invalid verdict into the word marginal. Both are report layer failures, and both are fatal to the
reader even when the underlying receipts are sound.

Every report value binds to an artifact, a JSON pointer, a transformation and a classification. Resolution
failure raises. Prose that asserts a stronger verdict class than the sealed one fails.

House style: no dashes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


class ReportFieldError(Exception):
    pass


# ---------------------------------------------------------------- pointer resolution


def resolve(doc, pointer: str):
    """RFC 6901 style pointer with a hard failure on every miss. Never returns a silent None."""
    if pointer in ("", "/"):
        return doc
    cur = doc
    for raw in pointer.lstrip("/").split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict):
            if key not in cur:
                raise ReportFieldError(f"pointer {pointer!r}: key {key!r} absent, keys are {sorted(cur)[:12]}")
            cur = cur[key]
        elif isinstance(cur, list):
            if not key.lstrip("-").isdigit() or int(key) >= len(cur):
                raise ReportFieldError(f"pointer {pointer!r}: index {key!r} out of range {len(cur)}")
            cur = cur[int(key)]
        else:
            raise ReportFieldError(f"pointer {pointer!r}: cannot descend into {type(cur).__name__}")
    return cur


def bind(root: Path, artifact: str, pointer: str, *, expect=None, transform=None, classification: str = "measured"):
    p = Path(root) / artifact
    if not p.is_file():
        raise ReportFieldError(f"artifact {artifact!r} does not exist under {root}")
    value = resolve(json.loads(p.read_text()), pointer)
    if value is None:
        raise ReportFieldError(f"{artifact}{pointer} resolved to None")
    if expect is not None and not isinstance(value, expect):
        raise ReportFieldError(f"{artifact}{pointer} is {type(value).__name__}, expected {expect.__name__}")
    if transform is not None:
        value = transform(value)
    return {
        "value": value,
        "source_artifact": artifact,
        "pointer": pointer,
        "transformation": getattr(transform, "__name__", "identity"),
        "classification": classification,
    }


# ---------------------------------------------------------------- verdict wording


VERDICT_STRENGTH = {
    "invalid": 0,
    "inactive_instrumentation": 0,
    "insufficient_power": 0,
    "unconverged_baseline": 0,
    "harm": 1,
    "null": 1,
    "mixed": 2,
    "supported": 3,
    "positive": 4,
}

# Words that assert membership of a verdict class. Prose may not use a word from a class stronger than the
# sealed verdict. The mixed row is the one that caught the fast state synthesis.
CLASS_TERMS = {
    "mixed": ("marginal", "partial", "suggestive", "promising", "trend", "nearly", "borderline", "encouraging"),
    "supported": ("supported", "consistent with the premise", "evidence for", "works", "holds"),
    "positive": (
        "positive",
        "confirms",
        "beats",
        "improves",
        "licensed",
        "selected",
        "activation is granted",
        "demonstrates",
        "wins",
    ),
}


def verdict_class(sealed: str) -> str:
    s = str(sealed).lower()
    for key in ("invalid", "inactive_instrumentation", "insufficient_power", "unconverged_baseline"):
        if key in s:
            return key
    for key in ("harm", "null", "mixed", "supported", "positive"):
        if key in s:
            return key
    return "null"


def wording_check(prose: str, sealed_verdict: str) -> dict:
    """Prose may narrow or restate a sealed verdict. It may never broaden it."""
    sealed_class = verdict_class(sealed_verdict)
    limit = VERDICT_STRENGTH[sealed_class]
    low = prose.lower()
    offenders = []
    # A term inside a negation does not assert its class. This handles the short forms that actually occur,
    # "not licensed", "no architecture is selected", "never positive", and nothing cleverer.
    # ponytail: three word negation window, replace with a parser only if a real sentence defeats it
    negators = r"(?:not|no|never|nothing|neither|cannot|fails? to|remains? false)"
    for cls, terms in CLASS_TERMS.items():
        if VERDICT_STRENGTH[cls] <= limit:
            continue
        for t in terms:
            pat = rf"\b{re.escape(t)}\b"
            hits = [m for m in re.finditer(pat, low)]
            for m in hits:
                window = low[max(0, m.start() - 40) : m.start()]
                if re.search(rf"\b{negators}\b(?:\s+\w+){{0,3}}\s*$", window):
                    continue
                offenders.append({"term": t, "asserts_class": cls, "sealed_class": sealed_class})
                break
    return {
        "sealed_verdict": sealed_verdict,
        "sealed_class": sealed_class,
        "offenders": offenders,
        "passes": not offenders,
    }


# ---------------------------------------------------------------- report assembly


def render(root: Path, spec: dict, prose: dict | None = None) -> dict:
    """spec maps question to a binding descriptor. Any unresolvable field fails the whole report."""
    fields, errors = {}, []
    for question, d in spec.items():
        try:
            fields[question] = bind(
                root,
                d["artifact"],
                d["pointer"],
                expect=d.get("expect"),
                transform=d.get("transform"),
                classification=d.get("classification", "measured"),
            )
        except ReportFieldError as e:
            errors.append(f"{question}: {e}")
    wording = {}
    for label, (text, sealed) in (prose or {}).items():
        w = wording_check(text, sealed)
        wording[label] = w
        if not w["passes"]:
            errors.append(f"{label}: prose broadens the sealed verdict {sealed!r} via {w['offenders']}")
    if errors:
        raise ReportFieldError("; ".join(errors))
    return {"fields": fields, "wording": wording, "n_fields": len(fields), "all_resolved": True}


def validate_spec(spec: dict, declared_output_schema: dict) -> dict:
    """Resolve every binding against the declared output schema before the experiment runs.

    This is the preregistration time half of the missing key defect. A pointer that cannot resolve against
    the schema the producer promises to emit is a broken report before a single update has been spent.
    """
    errors = []
    for question, d in spec.items():
        try:
            resolve(declared_output_schema, d["pointer"])
        except ReportFieldError as e:
            errors.append(f"{question}: {e}")
        except KeyError:
            errors.append(f"{question}: binding has no pointer")
    return {"n_fields": len(spec), "errors": errors, "passes": not errors}


def audit_report(root: Path, spec: dict, prose: dict | None = None) -> dict:
    """Non raising form for auditing an existing report. Returns the failures instead of throwing."""
    try:
        r = render(root, spec, prose)
        return {"passes": True, "errors": [], **r}
    except ReportFieldError as e:
        return {"passes": False, "errors": str(e).split("; ")}
