"""Loaders + validators for the developmental registries: paradigms (Frontier 20), the capacity
ladder (Frontier 32), and paper-watch (Frontier 30). Pure config + filesystem reads, no network.

Each registry has a closed schema and honesty rules enforced in code so a malformed or over-claiming
entry is caught before it reaches a report. A paradigm is always a CANDIDATE (never canonical science
on its own); the capacity ladder rungs are measurements with nulls; paper-watch is offline-first.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from ..config import REPO_ROOT
from .north_star import scan_text

REGISTRY_DIR = REPO_ROOT / "registry"
PARADIGMS_YAML = REGISTRY_DIR / "paradigms.yaml"
CAPACITIES_YAML = REGISTRY_DIR / "capacities.yaml"
PAPERWATCH_YAML = REGISTRY_DIR / "paperwatch.yaml"

# ---- closed vocabularies ----
PARADIGM_CHANGES = (
    "data",
    "memory",
    "optimizer",
    "architecture",
    "objective",
    "routing",
    "inference",
    "curriculum",
    "evaluation",
)
PARADIGM_STATUS = ("implement-now", "prototype-local", "studio-later", "paper-watch", "blocked")
PARADIGM_TAGS = (
    "plasticity",
    "abstraction",
    "memory",
    "local-learning",
    "meta-learning",
    "active-curriculum",
    "world-model",
    "modularity",
    "test-time-adaptation",
    "neuro-symbolic",
    "generative",
    "multimodal",
    "self-monitoring",
)
PARADIGM_COMPAT = ("cached-pooled", "dense-2.1", "action-conditioned", "metadata")
CAPACITY_STATUS = ("measured-now", "prototype-local", "studio-later", "deferred")
# capability families + honest provenance tags the capacity rungs may carry (closed sets)
CAPACITY_TAGS = (
    "perception",
    "abstraction",
    "memory",
    "plasticity",
    "active-curriculum",
    "world-model",
    "self-monitoring",
    "multimodal",
    "meta-learning",
)
CAPACITY_PROVENANCE = (
    "natural-video",
    "real-encoder",
    "structured-synthetic",
    "provisional",
    "metadata-only",
    "auxiliary",
)
PAPERWATCH_STATUS = ("active", "watching", "stale")
# free-text capacity/paradigm fields scanned by the sentience rail (no claim may hide in prose)
_SCANNED_CAP_FIELDS = ("thesis", "metric", "failure_interpretation", "promotion_rule")
_SCANNED_PAR_FIELDS = ("thesis", "expected_signature", "null_interpretation")

# F20 fields authored in the YAML; the loader normalizes the F37 contract (baseline/ablation/metric/
# null_hypothesis/local_test) from these so a paradigm presents the same contract a capacity does.
_PARADIGM_YAML_FIELDS = (
    "slug",
    "name",
    "thesis",
    "family",
    "changes",
    "vjepa_compat",
    "mvp_local_test",
    "studio_test",
    "compute_risk",
    "expected_signature",
    "null_interpretation",
    "tags",
    "status",
)
_PARADIGM_CONTRACT = ("baseline", "ablation", "metric", "null_hypothesis", "local_test", "status")

CAPACITY_REQUIRED = (
    "rung",
    "name",
    "tag",
    "thesis",
    "metric",
    "baseline",
    "ablation",
    "null_hypothesis",
    "local_test",
    "studio_test",
    "failure_interpretation",
    "provenance_tag",
    "status",
    "promotion_rule",
)
PAPERWATCH_REQUIRED = ("slug", "title", "why", "primary", "last_checked", "status")


def _sentience_scan(label: str, entry: dict, fields: tuple) -> list[str]:
    """Scan an entry's free-text fields for affirmative sentience claims (the rail, at the registry
    boundary, so a claim cannot hide in a thesis/metric and surface in a report later)."""
    problems: list[str] = []
    for f in fields:
        for hit in scan_text(str(entry.get(f, ""))):
            problems.append(f"{label}: sentience claim in {f!r}: {hit['match']!r}")
    return problems


def _load(path: Path, key: str) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"registry file missing: {path}")
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(data, dict) and key in data, f"{path} missing top-level {key!r}"
    return [dict(e) for e in data[key]]


# ---------------------------------------------------------------- paradigms (Frontier 20)
def load_paradigms(path: Path | None = None) -> list[dict]:
    """Paradigm candidates, sorted by status priority then slug. Each entry is NORMALIZED so it
    presents the full F37 contract (baseline/ablation/metric/null_hypothesis/local_test) derived from
    its F20 fields: metric<-expected_signature, null_hypothesis<-null_interpretation,
    local_test<-mvp_local_test, with generic-but-honest baseline/ablation (a candidate's baseline IS
    a matched control and its ablation IS disabling the mechanism)."""
    out = _load(path or PARADIGMS_YAML, "paradigms")
    order = {s: i for i, s in enumerate(PARADIGM_STATUS)}
    for e in out:
        e.setdefault("metric", e.get("expected_signature"))
        e.setdefault("null_hypothesis", e.get("null_interpretation"))
        e.setdefault("local_test", e.get("mvp_local_test"))
        e.setdefault("baseline", "matched-parameter control without the mechanism")
        e.setdefault("ablation", "disable the mechanism (the expected signature must vanish)")
    out.sort(key=lambda e: (order.get(str(e.get("status")), 99), str(e.get("slug"))))
    return out


def validate_paradigm(entry: dict) -> list[str]:
    """Problems with one paradigm entry (empty == valid). Schema (F20 fields), closed vocab, and the
    honesty rule that a paradigm is a CANDIDATE: it may never carry a real/natural-video result tag."""
    p: list[str] = []
    slug = entry.get("slug", "<no-slug>")
    for k in _PARADIGM_YAML_FIELDS:
        if k not in entry:
            p.append(f"{slug}: missing required field {k!r}")
    for k in _PARADIGM_CONTRACT:  # present after normalization (load_paradigms); guards raw dicts too
        if not entry.get(k):
            p.append(
                f"{slug}: missing contract field {k!r} (baseline/ablation/metric/null/local_test/status)"
            )
    if p:
        return p
    if entry["changes"] not in PARADIGM_CHANGES:
        p.append(f"{slug}: changes {entry['changes']!r} not in {PARADIGM_CHANGES}")
    if entry["status"] not in PARADIGM_STATUS:
        p.append(f"{slug}: status {entry['status']!r} not in {PARADIGM_STATUS}")
    if entry["vjepa_compat"] not in PARADIGM_COMPAT:
        p.append(f"{slug}: vjepa_compat {entry['vjepa_compat']!r} not in {PARADIGM_COMPAT}")
    if not isinstance(entry["tags"], list):
        p.append(f"{slug}: tags must be a list, got {type(entry['tags']).__name__}")
    else:
        bad_tags = [t for t in entry["tags"] if t not in PARADIGM_TAGS]
        if bad_tags:
            p.append(f"{slug}: unknown tags {bad_tags} (allowed {PARADIGM_TAGS})")
    # honesty: a candidate cannot smuggle in a canonical result claim (.strip() so a padded tag cannot evade)
    if str(entry.get("result_tag", "")).strip().lower() in ("real-encoder", "natural-video"):
        p.append(f"{slug}: a paradigm candidate may not claim a canonical result_tag")
    # an implement-now paradigm must have a concrete (non-deferred) local test
    if entry["status"] == "implement-now" and str(entry["mvp_local_test"]).lower().startswith("deferred"):
        p.append(f"{slug}: status implement-now but mvp_local_test is deferred (contradiction)")
    p += _sentience_scan(slug, entry, _SCANNED_PAR_FIELDS)  # no sentience claim may hide in prose
    return p


def validate_paradigms(path: Path | None = None) -> list[str]:
    items = load_paradigms(path)
    problems: list[str] = []
    seen: set = set()
    for e in items:
        problems += validate_paradigm(e)
        if e.get("slug") in seen:
            problems.append(f"duplicate paradigm slug {e.get('slug')!r}")
        seen.add(e.get("slug"))
    return problems


# ---------------------------------------------------------------- capacity ladder (Frontier 32)
def load_capacities(path: Path | None = None) -> list[dict]:
    """The developmental capacity ladder, sorted by rung."""
    out = _load(path or CAPACITIES_YAML, "capacities")
    out.sort(key=lambda e: int(e.get("rung", 0)))
    return out


def validate_capability(entry: dict) -> list[str]:
    """Problems with one capacity rung (empty == valid). Every rung is a measurement with a null:
    all CAPACITY_REQUIRED fields present, status in the closed set, and (the sentience rail) the
    thesis/metric must read as a measurement, never a claim (checked by north_star at report time)."""
    p: list[str] = []
    name = entry.get("name", "<no-name>")
    for k in CAPACITY_REQUIRED:
        if k not in entry or entry.get(k) in (None, ""):
            p.append(f"rung {entry.get('rung', '?')} ({name}): missing required field {k!r}")
    if p:
        return p
    if entry["status"] not in CAPACITY_STATUS:
        p.append(f"{name}: status {entry['status']!r} not in {CAPACITY_STATUS}")
    if entry["tag"] not in CAPACITY_TAGS:
        p.append(f"{name}: tag {entry['tag']!r} not in {CAPACITY_TAGS}")
    if entry["provenance_tag"] not in CAPACITY_PROVENANCE:
        p.append(f"{name}: provenance_tag {entry['provenance_tag']!r} not in {CAPACITY_PROVENANCE}")
    if not isinstance(entry["rung"], int) or entry["rung"] < 1:
        p.append(f"{name}: rung must be a positive integer")
    p += _sentience_scan(name, entry, _SCANNED_CAP_FIELDS)  # a rung is a measurement, never a claim
    return p


def validate_capacities(path: Path | None = None) -> list[str]:
    items = load_capacities(path)
    problems: list[str] = []
    rungs = [int(e.get("rung", 0)) for e in items]
    if rungs != sorted(set(rungs)) or len(rungs) != len(set(rungs)):
        problems.append(f"capacity rungs must be unique and ordered, got {rungs}")
    for e in items:
        problems += validate_capability(e)
    return problems


# ---------------------------------------------------------------- paper-watch (Frontier 30)
def load_paperwatch(path: Path | None = None) -> list[dict]:
    return _load(path or PAPERWATCH_YAML, "topics")


def validate_paperwatch(path: Path | None = None) -> list[str]:
    items = load_paperwatch(path)
    problems: list[str] = []
    for e in items:
        slug = e.get("slug", "<no-slug>")
        for k in PAPERWATCH_REQUIRED:
            if k not in e or e.get(k) in (None, ""):
                problems.append(f"paperwatch {slug}: missing field {k!r}")
        if e.get("status") and e["status"] not in PAPERWATCH_STATUS:
            problems.append(f"paperwatch {slug}: status {e['status']!r} not in {PAPERWATCH_STATUS}")
        if "primary" in e and not isinstance(e["primary"], list):
            problems.append(f"paperwatch {slug}: primary must be a list of sources")
    return problems


def validate_all() -> list[str]:
    """Validate all three developmental registries in one call (used by the doctor and tests)."""
    return validate_paradigms() + validate_capacities() + validate_paperwatch()
