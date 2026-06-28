"""Acquisition registry loader + validator (Frontier 1 datasets, Frontier 2 models). Reads
registry/datasets.yaml and registry/models.yaml, validates every entry against ONE schema, and
enforces the honesty rules that keep the catalog from lying: a source needing signed terms is
manual, full Ego4D is deferred, a distilled/quantized model can never claim to replace canonical
V-JEPA, and only canonical encoders may carry the real-encoder tag.

Pure config + filesystem read, no network. The planner, downloader, data-card writer, and license
ledger all consume load_datasets()/load_models(); validation runs in the doctor and in tests so a
malformed or dishonest registry is caught before any Studio hour is spent.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from ..config import REPO_ROOT
from ..substrate.encoder_registry import list_encoders, verified_real_ids

REGISTRY_DIR = REPO_ROOT / "registry"
DATASETS_YAML = REGISTRY_DIR / "datasets.yaml"
MODELS_YAML = REGISTRY_DIR / "models.yaml"

# the closed vocabularies the schema pins
STATUSES = ("available", "manual", "deferred", "blocked", "metadata-only")
KINDS = ("natural-video", "structured-synthetic", "metadata-only", "auxiliary", "provisional")
MODEL_ROLES = ("canonical", "auxiliary", "distilled", "quantized")
RISK_FACETS = ("license", "size", "link_rot", "annotation_quality", "privacy", "decode")

# every dataset entry must carry these keys (the machine-readable contract from the goal)
DATASET_REQUIRED = (
    "slug",
    "name",
    "modality",
    "domain",
    "task_fit",
    "priority",
    "raw_gb",
    "cache_gb_pooled",
    "recommended_subsets",
    "license",
    "citation",
    "allowed_use",
    "auth_steps",
    "download_method",
    "resume_method",
    "checksum_plan",
    "status",
    "risk",
    "output_layout",
    "kind",
)
MODEL_REQUIRED = (
    "slug",
    "name",
    "role",
    "hf_id",
    "embed_dim",
    "dense",
    "available",
    "result_tag",
    "replaces_canonical",
)

# slugs that must stay deferred no matter what (far beyond a 1 TB local disk)
ALWAYS_DEFERRED = frozenset({"ego4d_full", "ego_exo4d_subset"})


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"registry file missing: {path}")
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(data, dict), f"{path} must be a mapping"
    return data


def load_datasets(path: Path | None = None) -> list[dict]:
    """Every dataset source as a flat dict, sorted by descending priority then slug. Pure read."""
    data = _load_yaml(path or DATASETS_YAML)
    sources = data.get("sources", [])
    out = [dict(s) for s in sources]
    out.sort(key=lambda d: (-int(d.get("priority", 0)), str(d.get("slug", ""))))
    return out


def get_dataset(slug: str, path: Path | None = None) -> dict:
    """One source by slug, or raise with the known set."""
    for d in load_datasets(path):
        if d.get("slug") == slug:
            return d
    known = [d.get("slug") for d in load_datasets(path)]
    raise KeyError(f"unknown dataset slug {slug!r}; known: {known}")


def validate_dataset(entry: dict) -> list[str]:
    """Problems with ONE dataset entry (empty == valid). Schema + honesty in one place.

    Schema: every DATASET_REQUIRED key present; status in STATUSES; kind in KINDS; sizes numeric and
    non-negative; risk a dict over RISK_FACETS. Honesty: a source whose auth_steps describe signed
    terms must not be status available (it must be manual/deferred/blocked); the ALWAYS_DEFERRED
    slugs must be status deferred; a metadata-only source must not claim a video cache size."""
    p: list[str] = []
    slug = entry.get("slug", "<no-slug>")
    for k in DATASET_REQUIRED:
        if k not in entry:
            p.append(f"{slug}: missing required field {k!r}")
    if p:  # do not type-check fields that are absent
        return p

    if entry["status"] not in STATUSES:
        p.append(f"{slug}: status {entry['status']!r} not in {STATUSES}")
    if entry["kind"] not in KINDS:
        p.append(f"{slug}: kind {entry['kind']!r} not in {KINDS}")
    for size_key in ("raw_gb", "cache_gb_pooled"):
        v = entry[size_key]
        if not isinstance(v, int | float) or v < 0:
            p.append(f"{slug}: {size_key} must be a number >= 0, got {v!r}")
    if not isinstance(entry["recommended_subsets"], list):
        p.append(f"{slug}: recommended_subsets must be a list")
    risk = entry["risk"]
    if not isinstance(risk, dict):
        p.append(f"{slug}: risk must be a mapping over {RISK_FACETS}")
    else:
        missing = [f for f in RISK_FACETS if f not in risk]
        if missing:
            p.append(f"{slug}: risk missing facets {missing}")

    # honesty: gated-access language in auth_steps means it cannot be 'available'. Broad on purpose:
    # the safe direction is to force a gated source out of 'available' (manual), not the reverse.
    auth = str(entry["auth_steps"]).lower()
    _SIGNED_MARKERS = (
        "sign",
        "signed",
        "license agreement",
        "accept the",
        "accepted terms",
        "agree to",
        "terms of use",
        "data use agreement",
        "dua",
        "eula",
        "request access",
        "apply for",
        "application",
        "approval",
        "approved",
        "credentials",
        "registration required",
        "gated",
    )
    signed = any(w in auth for w in _SIGNED_MARKERS)
    if signed and entry["status"] == "available":
        p.append(f"{slug}: auth_steps describe signed terms but status is 'available' (should be manual)")
    if slug in ALWAYS_DEFERRED and entry["status"] != "deferred":
        p.append(f"{slug}: must be status 'deferred' (beyond a 1 TB local disk), got {entry['status']!r}")
    if entry["kind"] == "metadata-only" and float(entry["cache_gb_pooled"]) > 0:
        p.append(f"{slug}: metadata-only source must not claim a video cache size (cache_gb_pooled>0)")
    return p


def validate_datasets(path: Path | None = None) -> list[str]:
    """Validate the whole dataset registry, including cross-entry checks (unique slugs)."""
    sources = load_datasets(path)
    problems: list[str] = []
    seen: set[str | None] = set()
    for d in sources:
        problems += validate_dataset(d)
        s = d.get("slug")
        if s in seen:
            problems.append(f"duplicate dataset slug {s!r}")
        seen.add(s)
    return problems


def load_models(path: Path | None = None) -> list[dict]:
    """Merged model registry: canonical encoders (from configs/encoder, the science substrates)
    PLUS the optional auxiliary/distilled/quantized rows from registry/models.yaml, each tagged with
    role. Canonical rows are synthesized from encoder_registry so there is ONE source of truth for
    V-JEPA; availability of a canonical row is asserted against VERIFIED_REAL_IDS (hand-verified on
    HF), never probed at runtime."""
    verified = verified_real_ids()
    out: list[dict] = []
    for e in list_encoders():
        hf = e["hf_id"]
        out.append(
            {
                "slug": e["name"],
                "name": e["name"],
                "role": "canonical",
                "hf_id": hf,
                "embed_dim": e["embed_dim"],
                "dense": e["dense"],
                "available": hf in verified,  # real only if the hf_id is hand-verified present
                "result_tag": "real-encoder" if hf in verified else "provisional",
                "replaces_canonical": False,  # a canonical row IS the substrate; n/a but pinned False
                "note": "canonical frozen V-JEPA substrate" if hf in verified else "deferred (not on HF)",
            }
        )
    data = _load_yaml(path or MODELS_YAML)
    for m in data.get("models", []):
        out.append(dict(m))
    return out


def validate_model(entry: dict) -> list[str]:
    """Problems with ONE model entry (empty == valid). Schema + the canonical-protection invariants.

    Honesty: role in MODEL_ROLES; only role canonical may carry result_tag real-encoder; distilled
    and quantized rows must declare replaces_canonical false (an approximation never stands in for
    the frozen substrate); a row claiming available true must name an hf_id (cannot be real with no
    weights to point at)."""
    p: list[str] = []
    slug = entry.get("slug", "<no-slug>")
    for k in MODEL_REQUIRED:
        if k not in entry:
            p.append(f"{slug}: missing required field {k!r}")
    if p:
        return p
    if entry["role"] not in MODEL_ROLES:
        p.append(f"{slug}: role {entry['role']!r} not in {MODEL_ROLES}")
    if entry["result_tag"] == "real-encoder" and entry["role"] != "canonical":
        p.append(f"{slug}: only canonical models may carry result_tag 'real-encoder'")
    # a real-encoder canonical claim must point at a HAND-VERIFIED HF weight set, never an arbitrary id
    if (
        entry["role"] == "canonical"
        and entry["result_tag"] == "real-encoder"
        and str(entry["hf_id"]) not in verified_real_ids()
    ):
        p.append(
            f"{slug}: canonical+real-encoder but hf_id {entry['hf_id']!r} is not in VERIFIED_REAL_IDS "
            f"(cannot claim real weights that are not hand-verified on HF)"
        )
    # ANY non-canonical model (auxiliary/distilled/quantized) must never claim to replace canonical
    if entry["role"] != "canonical" and entry["replaces_canonical"]:
        p.append(f"{slug}: role {entry['role']!r} must NOT set replaces_canonical (it never replaces V-JEPA)")
    if entry["available"] and not str(entry["hf_id"]).strip() and entry["role"] not in ("distilled",):
        p.append(f"{slug}: available true but no hf_id to load (dishonest availability)")
    if int(entry["embed_dim"]) <= 0:
        p.append(f"{slug}: embed_dim must be positive")
    return p


def validate_models(path: Path | None = None) -> list[str]:
    """Validate the whole model registry, including unique slugs and at least one canonical row."""
    models = load_models(path)
    problems: list[str] = []
    seen: set[str | None] = set()
    for m in models:
        problems += validate_model(m)
        s = m.get("slug")
        if s in seen:
            problems.append(f"duplicate model slug {s!r}")
        seen.add(s)
    if not any(m.get("role") == "canonical" for m in models):
        problems.append("no canonical model in the registry (frozen V-JEPA substrate missing)")
    return problems


def validate_registry(datasets_path: Path | None = None, models_path: Path | None = None) -> list[str]:
    """Validate both registries in one call (used by the doctor and the pipeline preflight)."""
    return validate_datasets(datasets_path) + validate_models(models_path)
