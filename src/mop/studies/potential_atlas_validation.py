"""Fail-closed validation for the evidence-grounded MOP potential atlas.

The atlas is a snapshot with embedded source hashes. This validator recomputes its structural,
scoring, evidence, planning-partition, dependency, queue, and hardware-escalation invariants. It
does not update the atlas, any experiment ledger, or any running campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard

from ..config import REPO_ROOT

ATLAS_SCHEMA = "mop-potential-atlas/v1"
RECEIPT_SCHEMA = "mop-potential-atlas-validation/v1"
REQUIREMENTS_SCHEMA = "mop-extended-compute-requirements/v1"
EXPECTED_FACET_COUNT = 41
EXPECTED_DOMAIN_COUNT = 7
EXPECTED_WEIGHT_TOTAL = 100
EXPECTED_REQUIREMENTS_ROW_COUNT = 321
EXPECTED_CATEGORY2_COUNT = 119
EXPECTED_PRIMARY_CATEGORY_COUNTS = {1: 168, 2: 119, 3: 29, 6: 5}
P5_VERIFICATION_PATH = "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json"
P5_VERIFICATION_SCHEMA = "mop-p5-context-independent-verifier/v1"
P6_PREREQUISITE_REASON = "P6 and other dependent tasks fail closed until immutable prior receipts pass"
EXPECTED_EXHAUSTION_COUNTS = {
    "already-durable-hash-verifiable": 37,
    "freshly-executed-verified": 117,
    "implementation-blocked": 1,
    "measured-hardware-blocked": 0,
    "rights-data-blocked": 11,
    "runnable-not-yet-run": 3,
    "upstream-model-blocked": 8,
}
EXPECTED_FACET_IDS = frozenset(
    {
        "EV1",
        "EV2",
        "EV3",
        "EV4",
        "EV5",
        "EV6",
        "SR1",
        "SR2",
        "SR3",
        "SR4",
        "SR5",
        "SR6",
        "RA1",
        "RA2",
        "RA3",
        "RA4",
        "RA5",
        "RA6",
        "PA1",
        "PA2",
        "PA3",
        "PA4",
        "PA5",
        "PA6",
        "PA7",
        "PA8",
        "PA9",
        "OP1",
        "OP2",
        "OP3",
        "OP4",
        "OP5",
        "BM1",
        "BM2",
        "BM3",
        "BM4",
        "SG1",
        "SG2",
        "SG3",
        "SG4",
        "SG5",
    }
)
EXPECTED_DOMAIN_IDS = frozenset(
    {
        "evidence_data_substrate",
        "sensing_representation",
        "prediction_reasoning_action",
        "persistent_adaptive_ecology",
        "owned_substrate_performance",
        "bio_morphogenic_material",
        "safety_security_welfare",
    }
)
EXPECTED_QUEUE_IDS = frozenset(
    {
        "operationalize_missing_atlas_facets",
        "run_ordered_p5_campaign",
        "universal_evidence_identity",
        "native_audiovisual_intake",
        "natural_dense_e6_dr14_campaign",
        "natural_same_input_control_battery",
        "p6_progressive_execution_ladder",
        "bounded_generated_ecology",
        "unified_memory_lifecycle",
        "post_cm7_substrate_surface",
        "action_world_model_external_validity",
        "workspace_and_self_model_remaining_gates",
        "simulated_social_populations",
        "governed_rewrite_drill",
        "material_digital_twin",
        "survivor_confirmation",
    }
)
EXPECTED_CLUSTER_IDS = frozenset(
    {
        "H1_temporal_binding_acquisition",
        "H2_action_boundary_world_model",
        "H3_memory_workspace_self_model",
        "H4_lifetime_plasticity_openended",
        "H5_social_reference_culture",
        "H6_transactional_safety_material",
        "H7_dense_substrate_controls_search",
        "H8_execution_density",
    }
)
EXPECTED_GRAPH_NODES = frozenset(
    {"EV", "SENSE", "EVENT", "PRED", "ACT", "MEM", "WORK", "OPEN", "OWN", "EXEC", "SAFE", "MAT"}
)
SCORE_DIMENSIONS = ("scaffolding", "implementation", "experiment", "confirmation")
SCORE_WEIGHTS = {
    "scaffolding": 0.20,
    "implementation": 0.25,
    "experiment": 0.30,
    "confirmation": 0.25,
}
EXPECTED_RAW_FORMULA = "0.20*S + 0.25*I + 0.30*E + 0.25*C"
EXPECTED_BOTTLENECK_CAPS = (
    "score <= S + 2.5",
    "score <= I + 2.0",
    "score <= E + 1.5",
    "score <= C + 2.0",
)
EXPECTED_ROUNDING = "round the capped score to one decimal"
REQUIRED_BOUND_SOURCES = (
    "proof/EXTENDED_COMPUTE_REQUIREMENTS.json",
    "proof/PROJECT_EXPERIMENT_EXHAUSTION.json",
    "proof/FRONTIER_LOCALIZATION.json",
    "proof/FORM_SUBSTRATE/PRE_STUDIO_BOUNDARY.json",
    "proof/COMPLETION_CLAIM_AUDIT.json",
    "proof/E6_VITB_DENSE_PREFLIGHT.json",
    "proof/EXPANSION_WAVE0.json",
    "proof/CONTINUAL_MILLION_EVENT_PREFLIGHT.json",
    "proof/LOCAL_EXECUTION_THROTTLE_P6_10K_DRY_RUN.json",
    "proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json",
    "proof/P9_CAUSAL_MONITORING_PREFLIGHT.json",
)
RETIRED_CURRENT_PATHS = frozenset(
    {
        "proof/VJEPA_SCALE_ATLAS_LOCAL.json",
        "proof/FACTORIZED_STIMULUS_IDENTITY.json",
        "proof/REAL_ENCODER_LOCAL_ATTEMPT.json",
        "proof/REAL_ENCODER_VITH_LOCAL8.json",
        "proof/REAL_ENCODER_VITG_LOCAL8.json",
        "data/cache/vjepa2_vitl_local8_random_s0/cache_manifest.json",
    }
)
RETIRED_TEXT_MARKERS = (
    "VJEPA_SCALE_ATLAS_LOCAL",
    "FACTORIZED_STIMULUS_IDENTITY",
    "REAL_ENCODER_LOCAL_ATTEMPT",
    "REAL_ENCODER_VITH_LOCAL8",
    "REAL_ENCODER_VITG_LOCAL8",
    "vjepa2_vitl_local8",
    "ViT-H",
    "ViT-G",
    "L/H/g",
)
RETIRED_FRONTIER_MARKERS = tuple(
    marker for marker in RETIRED_TEXT_MARKERS if marker != "REAL_ENCODER_LOCAL_ATTEMPT"
)


def _finite_number(value: Any) -> TypeGuard[int | float]:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _round_one_decimal(value: float) -> float:
    """Apply the atlas generator's Python one-decimal rounding rule."""
    return round(value, 1)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> TypeGuard[str]:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    if resolved.is_relative_to(root):
        return str(resolved.relative_to(root))
    return str(resolved)


def _file_receipt(path: Path, repo_root: Path) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "path": _display_path(path, repo_root),
        "exists": path.is_file(),
    }
    if path.is_file():
        receipt.update({"bytes": path.stat().st_size, "sha256": _sha256(path)})
    return receipt


def _repo_file(repo_root: Path, raw: Any) -> tuple[Path | None, str | None]:
    if not isinstance(raw, str) or not raw.strip():
        return None, "path is not a non-empty string"
    relative = Path(raw)
    if relative.is_absolute():
        return None, f"absolute path is forbidden: {raw}"
    resolved = (repo_root / relative).resolve()
    if not resolved.is_relative_to(repo_root.resolve()):
        return None, f"path escapes repository: {raw}"
    return resolved, None


def _check(name: str, problems: list[str], detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "ok": not problems,
        "problems": problems,
        "detail": detail or {},
    }


def _guard_pair(
    label: str, operation: Callable[[], tuple[list[str], dict[str, Any]]]
) -> tuple[list[str], dict[str, Any]]:
    """Convert malformed-input exceptions into a closed validation failure."""
    try:
        return operation()
    except Exception as error:  # noqa: BLE001 - malformed artifacts must produce a receipt
        return [f"{label} validator refused malformed input: {type(error).__name__}: {error}"], {}


def _guard_source(
    operation: Callable[[], tuple[list[str], dict[str, Any], dict[str, dict[str, Any]]]],
) -> tuple[list[str], dict[str, Any], dict[str, dict[str, Any]]]:
    """Convert source-snapshot parser errors into a closed validation failure."""
    try:
        return operation()
    except Exception as error:  # noqa: BLE001 - malformed artifacts must produce a receipt
        return [f"source validator refused malformed input: {type(error).__name__}: {error}"], {}, {}


def _canonical_payload_sha256(value: dict[str, Any], *, omit: str) -> str:
    payload = {key: item for key, item in value.items() if key != omit}
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return _sha256_bytes(blob)


def _facet_contract(atlas: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    serialized_atlas = json.dumps(atlas, ensure_ascii=True)
    retired_markers = [marker for marker in RETIRED_TEXT_MARKERS if marker in serialized_atlas]
    if retired_markers:
        problems.append(f"retired current-path markers remain in atlas: {retired_markers}")
    retired_summary = atlas.get("retired_historical_evidence")
    if not isinstance(retired_summary, dict):
        problems.append("retired historical-evidence boundary is absent")
    else:
        expected_retired_fields = {
            "status": "historical-only",
            "current_dependency": False,
            "source_snapshot_member": False,
            "score_credit": False,
        }
        for key, expected in expected_retired_fields.items():
            if retired_summary.get(key) != expected:
                problems.append(f"retired historical-evidence field {key} has drifted")
        if not isinstance(retired_summary.get("statement"), str) or not retired_summary["statement"].strip():
            problems.append("retired historical-evidence statement is absent")
    facets = atlas.get("facets")
    if not isinstance(facets, list):
        return ["facets must be a list"], {}
    ids: list[str] = []
    weights: list[float] = []
    for index, facet in enumerate(facets):
        if not isinstance(facet, dict):
            problems.append(f"facet {index} is not an object")
            continue
        facet_id = facet.get("id")
        if not isinstance(facet_id, str) or not facet_id:
            problems.append(f"facet {index} has no valid id")
        else:
            ids.append(facet_id)
        weight = facet.get("weight")
        if not _finite_number(weight) or float(weight) <= 0:
            problems.append(f"facet {facet_id or index} has invalid weight {weight!r}")
        else:
            weights.append(float(weight))
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        problems.append(f"duplicate facet ids: {duplicates}")
    if len(facets) != EXPECTED_FACET_COUNT:
        problems.append(f"expected {EXPECTED_FACET_COUNT} facets, found {len(facets)}")
    if set(ids) != EXPECTED_FACET_IDS:
        problems.append(
            "facet identity set drift: "
            f"missing={sorted(EXPECTED_FACET_IDS - set(ids))}, "
            f"extra={sorted(set(ids) - EXPECTED_FACET_IDS)}"
        )
    weight_total = sum(weights)
    if not math.isclose(weight_total, EXPECTED_WEIGHT_TOTAL, abs_tol=1e-9):
        problems.append(f"facet weights sum to {weight_total:g}, expected {EXPECTED_WEIGHT_TOTAL}")
    portfolio = atlas.get("portfolio")
    if not isinstance(portfolio, dict):
        problems.append("portfolio must be an object")
    else:
        if portfolio.get("facet_count") != EXPECTED_FACET_COUNT:
            problems.append("portfolio facet_count disagrees with the fixed atlas contract")
        if portfolio.get("domain_weight_total") != EXPECTED_WEIGHT_TOTAL:
            problems.append("portfolio domain_weight_total disagrees with the fixed atlas contract")
    return problems, {"facet_count": len(facets), "facet_weight_total": weight_total, "ids": ids}


def _score_contract(atlas: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    scoring = atlas.get("scoring")
    if not isinstance(scoring, dict):
        problems.append("scoring declaration must be an object")
    else:
        dimensions = scoring.get("dimensions")
        if not isinstance(dimensions, dict):
            problems.append("scoring dimensions must be an object")
        else:
            for dimension, expected_weight in SCORE_WEIGHTS.items():
                declaration = dimensions.get(dimension)
                declared_weight = declaration.get("weight") if isinstance(declaration, dict) else None
                if not _finite_number(declared_weight) or not math.isclose(
                    float(declared_weight), expected_weight, abs_tol=1e-12
                ):
                    problems.append(f"declared {dimension} weight {declared_weight!r} != {expected_weight}")
        if scoring.get("raw_formula") != EXPECTED_RAW_FORMULA:
            problems.append("declared raw score formula has drifted")
        if scoring.get("bottleneck_caps") != list(EXPECTED_BOTTLENECK_CAPS):
            problems.append("declared bottleneck caps have drifted")
        if scoring.get("rounding") != EXPECTED_ROUNDING:
            problems.append("declared score rounding rule has drifted")
    facets = atlas.get("facets")
    if not isinstance(facets, list):
        return ["cannot validate scores without a facet list"], {}
    weighted_total = 0.0
    valid_weight_total = 0.0
    for index, facet in enumerate(facets):
        if not isinstance(facet, dict):
            continue
        facet_id = str(facet.get("id", index))
        scores = facet.get("scores")
        if not isinstance(scores, dict):
            problems.append(f"facet {facet_id} has no score object")
            continue
        values: dict[str, int] = {}
        for dimension in SCORE_DIMENSIONS:
            value = scores.get(dimension)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
                problems.append(f"facet {facet_id} has invalid {dimension} score {value!r}")
            else:
                values[dimension] = value
        if len(values) != len(SCORE_DIMENSIONS):
            continue
        # Preserve the declared expression's evaluation order. This matters at binary-float ties.
        raw = (
            0.20 * values["scaffolding"]
            + 0.25 * values["implementation"]
            + 0.30 * values["experiment"]
            + 0.25 * values["confirmation"]
        )
        stored_raw = scores.get("raw")
        if not _finite_number(stored_raw):
            problems.append(f"facet {facet_id} has invalid raw score {stored_raw!r}")
        elif not math.isclose(float(stored_raw), raw, abs_tol=0.011):
            problems.append(f"facet {facet_id} raw score {stored_raw} does not recompute to {raw:.4f}")
        capped = min(
            raw,
            values["scaffolding"] + 2.5,
            values["implementation"] + 2.0,
            values["experiment"] + 1.5,
            values["confirmation"] + 2.0,
        )
        expected_overall = _round_one_decimal(capped)
        overall = scores.get("overall")
        if not _finite_number(overall):
            problems.append(f"facet {facet_id} has invalid overall score {overall!r}")
            continue
        if not math.isclose(float(overall), expected_overall, abs_tol=0.011):
            problems.append(
                f"facet {facet_id} overall score {overall} does not match capped score {expected_overall}"
            )
        weight = facet.get("weight")
        if _finite_number(weight) and float(weight) > 0:
            weighted_total += float(weight) * float(overall)
            valid_weight_total += float(weight)
    portfolio_score = weighted_total / valid_weight_total if valid_weight_total else None
    portfolio = atlas.get("portfolio")
    if not isinstance(portfolio, dict):
        problems.append("cannot validate portfolio score without portfolio object")
    else:
        stored = portfolio.get("weighted_actionable_realization_score")
        if not _finite_number(stored):
            problems.append("portfolio weighted score is not numeric")
        elif portfolio_score is None:
            problems.append("portfolio weighted score cannot be recomputed")
        elif not math.isclose(float(stored), portfolio_score, abs_tol=0.011):
            problems.append(f"portfolio weighted score {stored} does not recompute to {portfolio_score:.4f}")
        display = portfolio.get("display_score")
        if not _finite_number(display):
            problems.append("portfolio display score is not numeric")
        elif portfolio_score is None:
            problems.append("portfolio display score cannot be recomputed")
        elif not math.isclose(float(display), _round_one_decimal(portfolio_score), abs_tol=0.011):
            problems.append("portfolio display score does not match rounded weighted score")
    return problems, {"weighted_score": portfolio_score, "valid_weight_total": valid_weight_total}


def _domain_contract(atlas: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    domains = atlas.get("domains")
    facets = atlas.get("facets")
    if not isinstance(domains, list) or not isinstance(facets, list):
        return ["domains and facets must both be lists"], {}
    facet_by_id = {
        str(facet.get("id")): facet
        for facet in facets
        if isinstance(facet, dict) and isinstance(facet.get("id"), str)
    }
    domain_ids: list[str] = []
    assigned: list[str] = []
    domain_weight_total = 0.0
    for index, domain in enumerate(domains):
        if not isinstance(domain, dict):
            problems.append(f"domain {index} is not an object")
            continue
        domain_id = domain.get("id")
        if not isinstance(domain_id, str) or not domain_id:
            problems.append(f"domain {index} has no valid id")
            continue
        domain_ids.append(domain_id)
        weight = domain.get("weight")
        if not _finite_number(weight) or float(weight) <= 0:
            problems.append(f"domain {domain_id} has invalid weight {weight!r}")
        else:
            domain_weight_total += float(weight)
        members = domain.get("facet_ids")
        if not isinstance(members, list) or not members:
            problems.append(f"domain {domain_id} has no facet_ids")
            continue
        valid_members = [member for member in members if isinstance(member, str)]
        if len(valid_members) != len(members):
            problems.append(f"domain {domain_id} contains a non-string facet id")
        for member in valid_members:
            if member not in facet_by_id:
                problems.append(f"domain {domain_id} references unknown facet {member!r}")
                continue
            assigned.append(member)
            if facet_by_id[member].get("domain") != domain_id:
                problems.append(f"facet {member} declares a different domain than {domain_id}")
        member_weights: list[float] = []
        invalid_member_weight = False
        for member in valid_members:
            if member not in facet_by_id:
                continue
            member_weight = facet_by_id[member].get("weight")
            if _finite_number(member_weight):
                member_weights.append(float(member_weight))
            else:
                invalid_member_weight = True
        if not invalid_member_weight:
            recomputed_weight = sum(member_weights)
            if _finite_number(weight) and not math.isclose(float(weight), recomputed_weight, abs_tol=1e-9):
                problems.append(
                    f"domain {domain_id} weight {weight} does not match facet weights {recomputed_weight:g}"
                )
        else:
            problems.append(f"domain {domain_id} contains a facet with invalid weight")
    duplicates = sorted(key for key, count in Counter(assigned).items() if count > 1)
    if duplicates:
        problems.append(f"facets assigned to multiple domains: {duplicates}")
    if set(assigned) != set(facet_by_id):
        missing = sorted(set(facet_by_id) - set(assigned))
        extra = sorted(set(assigned) - set(facet_by_id))
        problems.append(f"domain facet partition mismatch: missing={missing}, extra={extra}")
    domain_duplicates = sorted(key for key, count in Counter(domain_ids).items() if count > 1)
    if domain_duplicates:
        problems.append(f"duplicate domain ids: {domain_duplicates}")
    if len(domains) != EXPECTED_DOMAIN_COUNT:
        problems.append(f"expected {EXPECTED_DOMAIN_COUNT} domains, found {len(domains)}")
    if set(domain_ids) != EXPECTED_DOMAIN_IDS:
        problems.append(
            "domain identity set drift: "
            f"missing={sorted(EXPECTED_DOMAIN_IDS - set(domain_ids))}, "
            f"extra={sorted(set(domain_ids) - EXPECTED_DOMAIN_IDS)}"
        )
    if not math.isclose(domain_weight_total, EXPECTED_WEIGHT_TOTAL, abs_tol=1e-9):
        problems.append(f"domain weights sum to {domain_weight_total:g}, expected {EXPECTED_WEIGHT_TOTAL}")
    portfolio = atlas.get("portfolio")
    domain_scores = portfolio.get("domain_scores") if isinstance(portfolio, dict) else None
    if not isinstance(domain_scores, dict):
        problems.append("portfolio domain_scores must be an object")
    else:
        for domain_id in domain_ids:
            rows = [facet for facet in facet_by_id.values() if facet.get("domain") == domain_id]
            values: list[tuple[float, float]] = []
            for facet in rows:
                weight = facet.get("weight")
                scores = facet.get("scores")
                overall = scores.get("overall") if isinstance(scores, dict) else None
                if not _finite_number(weight) or not _finite_number(overall):
                    problems.append(f"domain {domain_id} contains a non-numeric facet score")
                    continue
                values.append((float(weight), float(overall)))
            denominator = sum(weight for weight, _overall in values)
            recomputed = (
                sum(weight * overall for weight, overall in values) / denominator if denominator else None
            )
            stored = domain_scores.get(domain_id)
            if not _finite_number(stored):
                problems.append(f"domain {domain_id} has no numeric portfolio score")
            elif recomputed is None:
                problems.append(f"domain {domain_id} score cannot be recomputed")
            elif not math.isclose(float(stored), recomputed, abs_tol=0.011):
                problems.append(f"domain {domain_id} score {stored} does not recompute to {recomputed:.4f}")
    return problems, {"domain_count": len(domains), "domain_weight_total": domain_weight_total}


def _source_contract(
    atlas: dict[str, Any], repo_root: Path
) -> tuple[list[str], dict[str, Any], dict[str, dict[str, Any]]]:
    problems: list[str] = []
    snapshot = atlas.get("source_snapshot")
    if not isinstance(snapshot, list) or not snapshot:
        return ["source_snapshot must be a non-empty list"], {}, {}
    paths: list[str] = []
    parsed_sources: dict[str, dict[str, Any]] = {}
    verified_count = 0
    for index, row in enumerate(snapshot):
        if not isinstance(row, dict):
            problems.append(f"source snapshot row {index} is not an object")
            continue
        raw_path = row.get("path")
        digest = row.get("sha256")
        if isinstance(raw_path, str):
            paths.append(raw_path)
        path, path_problem = _repo_file(repo_root, raw_path)
        if path_problem:
            problems.append(f"source row {index}: {path_problem}")
            continue
        assert path is not None
        if not _valid_sha256(digest):
            problems.append(f"source {raw_path} has invalid sha256")
            continue
        if not path.is_file():
            problems.append(f"source {raw_path} does not exist as a file")
            continue
        actual = _sha256(path)
        if actual != digest:
            problems.append(f"source hash drift for {raw_path}: expected {digest}, found {actual}")
            continue
        verified_count += 1
        if path.suffix == ".json":
            try:
                parsed = _load_json(path)
            except (OSError, json.JSONDecodeError) as error:
                problems.append(f"source {raw_path} is not valid JSON: {error}")
                continue
            if isinstance(parsed, dict):
                parsed_sources[str(raw_path)] = parsed
    duplicates = sorted(key for key, count in Counter(paths).items() if count > 1)
    if duplicates:
        problems.append(f"duplicate source snapshot paths: {duplicates}")
    missing_required = sorted(set(REQUIRED_BOUND_SOURCES) - set(paths))
    if missing_required:
        problems.append(f"required bound sources missing from source_snapshot: {missing_required}")
    retired_sources = sorted(set(paths) & RETIRED_CURRENT_PATHS)
    if retired_sources:
        problems.append(f"retired sources remain in source_snapshot: {retired_sources}")
    return problems, {"source_count": len(snapshot), "verified_count": verified_count}, parsed_sources


def _evidence_contract(atlas: dict[str, Any], repo_root: Path) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    facets = atlas.get("facets")
    if not isinstance(facets, list):
        return ["cannot validate evidence without facets"], {}
    references = 0
    for index, facet in enumerate(facets):
        if not isinstance(facet, dict):
            continue
        facet_id = str(facet.get("id", index))
        evidence = facet.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            problems.append(f"facet {facet_id} has no evidence paths")
            continue
        seen: set[str] = set()
        for raw in evidence:
            references += 1
            if not isinstance(raw, str):
                problems.append(f"facet {facet_id} evidence path is not a string")
                continue
            if raw in seen:
                problems.append(f"facet {facet_id} repeats evidence path {raw}")
            seen.add(raw)
            if raw in RETIRED_CURRENT_PATHS:
                problems.append(f"facet {facet_id} cites retired current-path evidence: {raw}")
            path, path_problem = _repo_file(repo_root, raw)
            if path_problem:
                problems.append(f"facet {facet_id}: {path_problem}")
            elif path is None or not path.is_file():
                problems.append(f"facet {facet_id} evidence path does not exist as a file: {raw}")
    return problems, {"evidence_reference_count": references}


def _category2_contract(
    atlas: dict[str, Any], requirements: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    if requirements.get("schema") != REQUIREMENTS_SCHEMA:
        problems.append(f"unexpected requirements schema {requirements.get('schema')!r}")
    stored_payload_sha256 = requirements.get("payload_sha256")
    if not _valid_sha256(stored_payload_sha256):
        problems.append("requirements payload_sha256 is absent or invalid")
    else:
        try:
            recomputed_payload_sha256 = _canonical_payload_sha256(requirements, omit="payload_sha256")
        except (TypeError, ValueError) as error:
            problems.append(f"requirements payload cannot be canonicalized: {error}")
        else:
            if stored_payload_sha256 != recomputed_payload_sha256:
                problems.append("requirements payload_sha256 does not match its canonical content")
    rows = requirements.get("rows")
    if not isinstance(rows, list):
        return problems + ["requirements rows must be a list"], {}
    if len(rows) != EXPECTED_REQUIREMENTS_ROW_COUNT:
        problems.append(f"requirements contain {len(rows)} rows, expected {EXPECTED_REQUIREMENTS_ROW_COUNT}")
    primary_category_counts = Counter(row.get("primary_category") for row in rows if isinstance(row, dict))
    if dict(primary_category_counts) != EXPECTED_PRIMARY_CATEGORY_COUNTS:
        problems.append(
            "requirements primary-category counts drift: "
            f"expected {EXPECTED_PRIMARY_CATEGORY_COUNTS}, found {dict(primary_category_counts)}"
        )
    source_rows = [row for row in rows if isinstance(row, dict) and row.get("primary_category") == 2]
    rows_by_id = {
        row.get("id"): row for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    expected_transition_categories = {
        "e6_relational": 3,
        "mop_dr14_corruption": 3,
        "mop_al2_shared_latent_alignment": 3,
        "mop_dr5_cross_substrate_consistency": 2,
        "frontier_continual_million_event_learning": 2,
        "frontier_action_conditioned_world_models": 3,
        "mop_cm10_action_forward_model": 3,
        "frontier_workspace_operational_self_model": 3,
        "frontier_full_system_density_accounting": 3,
    }
    for row_id, expected_category in expected_transition_categories.items():
        row = rows_by_id.get(row_id)
        if not isinstance(row, dict) or row.get("primary_category") != expected_category:
            problems.append(f"requirements transition row {row_id} is not category {expected_category}")
    p6_row = rows_by_id.get("frontier_continual_million_event_learning")
    p6_evidence = p6_row.get("evidence_refs") if isinstance(p6_row, dict) else None
    if not isinstance(p6_evidence, list) or (
        "local:proof/LOCAL_EXECUTION_THROTTLE_P6_10K_DRY_RUN.json" not in p6_evidence
    ):
        problems.append("P6 requirements row is not bound to its 10k scheduler dry-run")
    retired_hg_markers = (
        "VJEPA_SCALE_ATLAS_LOCAL",
        "FACTORIZED_STIMULUS_IDENTITY",
        "REAL_ENCODER_VITH_LOCAL8",
        "REAL_ENCODER_VITG_LOCAL8",
        "ViT-H",
        "ViT-G",
        "L/H/g",
    )
    for row_id in ("mop_al2_shared_latent_alignment", "mop_dr5_cross_substrate_consistency"):
        row = rows_by_id.get(row_id)
        serialized_row = json.dumps(row, ensure_ascii=True) if isinstance(row, dict) else ""
        if any(marker in serialized_row for marker in retired_hg_markers):
            problems.append(f"requirements row {row_id} retains a retired H/g dependency")
    source_ids = [row.get("id") for row in source_rows]
    if any(not isinstance(value, str) or not value for value in source_ids):
        problems.append("category-2 source contains a row without a valid id")
    source_id_strings = [str(value) for value in source_ids]
    source_duplicates = sorted(key for key, count in Counter(source_id_strings).items() if count > 1)
    if source_duplicates:
        problems.append(f"duplicate category-2 source ids: {source_duplicates}")
    cluster_block = atlas.get("category2_harness_clusters")
    if not isinstance(cluster_block, dict):
        return problems + ["category2_harness_clusters must be an object"], {}
    if cluster_block.get("category2_row_count") != EXPECTED_CATEGORY2_COUNT:
        problems.append(f"atlas category2_row_count is not the fixed {EXPECTED_CATEGORY2_COUNT}-row snapshot")
    if cluster_block.get("partition_exactly_once") is not True:
        problems.append("atlas does not declare an exact category-2 partition")
    clusters = cluster_block.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        return problems + ["category-2 clusters must be a non-empty list"], {}
    cluster_ids: list[str] = []
    members: list[str] = []
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            problems.append(f"category-2 cluster {index} is not an object")
            continue
        cluster_id = cluster.get("id")
        if not isinstance(cluster_id, str) or not cluster_id:
            problems.append(f"category-2 cluster {index} has no valid id")
        else:
            cluster_ids.append(cluster_id)
        cluster_members = cluster.get("members")
        if not isinstance(cluster_members, list) or not all(
            isinstance(value, str) and value for value in cluster_members
        ):
            problems.append(f"category-2 cluster {cluster_id or index} has invalid members")
            continue
        if cluster.get("count") != len(cluster_members):
            problems.append(f"category-2 cluster {cluster_id or index} count does not match members")
        members.extend(cluster_members)
    duplicate_clusters = sorted(key for key, count in Counter(cluster_ids).items() if count > 1)
    if duplicate_clusters:
        problems.append(f"duplicate category-2 cluster ids: {duplicate_clusters}")
    if set(cluster_ids) != EXPECTED_CLUSTER_IDS:
        problems.append(
            "category-2 cluster identity set drift: "
            f"missing={sorted(EXPECTED_CLUSTER_IDS - set(cluster_ids))}, "
            f"extra={sorted(set(cluster_ids) - EXPECTED_CLUSTER_IDS)}"
        )
    duplicate_members = sorted(key for key, count in Counter(members).items() if count > 1)
    if duplicate_members:
        problems.append(f"category-2 rows assigned more than once: {duplicate_members}")
    if len(source_id_strings) != EXPECTED_CATEGORY2_COUNT:
        problems.append(
            f"requirements contain {len(source_id_strings)} category-2 rows, "
            f"expected {EXPECTED_CATEGORY2_COUNT}"
        )
    if len(members) != EXPECTED_CATEGORY2_COUNT:
        problems.append(f"clusters contain {len(members)} members, expected {EXPECTED_CATEGORY2_COUNT}")
    if set(members) != set(source_id_strings):
        missing = sorted(set(source_id_strings) - set(members))
        extra = sorted(set(members) - set(source_id_strings))
        problems.append(f"category-2 member set drift: missing={missing}, extra={extra}")
    scope_counts = Counter(str(row.get("scope")) for row in source_rows)
    declared_scope_counts = cluster_block.get("scope_counts")
    if not isinstance(declared_scope_counts, dict):
        problems.append("category-2 scope_counts must be an object")
    elif dict(scope_counts) != declared_scope_counts:
        problems.append(
            f"category-2 scope counts drift: expected {dict(scope_counts)}, found {declared_scope_counts}"
        )
    reclassified = cluster_block.get("reclassified_after_local_integration")
    if not isinstance(reclassified, dict):
        problems.append("category-2 reclassification receipt is absent")
    else:
        if reclassified.get("from_category") != 2 or reclassified.get("to_category") != 3:
            problems.append("E6/DR14 category transition is not declared as 2 to 3")
        if set(reclassified.get("ids") or []) != {"e6_relational", "mop_dr14_corruption"}:
            problems.append("E6/DR14 reclassification ids have drifted")
    p7_reclassified = cluster_block.get("p7_reclassified_after_local_integration")
    if not isinstance(p7_reclassified, dict):
        problems.append("P7 action-world-model reclassification receipt is absent")
    else:
        if p7_reclassified.get("from_category") != 2 or p7_reclassified.get("to_category") != 3:
            problems.append("P7 action-world-model category transition is not declared as 2 to 3")
        if set(p7_reclassified.get("ids") or []) != {
            "frontier_action_conditioned_world_models",
            "mop_cm10_action_forward_model",
        }:
            problems.append("P7 action-world-model reclassification ids have drifted")
    p9_reclassified = cluster_block.get("p9_reclassified_after_local_integration")
    if not isinstance(p9_reclassified, dict):
        problems.append("P9 monitoring/accounting reclassification receipt is absent")
    else:
        if p9_reclassified.get("from_category") != 2 or p9_reclassified.get("to_category") != 3:
            problems.append("P9 monitoring/accounting category transition is not declared as 2 to 3")
        if set(p9_reclassified.get("ids") or []) != {
            "frontier_workspace_operational_self_model",
            "frontier_full_system_density_accounting",
        }:
            problems.append("P9 monitoring/accounting reclassification ids have drifted")
    return problems, {
        "requirements_row_count": len(rows),
        "primary_category_counts": dict(primary_category_counts),
        "category2_source_count": len(source_id_strings),
        "category2_member_count": len(members),
        "cluster_count": len(clusters),
        "scope_counts": dict(scope_counts),
    }


def _snapshot_accounting_contract(
    atlas: dict[str, Any],
    requirements: dict[str, Any],
    parsed_sources: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    portfolio = atlas.get("portfolio")
    if not isinstance(portfolio, dict):
        return ["portfolio is unavailable for snapshot accounting"], {}

    project = parsed_sources.get("proof/PROJECT_EXPERIMENT_EXHAUSTION.json")
    project_coverage = project.get("coverage") if isinstance(project, dict) else None
    project_entries = project.get("entries") if isinstance(project, dict) else None
    if not isinstance(project, dict) or project.get("schema") != "mop-project-experiment-exhaustion/v1":
        problems.append("verified project-exhaustion source is unavailable or has the wrong schema")
    if not isinstance(project_coverage, dict):
        problems.append("project-exhaustion coverage is unavailable")
    else:
        if project_coverage.get("accounted_exactly_once") is not True:
            problems.append("project-exhaustion rows are not accounted exactly once")
        if project_coverage.get("registry_non_f_total") != 177:
            problems.append("project-exhaustion non-F total is not 177")
        if project_coverage.get("classification_counts") != EXPECTED_EXHAUSTION_COUNTS:
            problems.append("project-exhaustion classification counts have drifted")
    if not isinstance(project_entries, list):
        problems.append("project-exhaustion entries are unavailable")
        scientific_ready_count = None
    else:
        scientific_ready_count = sum(
            1
            for row in project_entries
            if isinstance(row, dict) and row.get("scientific_claim_ready") is True
        )
        if scientific_ready_count != 0:
            problems.append("project-exhaustion contains a scientific-claim-ready row")

    frontier = parsed_sources.get("proof/FRONTIER_LOCALIZATION.json")
    frontier_coverage = frontier.get("coverage") if isinstance(frontier, dict) else None
    frontier_entries = frontier.get("entries") if isinstance(frontier, dict) else None
    expected_localization_counts = {
        "external-input-blocked": 16,
        "local-custom-preflight-proven-upstream-blocked": 1,
        "local-custom-training-five-seed-null-bound": 1,
        "local-dense-task-integration-proven-data-blocked": 1,
        "local-mechanics-proven": 5,
    }
    if not isinstance(frontier, dict) or frontier.get("schema") != "mop-frontier-localization/v1":
        problems.append("verified frontier-localization source is unavailable or has the wrong schema")
    if not isinstance(frontier_coverage, dict):
        problems.append("frontier-localization coverage is unavailable")
    else:
        if frontier_coverage.get("localization_counts") != expected_localization_counts:
            problems.append("frontier-localization counts have drifted")
        if frontier_coverage.get("measured_hardware_blocked_count") != 0:
            problems.append("frontier-localization reports a measured hardware blocker")
        if frontier_coverage.get("historical_frontier_count") != 24:
            problems.append("frontier-localization historical row count is not 24")
    if not isinstance(frontier_entries, list):
        problems.append("frontier-localization entries are unavailable")
    else:
        frontier_by_id = {
            row.get("id"): row
            for row in frontier_entries
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        }
        expected_frontier_states = {
            "mop_al2_shared_latent_alignment": ("external-input-blocked", "rights-data-blocked"),
            "mop_dr5_cross_substrate_consistency": (
                "external-input-blocked",
                "upstream-model-blocked",
            ),
            "mop_dr14_corruption": (
                "local-dense-task-integration-proven-data-blocked",
                "rights-data-blocked",
            ),
        }
        for row_id, (localization, exhaustion) in expected_frontier_states.items():
            row = frontier_by_id.get(row_id)
            if not isinstance(row, dict):
                problems.append(f"frontier-localization row {row_id} is absent")
                continue
            if row.get("localization") != localization:
                problems.append(f"frontier-localization row {row_id} has stale localization")
            if row.get("project_exhaustion_classification") != exhaustion:
                problems.append(f"frontier-localization row {row_id} disagrees with project exhaustion")
            if "scale_atlas_receipt" in row:
                problems.append(f"frontier-localization row {row_id} retains a retired dependency field")
        serialized_frontier = json.dumps(frontier, ensure_ascii=True)
        if any(marker in serialized_frontier for marker in RETIRED_FRONTIER_MARKERS):
            problems.append("frontier-localization retains a retired current-path marker")

    current_summary = portfolio.get("current_registry_summary")
    expected_current_summary = {
        "non_f_rows": 177,
        "freshly_executed_verified": 117,
        "already_durable_hash_verifiable": 37,
        "implementation_blocked": 1,
        "rights_data_blocked": 11,
        "upstream_model_blocked": 8,
        "runnable_not_yet_run": 3,
        "measured_hardware_blocked": 0,
        "scientific_claim_ready": 0,
    }
    if not isinstance(current_summary, dict):
        problems.append("atlas current_registry_summary is absent")
    else:
        for key, expected in expected_current_summary.items():
            if current_summary.get(key) != expected:
                problems.append(f"atlas current-registry field {key} has drifted")

    rows = requirements.get("rows")
    requirement_summary = portfolio.get("requirements_summary")
    primary_counts = (
        Counter(row.get("primary_category") for row in rows if isinstance(row, dict))
        if isinstance(rows, list)
        else Counter()
    )
    category2_current_registry = (
        sum(
            1
            for row in rows
            if isinstance(row, dict)
            and row.get("primary_category") == 2
            and row.get("scope") == "current_registry"
        )
        if isinstance(rows, list)
        else None
    )
    expected_requirement_summary = {
        "row_count": EXPECTED_REQUIREMENTS_ROW_COUNT,
        "category_counts": {
            "1": 168,
            "2": 119,
            "3": 29,
            "6": 5,
            "8": 0,
            "9": 0,
        },
        "category2_current_registry_rows": 39,
        "measured_hardware_rows": 0,
    }
    if requirement_summary != expected_requirement_summary:
        problems.append("atlas requirements_summary has drifted")
    if dict(primary_counts) != EXPECTED_PRIMARY_CATEGORY_COUNTS:
        problems.append("requirements counts disagree with atlas accounting contract")
    if category2_current_registry != 39:
        problems.append("requirements current-registry category-2 count is not 39")
    return problems, {
        "registry_non_f_rows": 177 if isinstance(project_coverage, dict) else None,
        "scientific_claim_ready_rows": scientific_ready_count,
        "category2_current_registry_rows": category2_current_registry,
        "frontier_historical_rows": (
            frontier_coverage.get("historical_frontier_count")
            if isinstance(frontier_coverage, dict)
            else None
        ),
    }


def _queue_contract(atlas: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    facets = atlas.get("facets")
    facet_ids = {
        str(row.get("id")) for row in facets or [] if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    queue = atlas.get("highest_leverage_local_queue")
    if not isinstance(queue, list) or not queue:
        return ["highest_leverage_local_queue must be a non-empty list"], {}
    ranks: list[int] = []
    queue_ids: list[str] = []
    for index, row in enumerate(queue):
        if not isinstance(row, dict):
            problems.append(f"queue row {index} is not an object")
            continue
        rank = row.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            problems.append(f"queue row {index} has invalid rank {rank!r}")
        else:
            ranks.append(rank)
        queue_id = row.get("id")
        if not isinstance(queue_id, str) or not queue_id:
            problems.append(f"queue row {index} has no valid id")
        else:
            queue_ids.append(queue_id)
        refs = row.get("facets")
        if not isinstance(refs, list) or not refs:
            problems.append(f"queue row {queue_id or index} has no facet references")
        else:
            unknown = sorted({str(ref) for ref in refs if ref not in facet_ids})
            if unknown:
                problems.append(f"queue row {queue_id or index} references unknown facets {unknown}")
        if not isinstance(row.get("work"), str) or not row["work"].strip():
            problems.append(f"queue row {queue_id or index} has no work statement")
        if not isinstance(row.get("exit_receipt"), str) or not row["exit_receipt"].strip():
            problems.append(f"queue row {queue_id or index} has no exit receipt")
    if ranks != list(range(1, len(queue) + 1)):
        problems.append(f"queue ranks are not contiguous and ordered: {ranks}")
    duplicate_ids = sorted(key for key, count in Counter(queue_ids).items() if count > 1)
    if duplicate_ids:
        problems.append(f"duplicate queue ids: {duplicate_ids}")
    if set(queue_ids) != EXPECTED_QUEUE_IDS:
        problems.append(
            "queue identity set drift: "
            f"missing={sorted(EXPECTED_QUEUE_IDS - set(queue_ids))}, "
            f"extra={sorted(set(queue_ids) - EXPECTED_QUEUE_IDS)}"
        )
    return problems, {"queue_count": len(queue), "ranks": ranks}


def _dependency_contract(
    atlas: dict[str, Any], requirements: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    facets = atlas.get("facets")
    facet_ids = {
        str(row.get("id")) for row in facets or [] if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    dependency_count = 0
    for row in facets or []:
        if not isinstance(row, dict):
            continue
        facet_id = str(row.get("id"))
        dependencies = row.get("dependencies")
        if not isinstance(dependencies, list):
            problems.append(f"facet {facet_id} dependencies must be a list")
            continue
        dependency_count += len(dependencies)
        valid_dependencies = [value for value in dependencies if isinstance(value, str)]
        if len(valid_dependencies) != len(dependencies):
            problems.append(f"facet {facet_id} has a non-string dependency id")
        unknown = sorted({value for value in valid_dependencies if value not in facet_ids})
        if unknown:
            problems.append(f"facet {facet_id} references unknown dependencies {unknown}")
        if facet_id in valid_dependencies:
            problems.append(f"facet {facet_id} depends on itself")
        if len(valid_dependencies) != len(set(valid_dependencies)):
            problems.append(f"facet {facet_id} repeats a dependency")
    graph = atlas.get("dependency_graph")
    if not isinstance(graph, dict):
        problems.append("dependency_graph must be an object")
    else:
        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list) or not all(isinstance(node, str) and node for node in nodes):
            problems.append("dependency_graph nodes are invalid")
        elif len(nodes) != len(set(nodes)):
            problems.append("dependency_graph nodes are duplicated")
        elif set(nodes) != EXPECTED_GRAPH_NODES:
            problems.append(
                "dependency_graph node identity set drift: "
                f"missing={sorted(EXPECTED_GRAPH_NODES - set(nodes))}, "
                f"extra={sorted(set(nodes) - EXPECTED_GRAPH_NODES)}"
            )
        if not isinstance(edges, list):
            problems.append("dependency_graph edges must be a list")
        elif isinstance(nodes, list):
            node_set = set(nodes)
            for index, edge in enumerate(edges):
                if not isinstance(edge, list) or len(edge) != 2:
                    problems.append(f"dependency_graph edge {index} is not a pair")
                elif edge[0] not in node_set or edge[1] not in node_set:
                    problems.append(f"dependency_graph edge {index} references an unknown node")
        if graph.get("studio_is_not_a_prerequisite_node") is not True:
            problems.append("dependency_graph no longer excludes Studio as a prerequisite node")
    source_ids = {
        str(row.get("id"))
        for row in requirements.get("rows") or []
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    wave = atlas.get("smallest_reusable_local_wave")
    if not isinstance(wave, dict):
        problems.append("smallest_reusable_local_wave must be an object")
    else:
        sentinels = wave.get("sentinel_experiments")
        if not isinstance(sentinels, list) or not sentinels:
            problems.append("smallest local wave has no sentinel experiments")
        else:
            for index, sentinel in enumerate(sentinels):
                sentinel_id = sentinel.get("id") if isinstance(sentinel, dict) else None
                if sentinel_id not in source_ids:
                    problems.append(f"wave sentinel {index} is absent from requirements: {sentinel_id!r}")
    return problems, {"facet_dependency_count": dependency_count}


def _p6_missing_verifier_refusal(decision: dict[str, Any]) -> bool:
    gates = [gate for gate in decision.get("gates") or [] if isinstance(gate, dict)]
    prerequisite_gates = [gate for gate in gates if gate.get("name") == "receipt_prerequisites"]
    if len(prerequisite_gates) != 1:
        return False
    prerequisite_gate = prerequisite_gates[0]
    observed = prerequisite_gate.get("observed")
    row = observed[0] if isinstance(observed, list) and len(observed) == 1 else None
    denied_reasons = decision.get("denied_reasons")
    failing_reasons = {
        gate.get("reason")
        for gate in gates
        if gate.get("ok") is False and isinstance(gate.get("reason"), str)
    }
    required_green_gates = {"exclusive_lane", "resource_measurement", "forecasted_disk"}
    green_names = {gate.get("name") for gate in gates if gate.get("ok") is True}
    return bool(
        decision.get("allowed") is False
        and decision.get("active_lanes") == []
        and isinstance(denied_reasons, list)
        and P6_PREREQUISITE_REASON in denied_reasons
        and len(denied_reasons) == len(set(denied_reasons))
        and set(denied_reasons) == failing_reasons
        and prerequisite_gate.get("ok") is False
        and prerequisite_gate.get("reason") == P6_PREREQUISITE_REASON
        and isinstance(row, dict)
        and row.get("path") == P5_VERIFICATION_PATH
        and row.get("all_ok") is False
        and row.get("schema") is None
        and row.get("sha256") is None
        and row.get("governor_provenance") is None
        and "receipt is missing" in (row.get("problems") or [])
        and required_green_gates <= green_names
    )


def _mechanics_progress_contract(
    atlas: dict[str, Any], parsed_sources: dict[str, dict[str, Any]], repo_root: Path
) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    dense_source = parsed_sources.get("proof/E6_VITB_DENSE_PREFLIGHT.json")
    if not isinstance(dense_source, dict):
        problems.append("verified E6/DR14 dense-task preflight is unavailable")
    else:
        dense_gates = dense_source.get("gates")
        interfaces = dense_source.get("interfaces")
        if dense_source.get("schema") != "mop-vjepa21-dense-task-preflight/v1":
            problems.append("E6/DR14 dense-task preflight schema has drifted")
        if dense_source.get("all_ok") is not True:
            problems.append("E6/DR14 dense-task preflight is not all_ok")
        if not isinstance(dense_gates, dict) or dense_gates.get("implementation_ready") is not True:
            problems.append("E6/DR14 implementation-ready gate is not closed")
        if not isinstance(dense_gates, dict) or dense_gates.get("input_manifest_ready") is not False:
            problems.append("E6/DR14 input-manifest gap is not represented as open")
        if (
            not isinstance(interfaces, dict)
            or not interfaces
            or not all(value is True for value in interfaces.values())
        ):
            problems.append("E6/DR14 task-cache-control interfaces are incomplete")
        if dense_source.get("scientific_promotion") is not False:
            problems.append("E6/DR14 mechanics preflight claims scientific promotion")
        if dense_source.get("model_constructed") is not False:
            problems.append("E6/DR14 no-heavy preflight unexpectedly constructed a model")
    dense_atlas = atlas.get("dense_task_integration")
    if not isinstance(dense_atlas, dict):
        problems.append("atlas dense_task_integration summary is absent")
    else:
        expected_dense_fields = {
            "status": "mechanics-pass",
            "receipt": "proof/E6_VITB_DENSE_PREFLIGHT.json",
            "task_cache_control_integration_verified": True,
            "heavy_model_work_executed": False,
            "input_manifest_ready": False,
            "scientific_promotion": False,
        }
        for key, expected in expected_dense_fields.items():
            if dense_atlas.get(key) != expected:
                problems.append(f"atlas dense-task field {key} has drifted")
        remaining = dense_atlas.get("remaining")
        if not isinstance(remaining, list) or len(remaining) != 3:
            problems.append("atlas dense-task natural-data/verifier remainder is incomplete")

    wave_source = parsed_sources.get("proof/EXPANSION_WAVE0.json")
    if not isinstance(wave_source, dict):
        problems.append("verified Wave E0 receipt is unavailable")
    else:
        verifier = wave_source.get("independent_verifier")
        if wave_source.get("schema") != "mop-expansion-wave0/v1":
            problems.append("Wave E0 receipt schema has drifted")
        if wave_source.get("status") != "mechanics-pass":
            problems.append("Wave E0 receipt is not mechanics-pass")
        if wave_source.get("all_sentinels_pass") is not True:
            problems.append("Wave E0 sentinels do not all pass")
        if not isinstance(verifier, dict) or verifier.get("verified") is not True:
            problems.append("Wave E0 independent verifier is not closed")
        if not isinstance(verifier, dict) or verifier.get("all_mutations_rejected") is not True:
            problems.append("Wave E0 mutation suite is not closed")
    wave_atlas = atlas.get("smallest_reusable_local_wave")
    if not isinstance(wave_atlas, dict):
        problems.append("atlas Wave E0 summary is absent")
    else:
        if wave_atlas.get("status") != "mechanics-pass":
            problems.append("atlas Wave E0 status is not mechanics-pass")
        if wave_atlas.get("receipt") != "proof/EXPANSION_WAVE0.json":
            problems.append("atlas Wave E0 receipt binding has drifted")
        if wave_atlas.get("acceptance_satisfied") is not True:
            problems.append("atlas Wave E0 acceptance is not closed")
        verification = wave_atlas.get("independent_verification")
        expected_verification = {
            "verified": True,
            "sentinel_count": 3,
            "unit_count": 3,
            "recomputed_metric_count": 72,
            "mutations_rejected": 9,
        }
        if verification != expected_verification:
            problems.append("atlas Wave E0 verification summary has drifted")

    p6_source = parsed_sources.get("proof/CONTINUAL_MILLION_EVENT_PREFLIGHT.json")
    if not isinstance(p6_source, dict):
        problems.append("verified P6 continual-stream preflight is unavailable")
    else:
        checks = p6_source.get("checks")
        full_gate = p6_source.get("remaining_full_run_gate")
        envelope = p6_source.get("resource_envelope")
        if p6_source.get("schema") != "mop-continual-million-event-preflight/v1":
            problems.append("P6 preflight schema has drifted")
        if p6_source.get("status") != "mechanics-pass":
            problems.append("P6 preflight is not mechanics-pass")
        if p6_source.get("all_mechanics_ok") is not True:
            problems.append("P6 mechanics are not all closed")
        if p6_source.get("no_heavy_preflight") is not True:
            problems.append("P6 receipt is not a no-heavy preflight")
        if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
            problems.append("P6 mechanics check family is incomplete")
        if not isinstance(envelope, dict) or envelope.get("configured_stream_events") != 384:
            problems.append("P6 preflight is not bound to the 384-event rung")
        if not isinstance(full_gate, dict):
            problems.append("P6 remaining full-run gate is absent")
        else:
            if full_gate.get("progressive_rungs") != [10000, 100000, 1000000]:
                problems.append("P6 progressive execution ladder has drifted")
            if full_gate.get("minimum_independent_seeds") != 5:
                problems.append("P6 full-run seed requirement has drifted")
            if full_gate.get("hardware_boundary_earned") is not False:
                problems.append("P6 preflight claims a hardware boundary")
    p6_atlas = atlas.get("continual_million_event_preflight")
    if not isinstance(p6_atlas, dict):
        problems.append("atlas P6 continual-stream summary is absent")
    else:
        expected_p6_fields = {
            "status": "mechanics-pass",
            "receipt": "proof/CONTINUAL_MILLION_EVENT_PREFLIGHT.json",
            "current_events_per_stream": 384,
            "current_total_arm_events": 2304,
            "exact_atomic_resume": True,
            "no_heavy_preflight": True,
            "scientific_promotion": False,
            "progressive_rungs": [10000, 100000, 1000000],
            "minimum_independent_seeds": 5,
        }
        for key, expected in expected_p6_fields.items():
            if p6_atlas.get(key) != expected:
                problems.append(f"atlas P6 field {key} has drifted")
        scheduler_atlas = p6_atlas.get("scheduler_preflight")
        expected_scheduler_fields = {
            "receipt": "proof/LOCAL_EXECUTION_THROTTLE_P6_10K_DRY_RUN.json",
            "task_id": "p6_10k_resource_probe_cpu",
            "requires_empty_lanes": True,
            "admission_allowed": False,
            "command_executed": False,
        }
        if not isinstance(scheduler_atlas, dict):
            problems.append("atlas P6 scheduler preflight is absent")
        else:
            for key, expected in expected_scheduler_fields.items():
                if scheduler_atlas.get(key) != expected:
                    problems.append(f"atlas P6 scheduler field {key} has drifted")
    p6_scheduler = parsed_sources.get("proof/LOCAL_EXECUTION_THROTTLE_P6_10K_DRY_RUN.json")
    if not isinstance(p6_scheduler, dict):
        problems.append("verified P6 10k scheduler dry-run is unavailable")
    else:
        task = p6_scheduler.get("task")
        admission = p6_scheduler.get("admission")
        decisions = p6_scheduler.get("decisions")
        if p6_scheduler.get("schema") != "mop-local-throttle-receipt/v1":
            problems.append("P6 10k scheduler receipt schema has drifted")
        if p6_scheduler.get("mode") not in {"dry-run", "run-dry-run"}:
            problems.append("P6 10k scheduler receipt is not a dry-run")
        if p6_scheduler.get("command_executed") is not False:
            problems.append("P6 10k scheduler dry-run claims command execution")
        if not isinstance(task, dict) or task.get("task_id") != "p6_10k_resource_probe_cpu":
            problems.append("P6 10k scheduler task identity has drifted")
        if not isinstance(task, dict) or task.get("requires_empty_lanes") is not True:
            problems.append("P6 10k resource probe is not exclusive")
        prerequisites = task.get("prerequisites") if isinstance(task, dict) else None
        if (
            not isinstance(prerequisites, list)
            or len(prerequisites) != 1
            or not isinstance(prerequisites[0], dict)
            or prerequisites[0].get("path") != P5_VERIFICATION_PATH
            or prerequisites[0].get("schema") != P5_VERIFICATION_SCHEMA
        ):
            problems.append("P6 10k scheduler P5 prerequisite has drifted")
        if not isinstance(admission, dict) or admission.get("allowed") is not False:
            problems.append("P6 10k scheduler receipt does not fail closed before P5 verification")
        if (
            not isinstance(decisions, list)
            or len(decisions) != 3
            or any(
                not isinstance(decision, dict) or not _p6_missing_verifier_refusal(decision)
                for decision in decisions
            )
        ):
            problems.append("P6 10k scheduler missing-P5 refusal decisions have drifted")

    p7_source = parsed_sources.get("proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json")
    if not isinstance(p7_source, dict):
        problems.append("verified P7 action/world-model preflight is unavailable")
    else:
        p7_checks = p7_source.get("checks")
        p7_boundary = p7_source.get("claim_boundary")
        p7_units = p7_source.get("units")
        if p7_source.get("schema") != "mop-p7-action-world-model-preflight/v1":
            problems.append("P7 action/world-model schema has drifted")
        if p7_source.get("status") != "mechanics-pass":
            problems.append("P7 action/world-model preflight is not mechanics-pass")
        if p7_source.get("all_mechanics_ok") is not True:
            problems.append("P7 action/world-model mechanics are not all closed")
        if (
            not isinstance(p7_checks, dict)
            or not p7_checks
            or not all(value is True for value in p7_checks.values())
        ):
            problems.append("P7 action/world-model check family is incomplete")
        if not isinstance(p7_boundary, dict) or p7_boundary.get("mechanics_only") is not True:
            problems.append("P7 action/world-model boundary is not mechanics-only")
        if not isinstance(p7_boundary, dict) or p7_boundary.get("scientific_promotion_allowed") is not False:
            problems.append("P7 action/world-model preflight claims scientific promotion")
        if not isinstance(p7_units, list) or len(p7_units) != 3:
            problems.append("P7 action/world-model unit count has drifted")
        else:
            for index, unit in enumerate(p7_units):
                arms = unit.get("arms") if isinstance(unit, dict) else None
                mutation_suite = unit.get("mutation_suite") if isinstance(unit, dict) else None
                equal_core = unit.get("equal_core_compute") if isinstance(unit, dict) else None
                if not isinstance(unit, dict) or unit.get("all_mechanics_ok") is not True:
                    problems.append(f"P7 unit {index} is not mechanics-pass")
                if not isinstance(arms, dict) or len(arms) != 8:
                    problems.append(f"P7 unit {index} does not execute eight arms")
                if not isinstance(mutation_suite, dict) or mutation_suite.get("all_rejected") is not True:
                    problems.append(f"P7 unit {index} mutation suite is incomplete")
                if not isinstance(equal_core, dict) or equal_core.get("matched") is not True:
                    problems.append(f"P7 unit {index} equal-core contract is not matched")
    p7_atlas = atlas.get("action_world_model_preflight")
    if not isinstance(p7_atlas, dict):
        problems.append("atlas P7 action/world-model summary is absent")
    else:
        expected_p7_fields = {
            "status": "mechanics-pass",
            "receipt": "proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json",
            "independent_units": 3,
            "arm_count": 8,
            "all_replay_mutations_rejected": True,
            "equal_core_compute_verified": True,
            "scientific_promotion": False,
        }
        for key, expected in expected_p7_fields.items():
            if p7_atlas.get(key) != expected:
                problems.append(f"atlas P7 field {key} has drifted")

    p9_source = parsed_sources.get("proof/P9_CAUSAL_MONITORING_PREFLIGHT.json")
    if not isinstance(p9_source, dict):
        problems.append("verified P9 causal-monitoring preflight is unavailable")
    else:
        p9_checks = p9_source.get("checks")
        p9_boundary = p9_source.get("claim_boundary")
        p9_dataset = p9_source.get("dataset")
        p9_mutations = p9_source.get("mutation_suite")
        p9_resume = p9_source.get("resume")
        p9_units = p9_source.get("units")
        p9_aggregate = p9_source.get("causal_vs_correlational_aggregate")
        if p9_source.get("schema") != "mop-p9-causal-monitoring-preflight/v1":
            problems.append("P9 causal-monitoring schema has drifted")
        if p9_source.get("status") != "mechanics-pass":
            problems.append("P9 causal-monitoring preflight is not mechanics-pass")
        if p9_source.get("all_mechanics_ok") is not True:
            problems.append("P9 causal-monitoring mechanics are not all closed")
        if (
            not isinstance(p9_checks, dict)
            or not p9_checks
            or not all(value is True for value in p9_checks.values())
        ):
            problems.append("P9 causal-monitoring check family is incomplete")
        expected_boundary = {
            "capability_claim": False,
            "cognition_or_sentience_claim": False,
            "energy_measured": False,
            "mechanics_only": True,
            "natural_workloads": False,
            "physical_failures": False,
            "scientific_promotion_allowed": False,
        }
        if not isinstance(p9_boundary, dict):
            problems.append("P9 causal-monitoring claim boundary is absent")
        else:
            for key, expected in expected_boundary.items():
                if p9_boundary.get(key) is not expected:
                    problems.append(f"P9 claim-boundary field {key} has drifted")
        p9_budget = p9_dataset.get("budget_contract") if isinstance(p9_dataset, dict) else None
        p9_verification = p9_dataset.get("verification") if isinstance(p9_dataset, dict) else None
        expected_budget = {
            "branches_per_lineage": 5,
            "independent_units": 5,
            "lineages_per_unit": 52,
            "total_branches": 1300,
            "total_lineages": 260,
        }
        if not isinstance(p9_budget, dict):
            problems.append("P9 dataset budget is absent")
        else:
            for key, expected in expected_budget.items():
                if p9_budget.get(key) != expected:
                    problems.append(f"P9 dataset-budget field {key} has drifted")
        if not isinstance(p9_verification, dict) or p9_verification.get("verified") is not True:
            problems.append("P9 dataset verification is not closed")
        if (
            not isinstance(p9_mutations, dict)
            or p9_mutations.get("count") != 8
            or p9_mutations.get("rejected") != 8
            or p9_mutations.get("all_rejected") is not True
        ):
            problems.append("P9 eight-mutation suite is not closed")
        if (
            not isinstance(p9_resume, dict)
            or p9_resume.get("exact") is not True
            or p9_resume.get("corrupt_checkpoint_rejected") is not True
            or p9_resume.get("completed_chunks") != 15
            or p9_resume.get("final_dataset_sha256") != p9_resume.get("clean_dataset_sha256")
        ):
            problems.append("P9 interrupted-resume contract is not exact")
        if not isinstance(p9_units, list) or len(p9_units) != 5:
            problems.append("P9 causal-monitoring unit count has drifted")
        else:
            for index, unit in enumerate(p9_units):
                arms = unit.get("arms") if isinstance(unit, dict) else None
                matched = unit.get("matched_histogram_capacity") if isinstance(unit, dict) else None
                if not isinstance(unit, dict) or unit.get("all_mechanics_ok") is not True:
                    problems.append(f"P9 unit {index} is not mechanics-pass")
                if not isinstance(arms, dict) or len(arms) != 9:
                    problems.append(f"P9 unit {index} does not execute nine arms")
                if not isinstance(matched, dict) or matched.get("matched") is not True:
                    problems.append(f"P9 unit {index} matched-capacity contract is open")
                if not isinstance(unit, dict) or unit.get("scientific_promotion_allowed") is not False:
                    problems.append(f"P9 unit {index} permits scientific promotion")
        expected_aggregate_metrics = {
            "brier_improvement",
            "controller_utility_delta",
            "intervention_sign_agreement_delta",
            "roc_auc_delta",
        }
        if not isinstance(p9_aggregate, dict) or set(p9_aggregate) != expected_aggregate_metrics:
            problems.append("P9 causal-versus-correlational aggregate is incomplete")
        else:
            for metric, value in p9_aggregate.items():
                if (
                    not isinstance(value, dict)
                    or value.get("positive_units") != 5
                    or value.get("negative_units") != 0
                    or value.get("zero_units") != 0
                ):
                    problems.append(f"P9 aggregate metric {metric} has drifted")
        implementation = p9_source.get("implementation")
        accounting_row = next(
            (
                row
                for row in implementation or []
                if isinstance(row, dict) and row.get("path") == "proof/P9_ACCOUNTING_MECHANICS.json"
            ),
            None,
        )
        if not isinstance(accounting_row, dict):
            problems.append("P9 preflight does not bind the standalone accounting proof")
        else:
            accounting_path, accounting_problem = _repo_file(repo_root, accounting_row.get("path"))
            if accounting_problem:
                problems.append(f"P9 accounting proof path is invalid: {accounting_problem}")
            elif accounting_path is None or not accounting_path.is_file():
                problems.append("P9 accounting proof is absent")
            elif accounting_row.get("sha256") != _sha256(accounting_path):
                problems.append("P9 accounting proof hash disagrees with its implementation binding")
            else:
                accounting = _load_json(accounting_path)
                accounting_boundary = (
                    accounting.get("claim_boundary") if isinstance(accounting, dict) else None
                )
                accounting_energy = accounting.get("energy") if isinstance(accounting, dict) else None
                accounting_phases = accounting.get("phases") if isinstance(accounting, dict) else None
                if (
                    not isinstance(accounting, dict)
                    or accounting.get("schema") != "mop-p9-workload-accounting/v1"
                ):
                    problems.append("P9 accounting proof schema has drifted")
                if (
                    not isinstance(accounting_boundary, dict)
                    or accounting_boundary.get("scientific_promotion") is not False
                ):
                    problems.append("P9 accounting proof permits scientific promotion")
                if not isinstance(accounting_energy, dict) or accounting_energy.get("measured") is not False:
                    problems.append("P9 accounting proof claims measured energy")
                if not isinstance(accounting_phases, list) or len(accounting_phases) != 4:
                    problems.append("P9 accounting proof phase ledger has drifted")
    p9_atlas = atlas.get("causal_monitoring_accounting_preflight")
    if not isinstance(p9_atlas, dict):
        problems.append("atlas P9 causal-monitoring/accounting summary is absent")
    else:
        expected_p9_fields = {
            "status": "mechanics-pass",
            "receipt": "proof/P9_CAUSAL_MONITORING_PREFLIGHT.json",
            "accounting_receipt": "proof/P9_ACCOUNTING_MECHANICS.json",
            "independent_units": 5,
            "total_lineages": 260,
            "total_branches": 1300,
            "arm_count": 9,
            "mutation_count": 8,
            "all_mutations_rejected": True,
            "exact_interrupted_resume": True,
            "energy_measured": False,
            "scientific_promotion": False,
        }
        for key, expected in expected_p9_fields.items():
            if p9_atlas.get(key) != expected:
                problems.append(f"atlas P9 field {key} has drifted")
    return problems, {
        "dense_integration_complete": not any(
            "E6/DR14" in problem or "dense-task" in problem or "dense_task" in problem for problem in problems
        ),
        "wave_e0_mechanics_pass": not any("Wave E0" in problem for problem in problems),
        "p6_mechanics_pass": not any("P6" in problem for problem in problems),
        "p6_current_events_per_stream": 384,
        "p6_remaining_rungs": [10000, 100000, 1000000],
        "p7_mechanics_pass": not any("P7" in problem for problem in problems),
        "p7_independent_units": 3,
        "p7_arm_count": 8,
        "p9_mechanics_pass": not any("P9" in problem for problem in problems),
        "p9_independent_units": 5,
        "p9_total_lineages": 260,
        "p9_total_branches": 1300,
        "p9_arm_count": 9,
    }


def _studio_gate_contract(
    atlas: dict[str, Any], requirements: dict[str, Any], parsed_sources: dict[str, dict[str, Any]]
) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    portfolio = atlas.get("portfolio")
    hardware = portfolio.get("hardware_boundary") if isinstance(portfolio, dict) else None
    if not isinstance(hardware, dict):
        problems.append("portfolio hardware_boundary must be an object")
    else:
        if hardware.get("studio_scale_required_now") is not False:
            problems.append("atlas claims Studio scale is required now")
        if hardware.get("extended_compute_beneficial_rows") != 0:
            problems.append("atlas claims a nonzero extended-compute benefit count")
        if hardware.get("extended_compute_required_rows") != 0:
            problems.append("atlas claims a nonzero extended-compute requirement count")
    escalation = atlas.get("studio_escalation")
    if not isinstance(escalation, dict):
        problems.append("atlas Studio escalation state is absent")
    elif escalation.get("earned_now") is not False:
        problems.append("atlas Studio escalation state is not fail-closed false")
    else:
        for gate_name in ("benefit_gate", "necessity_gate", "not_sufficient"):
            gate = escalation.get(gate_name)
            if (
                not isinstance(gate, list)
                or not gate
                or not all(isinstance(value, str) and value.strip() for value in gate)
            ):
                problems.append(f"atlas Studio {gate_name} is absent or invalid")
    catalog = atlas.get("irreducible_gate_catalog")
    if not isinstance(catalog, dict) or catalog.get("hardware_compute") != []:
        problems.append("atlas irreducible hardware_compute gate list is not empty")
    decision = requirements.get("decision")
    if not isinstance(decision, dict):
        problems.append("requirements decision is absent")
    else:
        if decision.get("studio_scale_required_now") is not False:
            problems.append("requirements decision claims Studio scale is required now")
        if decision.get("extended_compute_beneficial_count") != 0:
            problems.append("requirements decision has a nonzero category-8 benefit count")
        if decision.get("extended_compute_required_count") != 0:
            problems.append("requirements decision has a nonzero category-9 requirement count")
    rows = requirements.get("rows")
    category8: list[dict[str, Any]] = []
    category9: list[dict[str, Any]] = []
    hardware_required: list[dict[str, Any]] = []
    if isinstance(rows, list):
        category8 = [row for row in rows if isinstance(row, dict) and row.get("primary_category") == 8]
        category9 = [row for row in rows if isinstance(row, dict) and row.get("primary_category") == 9]
        hardware_required = [
            row for row in rows if isinstance(row, dict) and row.get("hardware_required") is True
        ]
        if category8:
            problems.append(f"requirements contain category-8 rows: {[row.get('id') for row in category8]}")
        if category9:
            problems.append(f"requirements contain category-9 rows: {[row.get('id') for row in category9]}")
        if hardware_required:
            problems.append(
                f"requirements contain hardware-required rows: {[row.get('id') for row in hardware_required]}"
            )
    boundary = parsed_sources.get("proof/FORM_SUBSTRATE/PRE_STUDIO_BOUNDARY.json")
    if not isinstance(boundary, dict):
        problems.append("verified Form boundary source is unavailable")
    else:
        if boundary.get("schema") != "mop-form-pre-studio-boundary/v1":
            problems.append("Form boundary source has an unexpected schema")
        if boundary.get("studio_is_only_remaining_hardware_boundary") is not False:
            problems.append("Form boundary claims Studio is the only remaining hardware boundary")
        if boundary.get("ready_for_studio_handoff") is not False:
            problems.append("Form boundary claims readiness for Studio handoff")
    completion = parsed_sources.get("proof/COMPLETION_CLAIM_AUDIT.json")
    conclusion = completion.get("conclusion") if isinstance(completion, dict) else None
    completion_self = completion.get("self_verification") if isinstance(completion, dict) else None
    historical_evidence = completion.get("historical_evidence") if isinstance(completion, dict) else None
    if not isinstance(completion, dict) or not isinstance(conclusion, dict):
        problems.append("verified completion-claim source is unavailable")
    else:
        if completion.get("schema") != "mop-completion-claim-audit/v1":
            problems.append("completion-claim source has an unexpected schema")
        if conclusion.get("studio_is_currently_the_only_boundary") is not False:
            problems.append("completion audit claims Studio is currently the only boundary")
        if conclusion.get("measured_hardware_blocker_count") != 0:
            problems.append("completion audit reports a measured hardware blocker")
        if not isinstance(completion_self, dict):
            problems.append("completion audit self-verification is unavailable")
        else:
            if completion_self.get("retired_scale_active_dependency") is not False:
                problems.append("completion audit retains a retired scale dependency")
            if completion_self.get("retired_scale_active_selector_count") != 0:
                problems.append("completion audit retains an active retired-scale selector")
            if completion_self.get("final_source_hash_binding_complete") is not True:
                problems.append("completion audit final source binding is incomplete")
        if not isinstance(historical_evidence, list) or len(historical_evidence) != 2:
            problems.append("completion audit retired history is not preserved exactly")
        elif any(
            not isinstance(row, dict) or row.get("active_dependency") is not False
            for row in historical_evidence
        ):
            problems.append("completion audit historical evidence is active")
    return problems, {
        "category8_count": len(category8) if isinstance(rows, list) else None,
        "category9_count": len(category9) if isinstance(rows, list) else None,
        "hardware_required_count": len(hardware_required) if isinstance(rows, list) else None,
    }


def _markdown_contract(markdown_path: Path, atlas: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    if not markdown_path.is_file():
        return [f"markdown artifact does not exist: {markdown_path}"], {}
    text = markdown_path.read_text(encoding="utf-8")
    if "# MOP potential atlas, 2026-07" not in text:
        problems.append("markdown artifact has an unexpected title")
    if "proof/MOP_POTENTIAL_ATLAS.json" not in text:
        problems.append("markdown artifact does not name its machine-readable companion")
    retired_markers = [marker for marker in RETIRED_TEXT_MARKERS if marker in text]
    if retired_markers:
        problems.append(f"markdown retains retired current-path markers: {retired_markers}")
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        problems.append("markdown artifact is not ASCII-only")
    facets = atlas.get("facets")
    facets_by_id: dict[str, dict[str, Any]] = {}
    for row in facets or []:
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            facets_by_id[row["id"]] = row
    table_rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 8 and cells[0] in facets_by_id:
            facet_id = cells[0]
            if facet_id in table_rows:
                problems.append(f"markdown repeats facet score row {facet_id}")
            table_rows[facet_id] = cells
    missing_rows = sorted(set(facets_by_id) - set(table_rows))
    extra_rows = sorted(set(table_rows) - set(facets_by_id))
    if missing_rows or extra_rows:
        problems.append(f"markdown facet table drift: missing={missing_rows}, extra={extra_rows}")
    for facet_id, cells in table_rows.items():
        facet = facets_by_id[facet_id]
        scores = facet.get("scores")
        if not isinstance(scores, dict):
            problems.append(f"atlas facet {facet_id} has no scores for markdown comparison")
            continue
        try:
            table_numbers = {
                "weight": int(cells[2]),
                "scaffolding": int(cells[3]),
                "implementation": int(cells[4]),
                "experiment": int(cells[5]),
                "confirmation": int(cells[6]),
                "overall": float(cells[7]),
            }
        except ValueError:
            problems.append(f"markdown facet row {facet_id} contains a nonnumeric score")
            continue
        expected_numbers = {
            "weight": facet.get("weight"),
            "scaffolding": scores.get("scaffolding"),
            "implementation": scores.get("implementation"),
            "experiment": scores.get("experiment"),
            "confirmation": scores.get("confirmation"),
            "overall": scores.get("overall"),
        }
        if cells[1] != facet.get("title"):
            problems.append(f"markdown facet title {facet_id} disagrees with the atlas")
        for key, expected in expected_numbers.items():
            actual = table_numbers[key]
            if isinstance(expected, float) and isinstance(actual, float):
                agrees = math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            else:
                agrees = actual == expected
            if not agrees:
                problems.append(
                    f"markdown facet {facet_id} field {key} disagrees: {actual!r} != {expected!r}"
                )
    return problems, {
        "bytes": markdown_path.stat().st_size,
        "sha256": _sha256(markdown_path),
        "facet_score_rows": len(table_rows),
    }


def _json_array_span(text: str, member_name: str) -> tuple[int, int]:
    """Locate one top-level JSON member's array without reformatting the document."""
    marker = json.dumps(member_name)
    marker_positions = [match.start() for match in re.finditer(re.escape(marker), text)]
    if len(marker_positions) != 1:
        raise ValueError(f"expected one {member_name!r} member, found {len(marker_positions)}")
    colon = text.find(":", marker_positions[0] + len(marker))
    start = text.find("[", colon + 1) if colon >= 0 else -1
    if colon < 0 or start < 0:
        raise ValueError(f"could not locate the {member_name!r} array")
    depth = 0
    in_string = False
    escaped = False
    for offset in range(start, len(text)):
        character = text[offset]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return start, offset + 1
    raise ValueError(f"unterminated {member_name!r} array")


def _render_refreshed_source_hashes(
    atlas_path: Path, repo_root: Path
) -> tuple[str | None, list[dict[str, str]], list[str]]:
    """Render source-hash-only updates while preserving every unrelated byte."""
    problems: list[str] = []
    try:
        original_text = atlas_path.read_text(encoding="utf-8")
        loaded = json.loads(original_text)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, [], [f"atlas cannot be read for source-hash refresh: {error}"]
    if not isinstance(loaded, dict) or loaded.get("schema") != ATLAS_SCHEMA:
        return None, [], ["source-hash refresh requires the canonical atlas schema"]
    snapshot = loaded.get("source_snapshot")
    if not isinstance(snapshot, list) or not snapshot:
        return None, [], ["source-hash refresh requires a non-empty source_snapshot"]
    try:
        start, end = _json_array_span(original_text, "source_snapshot")
    except ValueError as error:
        return None, [], [str(error)]
    source_text = original_text[start:end]
    updated_snapshot = json.loads(json.dumps(snapshot))
    changes: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for index, row in enumerate(snapshot):
        if not isinstance(row, dict):
            problems.append(f"source snapshot row {index} is not an object")
            continue
        raw_path = row.get("path")
        old_digest = row.get("sha256")
        path, path_problem = _repo_file(repo_root, raw_path)
        if path_problem:
            problems.append(f"source row {index}: {path_problem}")
            continue
        assert path is not None
        if not isinstance(raw_path, str) or raw_path in seen_paths:
            problems.append(f"source row {index} has a missing or duplicate path")
            continue
        seen_paths.add(raw_path)
        if not _valid_sha256(old_digest):
            problems.append(f"source {raw_path} has an invalid existing sha256")
            continue
        if not path.is_file():
            problems.append(f"source {raw_path} does not exist as a file")
            continue
        actual = _sha256(path)
        encoded_path = json.dumps(raw_path, ensure_ascii=True)
        pattern = re.compile(
            r'("path"\s*:\s*' + re.escape(encoded_path) + r'\s*,\s*"sha256"\s*:\s*")([0-9a-f]{64})(")'
        )
        matches = list(pattern.finditer(source_text))
        if len(matches) != 1:
            problems.append(f"source {raw_path} does not have exactly one refreshable hash field")
            continue
        match = matches[0]
        source_text = source_text[: match.start(2)] + actual + source_text[match.end(2) :]
        assert isinstance(updated_snapshot[index], dict)
        updated_snapshot[index]["sha256"] = actual
        if old_digest != actual:
            changes.append({"path": raw_path, "old_sha256": old_digest, "new_sha256": actual})
    if problems:
        return None, changes, problems
    rendered = original_text[:start] + source_text + original_text[end:]
    try:
        rendered_payload = json.loads(rendered)
    except json.JSONDecodeError as error:
        return None, changes, [f"refreshed atlas would be invalid JSON: {error}"]
    loaded["source_snapshot"] = updated_snapshot
    if rendered_payload != loaded:
        return None, changes, ["source-hash refresh changed content outside source_snapshot hashes"]
    return rendered, changes, []


def validate_potential_atlas(
    atlas_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    requirements_path: Path | None = None,
    markdown_path: Path | None = None,
) -> dict[str, Any]:
    """Return a self-contained validation receipt without mutating any source artifact."""
    repo_root = repo_root.resolve()
    atlas_path = atlas_path.resolve()
    requirements_path = (
        requirements_path.resolve()
        if requirements_path is not None
        else repo_root / "proof" / "EXTENDED_COMPUTE_REQUIREMENTS.json"
    )
    markdown_path = (
        markdown_path.resolve() if markdown_path is not None else repo_root / "MOP_POTENTIAL_ATLAS_2026_07.md"
    )
    checks: list[dict[str, Any]] = []
    top_level_problems: list[str] = []
    try:
        loaded_atlas = _load_json(atlas_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        loaded_atlas = None
        top_level_problems.append(f"atlas cannot be loaded as JSON: {error}")
    if not isinstance(loaded_atlas, dict):
        top_level_problems.append("atlas root must be an object")
        atlas: dict[str, Any] = {}
    else:
        atlas = loaded_atlas
    schema_problems = list(top_level_problems)
    if atlas and atlas.get("schema") != ATLAS_SCHEMA:
        schema_problems.append(f"unexpected atlas schema {atlas.get('schema')!r}")
    checks.append(_check("atlas_schema", schema_problems, {"expected": ATLAS_SCHEMA}))

    facet_problems, facet_detail = _guard_pair("facet contract", lambda: _facet_contract(atlas))
    checks.append(_check("facet_contract", facet_problems, facet_detail))
    score_problems, score_detail = _guard_pair("score contract", lambda: _score_contract(atlas))
    checks.append(_check("score_contract", score_problems, score_detail))
    domain_problems, domain_detail = _guard_pair("domain contract", lambda: _domain_contract(atlas))
    checks.append(_check("domain_contract", domain_problems, domain_detail))
    source_problems, source_detail, parsed_sources = _guard_source(lambda: _source_contract(atlas, repo_root))
    checks.append(_check("source_snapshot", source_problems, source_detail))
    evidence_problems, evidence_detail = _guard_pair(
        "evidence path", lambda: _evidence_contract(atlas, repo_root)
    )
    checks.append(_check("evidence_paths", evidence_problems, evidence_detail))

    requirements: dict[str, Any] = {}
    requirements_problems: list[str] = []
    try:
        loaded_requirements = _load_json(requirements_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        loaded_requirements = None
        requirements_problems.append(f"requirements cannot be loaded as JSON: {error}")
    if isinstance(loaded_requirements, dict):
        requirements = loaded_requirements
    else:
        requirements_problems.append("requirements root must be an object")
    expected_requirements_rel = _display_path(requirements_path, repo_root)
    bound_requirements = parsed_sources.get(expected_requirements_rel)
    if bound_requirements is None:
        requirements_problems.append(
            "requirements file is not present with a verified hash in source_snapshot"
        )
    elif requirements != bound_requirements:
        requirements_problems.append("requirements parse disagrees with verified source snapshot")
    checks.append(
        _check(
            "requirements_binding",
            requirements_problems,
            _file_receipt(requirements_path, repo_root),
        )
    )
    category_problems, category_detail = _guard_pair(
        "category-2 partition", lambda: _category2_contract(atlas, requirements)
    )
    checks.append(_check("category2_partition", category_problems, category_detail))
    accounting_problems, accounting_detail = _guard_pair(
        "snapshot accounting",
        lambda: _snapshot_accounting_contract(atlas, requirements, parsed_sources),
    )
    checks.append(_check("snapshot_accounting", accounting_problems, accounting_detail))
    queue_problems, queue_detail = _guard_pair("queue contract", lambda: _queue_contract(atlas))
    checks.append(_check("queue_contract", queue_problems, queue_detail))
    dependency_problems, dependency_detail = _guard_pair(
        "dependency contract", lambda: _dependency_contract(atlas, requirements)
    )
    checks.append(_check("dependency_contract", dependency_problems, dependency_detail))
    mechanics_problems, mechanics_detail = _guard_pair(
        "mechanics progress",
        lambda: _mechanics_progress_contract(atlas, parsed_sources, repo_root),
    )
    checks.append(_check("mechanics_progress", mechanics_problems, mechanics_detail))
    studio_problems, studio_detail = _guard_pair(
        "Studio gate",
        lambda: _studio_gate_contract(atlas, requirements, parsed_sources),
    )
    checks.append(_check("studio_gate_consistency", studio_problems, studio_detail))
    markdown_problems, markdown_detail = _guard_pair(
        "markdown binding", lambda: _markdown_contract(markdown_path, atlas)
    )
    checks.append(_check("markdown_binding", markdown_problems, markdown_detail))

    problems = [f"{check['name']}: {problem}" for check in checks for problem in check.get("problems", [])]
    summary = {
        "facet_count": facet_detail.get("facet_count"),
        "facet_weight_total": facet_detail.get("facet_weight_total"),
        "weighted_score": score_detail.get("weighted_score"),
        "domain_count": domain_detail.get("domain_count"),
        "source_count": source_detail.get("source_count"),
        "evidence_reference_count": evidence_detail.get("evidence_reference_count"),
        "requirements_row_count": category_detail.get("requirements_row_count"),
        "primary_category_counts": category_detail.get("primary_category_counts"),
        "category2_count": category_detail.get("category2_source_count"),
        "category2_cluster_count": category_detail.get("cluster_count"),
        "category2_current_registry_rows": accounting_detail.get("category2_current_registry_rows"),
        "registry_non_f_rows": accounting_detail.get("registry_non_f_rows"),
        "scientific_claim_ready_rows": accounting_detail.get("scientific_claim_ready_rows"),
        "frontier_historical_rows": accounting_detail.get("frontier_historical_rows"),
        "queue_count": queue_detail.get("queue_count"),
        "facet_dependency_count": dependency_detail.get("facet_dependency_count"),
        "dense_integration_complete": mechanics_detail.get("dense_integration_complete"),
        "wave_e0_mechanics_pass": mechanics_detail.get("wave_e0_mechanics_pass"),
        "p6_mechanics_pass": mechanics_detail.get("p6_mechanics_pass"),
        "p6_current_events_per_stream": mechanics_detail.get("p6_current_events_per_stream"),
        "p7_mechanics_pass": mechanics_detail.get("p7_mechanics_pass"),
        "p7_independent_units": mechanics_detail.get("p7_independent_units"),
        "p7_arm_count": mechanics_detail.get("p7_arm_count"),
        "p9_mechanics_pass": mechanics_detail.get("p9_mechanics_pass"),
        "p9_independent_units": mechanics_detail.get("p9_independent_units"),
        "p9_total_lineages": mechanics_detail.get("p9_total_lineages"),
        "p9_total_branches": mechanics_detail.get("p9_total_branches"),
        "p9_arm_count": mechanics_detail.get("p9_arm_count"),
        "studio_scale_required_now": False if not studio_problems else None,
    }
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "all_ok": not problems,
        "problems": problems,
        "checks": checks,
        "summary": summary,
        "artifacts": {
            "atlas": _file_receipt(atlas_path, repo_root),
            "markdown": _file_receipt(markdown_path, repo_root),
            "requirements": _file_receipt(requirements_path, repo_root),
            "validator": _file_receipt(Path(__file__), repo_root),
            "cli": _file_receipt(repo_root / "scripts" / "validate_mop_potential_atlas.py", repo_root),
        },
        "scope": (
            "Structural and evidence-integrity validation only. A passing receipt does not promote any "
            "facet, experiment, hardware purchase, or external claim."
        ),
    }
    return _seal_receipt(receipt)


def _seal_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    receipt.pop("payload_sha256", None)
    canonical = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    receipt["payload_sha256"] = _sha256_bytes(canonical)
    return receipt


def _attach_refresh_check(
    receipt: dict[str, Any], detail: dict[str, Any], problems: list[str]
) -> dict[str, Any]:
    receipt["checks"].append(_check("source_hash_refresh", problems, detail))
    receipt["problems"] = [
        f"{check['name']}: {problem}" for check in receipt["checks"] for problem in check.get("problems", [])
    ]
    receipt["all_ok"] = not receipt["problems"]
    receipt["source_hash_refresh"] = detail
    return _seal_receipt(receipt)


def refresh_source_hashes(
    atlas_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    requirements_path: Path | None = None,
    markdown_path: Path | None = None,
) -> dict[str, Any]:
    """Transactionally refresh only source_snapshot hashes after candidate validation.

    The default validator never calls this function. A changed candidate must pass every normal
    invariant before it can replace the atlas, so refreshing a semantically incompatible gate source
    is refused without publishing the candidate.
    """
    repo_root = repo_root.resolve()
    atlas_path = atlas_path.resolve()
    requirements_path = (
        requirements_path.resolve()
        if requirements_path is not None
        else repo_root / "proof" / "EXTENDED_COMPUTE_REQUIREMENTS.json"
    )
    markdown_path = (
        markdown_path.resolve() if markdown_path is not None else repo_root / "MOP_POTENTIAL_ATLAS_2026_07.md"
    )
    current_receipt = validate_potential_atlas(
        atlas_path,
        repo_root=repo_root,
        requirements_path=requirements_path,
        markdown_path=markdown_path,
    )
    rendered, changes, render_problems = _render_refreshed_source_hashes(atlas_path, repo_root)
    detail: dict[str, Any] = {
        "requested": True,
        "published": False,
        "changed_count": len(changes),
        "changes": changes,
        "policy": "source_snapshot sha256 fields only; publish only after full candidate validation",
    }
    if render_problems or rendered is None:
        return _attach_refresh_check(current_receipt, detail, render_problems)
    if not changes:
        detail["candidate_validation"] = "not_needed_no_hash_drift"
        return _attach_refresh_check(current_receipt, detail, [])

    temporary = atlas_path.with_name(f".{atlas_path.name}.source-hash-refresh.tmp")
    if temporary.exists():
        return _attach_refresh_check(
            current_receipt,
            detail,
            [f"refusing to overwrite existing refresh candidate {temporary}"],
        )
    try:
        temporary.write_text(rendered, encoding="utf-8")
        os.chmod(temporary, atlas_path.stat().st_mode & 0o777)
        candidate_receipt = validate_potential_atlas(
            temporary,
            repo_root=repo_root,
            requirements_path=requirements_path,
            markdown_path=markdown_path,
        )
        detail["candidate_payload_sha256"] = candidate_receipt["payload_sha256"]
        if not candidate_receipt["all_ok"]:
            detail["candidate_problems"] = candidate_receipt["problems"]
            return _attach_refresh_check(
                current_receipt,
                detail,
                ["refreshed candidate failed full validation; atlas was not changed"],
            )
        os.replace(temporary, atlas_path)
    except OSError as error:
        return _attach_refresh_check(
            current_receipt,
            detail,
            [f"source-hash refresh I/O failure: {error}"],
        )
    finally:
        temporary.unlink(missing_ok=True)

    final_receipt = validate_potential_atlas(
        atlas_path,
        repo_root=repo_root,
        requirements_path=requirements_path,
        markdown_path=markdown_path,
    )
    detail["published"] = True
    if not final_receipt["all_ok"]:
        detail["post_publication_problems"] = final_receipt["problems"]
        return _attach_refresh_check(
            final_receipt,
            detail,
            ["source files changed during publication; final validation is closed"],
        )
    return _attach_refresh_check(final_receipt, detail, [])


def write_validation_receipt(receipt: dict[str, Any], path: Path) -> None:
    """Atomically publish a validation receipt, including failures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--atlas", default="proof/MOP_POTENTIAL_ATLAS.json")
    parser.add_argument("--markdown", default="MOP_POTENTIAL_ATLAS_2026_07.md")
    parser.add_argument("--requirements", default="proof/EXTENDED_COMPUTE_REQUIREMENTS.json")
    parser.add_argument("--out", help="optional atomic validation-receipt path")
    parser.add_argument(
        "--refresh-source-hashes",
        action="store_true",
        help=(
            "transactionally update only source_snapshot hashes; a fully validated candidate is "
            "required before publication"
        ),
    )
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    root = Path(arguments.repo_root).resolve()

    def resolve_argument(raw: str) -> Path:
        path = Path(raw)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    validator_arguments = {
        "repo_root": root,
        "requirements_path": resolve_argument(arguments.requirements),
        "markdown_path": resolve_argument(arguments.markdown),
    }
    if arguments.refresh_source_hashes:
        receipt = refresh_source_hashes(
            resolve_argument(arguments.atlas),
            **validator_arguments,
        )
    else:
        receipt = validate_potential_atlas(
            resolve_argument(arguments.atlas),
            **validator_arguments,
        )
    if arguments.out:
        write_validation_receipt(receipt, resolve_argument(arguments.out))
    print(
        json.dumps(
            {
                "all_ok": receipt["all_ok"],
                "problem_count": len(receipt["problems"]),
                "summary": receipt["summary"],
                "out": arguments.out,
                "source_hash_refresh": receipt.get("source_hash_refresh"),
            },
            sort_keys=True,
        )
    )
    if not receipt["all_ok"]:
        for problem in receipt["problems"]:
            print(problem, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
