"""Staged execution for Substrate Cognitive Material Genesis.

This module owns stage ordering, the inheritance guard, and the published
stage record. The scientific work of each stage lives in the dedicated genesis
modules; this file only sequences them and refuses to advance when a
precondition is not met.
"""

from __future__ import annotations

from typing import Any

from substrate import genesis_config as C
from substrate import genesis_grok as grok
from substrate import genesis_io as io

PREFLIGHT = "SUBSTRATE_GENESIS_PREFLIGHT.json"
CONSTITUTION = "SUBSTRATE_GENESIS_CONSTITUTION.json"
STAGE_RECORD = "SUBSTRATE_GENESIS_STAGE_RECORD.json"

# The inherited Final Revision result. Every field here is verified byte-exact
# before the genesis program is allowed to run a single stage.
INHERITED = {
    "path": "evidence/substrate/final_revision/SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json",
    "sha256": "2f9fe201192d21c180049e7b622acd88a8f053c8a04347e581dcc7323f1b662b",
    "classification": C.STARTING_CLASSIFICATION,
    "nous_status": C.STARTING_NOUS_STATUS,
    "readiness": C.STARTING_READINESS,
    "outcome": "B",
    "starting_closure_result": "terminal_closed_null",
}


def _stop_guard() -> None:
    if io.stopped():
        raise io.Refused("operator stop switch is set")


def inheritance() -> dict[str, Any]:
    """Verify the inherited Outcome B result and history are exactly preserved."""
    path = io.ROOT / INHERITED["path"]
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}

    checks["inherited_classification_present"] = path.is_file()
    if path.is_file():
        document = io.load_json(path)
        checks["inherited_digest_exact"] = document.get("sha256") == INHERITED["sha256"]
        checks["inherited_classification_unchanged"] = document.get("classification") == INHERITED["classification"]
        checks["inherited_nous_status_unchanged"] = document.get("nous_status") == INHERITED["nous_status"]
        checks["inherited_readiness_unchanged"] = document.get("readiness") == INHERITED["readiness"]
        checks["inherited_outcome_unchanged"] = document.get("outcome") == INHERITED["outcome"]
        checks["inherited_closure_null_unchanged"] = document.get("starting_closure_result") == INHERITED["starting_closure_result"]
        checks["inherited_activation_false"] = document.get("activation") is False
        detail["inherited_document_digest"] = document.get("sha256")
    else:
        for key in (
            "inherited_digest_exact",
            "inherited_classification_unchanged",
            "inherited_nous_status_unchanged",
            "inherited_readiness_unchanged",
            "inherited_outcome_unchanged",
            "inherited_closure_null_unchanged",
            "inherited_activation_false",
        ):
            checks[key] = False

    resolved: dict[str, str | None] = {}
    for tag in C.PRESERVED_TAGS:
        resolved[tag] = io.ref_or_none(tag, peel=True)
    checks["preserved_tags_resolve"] = all(value is not None for value in resolved.values())
    detail["preserved_tags"] = resolved

    head_of_final_revision = io.ref_or_none(C.FINAL_REVISION_TERMINAL_TAG, peel=True)
    checks["final_revision_terminal_head_exact"] = head_of_final_revision == C.FINAL_REVISION_TERMINAL_HEAD
    detail["final_revision_terminal_head"] = head_of_final_revision

    merge_base = io.git("merge-base", "HEAD", C.FINAL_REVISION_TERMINAL_HEAD, check=False)
    checks["descends_from_final_revision"] = merge_base == C.FINAL_REVISION_TERMINAL_HEAD
    detail["merge_base"] = merge_base

    return {"checks": checks, "detail": detail, "all_pass": all(checks.values())}


def preflight(*, publish: bool = True) -> dict[str, Any]:
    """Stage 0. Refuse to start unless the inherited result is intact."""
    _stop_guard()
    inherited = inheritance()
    branch = io.git("rev-parse", "--abbrev-ref", "HEAD", check=False)
    checks = {
        **inherited["checks"],
        "branch_is_genesis": branch == C.IMPLEMENTATION_BRANCH,
        "activation_false": C.ACTIVATION is False,
        "claim_boundary_forbids_unqualified_nous": C.CLAIM_BOUNDARY["unqualified_nous"] is False,
        "claim_boundary_forbids_external_activation": C.CLAIM_BOUNDARY["external_activation"] is False,
        "candidate_count_at_least_eleven": len(C.CANDIDATES) >= 11,
        "controls_present": len(C.CONTROLS) >= 2,
        "challenge_families_at_least_twelve": len(C.CHALLENGE_FAMILIES) >= C.TOURNAMENT_MINIMUM_FAMILIES,
        "review_cells_meet_preferred": len(C.REVIEW_CELLS) >= C.GROK_PREFERRED_ROLES,
        "review_cells_distinct": len(set(C.REVIEW_CELLS)) == len(C.REVIEW_CELLS),
        "mutations_distinct": len(set(C.MUTATIONS)) == len(C.MUTATIONS),
        "sesoi_not_relaxed": C.SESOI >= 0.05,
        "decisive_claim_is_p10": C.DECISIVE_CLAIM == "P10",
    }
    report = io.authority(
        "substrate-genesis-preflight/v1",
        {
            "branch": branch,
            "head": io.git("rev-parse", "HEAD", check=False),
            "inheritance": inherited["detail"],
            "checks": checks,
            "all_pass": all(checks.values()),
        },
    )
    if publish:
        io.write_json(io.EVIDENCE / PREFLIGHT, report)
    return report


def constitution(*, publish: bool = True) -> dict[str, Any]:
    """Stage 1a. Publish and freeze the program constitution."""
    _stop_guard()
    configuration = C.configuration()
    document = io.authority(
        "substrate-genesis-constitution/v1",
        {
            "configuration": configuration,
            "configuration_sha256": C.configuration_digest(),
            "all_pass": True,
        },
    )
    if publish:
        io.write_json(io.CONFIG / "frozen_configuration.json", document)
        io.write_json(io.EVIDENCE / CONSTITUTION, document)
    return document


def stage_record() -> dict[str, Any]:
    """Which stages have published their terminal evidence."""
    existing = io.read_optional(STAGE_RECORD)
    completed = dict(existing.get("stages", {})) if existing else {}
    for stage in C.STAGES:
        completed.setdefault(stage, "pending")
    return {"stages": completed}


def mark_stage(stage: str, state: str) -> dict[str, Any]:
    if stage not in C.STAGES:
        raise io.Refused(f"unknown genesis stage {stage!r}")
    if state not in ("pending", "running", "complete", "refused"):
        raise io.Refused(f"unknown stage state {state!r}")
    record = stage_record()["stages"]
    record[stage] = state
    document = io.authority(
        "substrate-genesis-stage-record/v1",
        {"stages": record, "all_pass": True},
    )
    io.write_json(io.EVIDENCE / STAGE_RECORD, document)
    return document


def status() -> dict[str, Any]:
    return {
        "program": C.PROGRAM,
        "branch": io.git("rev-parse", "--abbrev-ref", "HEAD", check=False),
        "head": io.git("rev-parse", "HEAD", check=False),
        "starting_classification": C.STARTING_CLASSIFICATION,
        "starting_nous_status": C.STARTING_NOUS_STATUS,
        "starting_readiness": C.STARTING_READINESS,
        "stages": stage_record()["stages"],
        "grok": grok.summary(),
        "stop_switch": io.stopped(),
        "activation": False,
    }
