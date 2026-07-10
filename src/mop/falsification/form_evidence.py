"""Durable evidence campaign for the F1-F20 form-substrate series.

Raw harness runs live below ``runs/`` and are ignored by default.  This module does not treat their
presence as evidence.  It locks registry-backed null cards, verifies class/config/registry equality,
rejects stale run snapshots, and promotes a self-contained small JSON receipt into ``proof/``.  It
also emits component-wise operational-awareness and performance-density inputs for the campaign
scorecard.  Scientific positives remain gated by :mod:`mop.falsification.verdict_gate`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..config import REPO_ROOT
from ..diagnostics.operational_awareness import OA_COMPONENTS, OA_SCHEMA
from ..diagnostics.performance_density import DENSITY_SCHEMA
from .experiment_contracts import build_contract_audit, validate_manifest_contract
from .null_cards import render_card, validate_card
from .verdict_gate import build_verdict_gate, write_verdict_gate

CAMPAIGN_SCHEMA = "mop-form-substrate-campaign/v1"
RECEIPT_SCHEMA = "mop-form-experiment-receipt/v1"
PREFLIGHT_SCHEMA = "mop-form-preflight-receipt/v1"
SCORECARD_SCHEMA = "mop-form-campaign-scorecard/v1"
OA_INPUT_SCHEMA = "mop-form-oa-input/v1"
DENSITY_INPUT_SCHEMA = "mop-form-density-input/v1"

CAMPAIGN_PATH = Path("campaign/form_substrate_campaign.yaml")
PROOF_ROOT = Path("proof/FORM_SUBSTRATE")
CONTRACT_AUDIT_PATH = PROOF_ROOT / "CONTRACT_AUDIT.json"
OA_INPUT_PATH = PROOF_ROOT / "OA_INPUT.json"
DENSITY_INPUT_PATH = PROOF_ROOT / "DENSITY_INPUT.json"
SCORECARD_PATH = PROOF_ROOT / "SCORECARD.json"
LOCAL_RUN_SUMMARY_PATH = PROOF_ROOT / "LOCAL_RUN_SUMMARY.json"

LOCAL_REQUIREMENTS = {
    "canonical-run",
    "scaffold-and-smoke",
    "fail-closed-preflight-and-smoke",
}
SCALE_BOUNDARIES = {"local", "studio", "environment", "beyond-studio"}
FORM_CAMPAIGN_IDS = (
    "f1_form_alignment_gate",
    "f2_heldout_form_transfer",
    "f3_form_bottleneck_capacity",
    "f4_raw_payload_vs_form_tokens",
    "f5_cross_form_memory_binding",
    "f6_sensorimotor_form_closure",
    "f7_developmental_form_growth",
    "f8_plastic_substrate_rewrite",
    "f9_cross_form_compositional_binding",
    "f10_intrinsic_form_curriculum",
    "f11_form_dream_replay",
    "f12_private_form_language_stability",
    "f13_form_energy_budget",
    "f14_lifelong_form_expansion",
    "f15_embodied_affordance_form",
    "f16_perfect_slate_null",
    "f17_missing_form_recovery",
    "f18_counterfactual_form_intervention",
    "f19_cross_scale_referent_binding",
    "f20_substrate_crisis_test",
)
_LEGACY_FORM_ID = re.compile(r"^f(?:[1-9]|1[0-9]|20)_")


def load_form_campaign(
    path: Path | str | None = None, *, repo_root: Path | str = REPO_ROOT
) -> dict[str, Any]:
    source = Path(path) if path is not None else Path(repo_root) / CAMPAIGN_PATH
    data = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"form campaign {source} must be a mapping")
    return dict(data)


def validate_form_campaign(
    campaign: dict[str, Any], *, registry_rows: list[dict[str, Any]] | None = None
) -> list[str]:
    """Validate campaign schema, closed vocabularies, coverage, and dependency DAG."""
    from ..devel.registries import load_experiments

    problems: list[str] = []
    if campaign.get("schema") != CAMPAIGN_SCHEMA:
        problems.append(f"unexpected campaign schema {campaign.get('schema')!r}")
    try:
        minimum_seeds = int(campaign.get("minimum_seeds", 0))
    except Exception:
        minimum_seeds = 0
    if minimum_seeds < 3:
        problems.append("campaign minimum_seeds must be >= 3")
    legs = campaign.get("legs")
    if not isinstance(legs, list) or not legs:
        return [*problems, "campaign legs must be a non-empty list"]
    rows = registry_rows if registry_rows is not None else load_experiments()
    # This durable campaign is frozen to the original F1-F20 contracts. Numeric inference would
    # silently admit an alias such as ``f1_variant`` and would make later F-series scaffolds mutate
    # an already published campaign.
    expected = set(FORM_CAMPAIGN_IDS)
    registry_counts = {
        experiment_id: sum(str(row.get("id") or "") == experiment_id for row in rows)
        for experiment_id in FORM_CAMPAIGN_IDS
    }
    for experiment_id, count in registry_counts.items():
        if count != 1:
            problems.append(f"registry must contain {experiment_id} exactly once; found {count}")
            continue
        row = next(row for row in rows if str(row.get("id") or "") == experiment_id)
        if row.get("series") != "F":
            problems.append(f"registry row {experiment_id} must retain series F")
    legacy_aliases = sorted(
        str(row.get("id") or "")
        for row in rows
        if _LEGACY_FORM_ID.match(str(row.get("id") or "")) and str(row.get("id") or "") not in expected
    )
    if legacy_aliases:
        problems.append(f"unfrozen F1-F20 aliases are not campaign members: {legacy_aliases}")
    ids = [str(leg.get("id") or "") for leg in legs if isinstance(leg, dict)]
    if len(ids) != len(set(ids)):
        problems.append("campaign leg ids must be unique")
    actual = set(ids)
    if actual != expected:
        problems.append(
            f"campaign F coverage differs from registry: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    by_id = {str(leg.get("id")): leg for leg in legs if isinstance(leg, dict)}
    for eid, leg in by_id.items():
        requirement = str(leg.get("local_requirement") or "")
        boundary = str(leg.get("scale_boundary") or "")
        if requirement not in LOCAL_REQUIREMENTS:
            problems.append(f"{eid}: local_requirement {requirement!r} not in {sorted(LOCAL_REQUIREMENTS)}")
        if boundary not in SCALE_BOUNDARIES:
            problems.append(f"{eid}: scale_boundary {boundary!r} not in {sorted(SCALE_BOUNDARIES)}")
        dependencies = leg.get("depends_on")
        if not isinstance(dependencies, list):
            problems.append(f"{eid}: depends_on must be a list")
            continue
        for dependency in dependencies:
            if dependency not in by_id:
                problems.append(f"{eid}: dependency {dependency!r} is not a campaign leg")
    problems.extend(_dependency_cycle_problems(by_id))
    return problems


def build_null_card(
    row: dict[str, Any], *, intended_seeds: int, reconstructed_after_audit: bool = True
) -> dict[str, Any]:
    """Build a strict registry-backed F-series null card.

    The card remains conservative: its declared verdict is a tie until a separate run receipt and
    verdict gate decide otherwise.  ``reconstructed-after-audit`` is explicit because historical F
    runs predate this durable card layer; future canonical reruns occur after the contract lock.
    """
    proof = dict(row.get("proof") or {})
    controls = [str(value) for value in row.get("controls") or []]
    factor = str(proof.get("atlas_factor") or "explicit-structured-factor")
    badges = ["contract-locked", "structured-fixture"]
    if reconstructed_after_audit:
        badges.append("reconstructed-after-audit")
    evidence_kind = "RECEIPTS" if row.get("status") == "implemented" else "PREFLIGHT"
    return {
        "exp_id": str(row["id"]),
        "title": str(row["name"]),
        "hypothesis": str(row.get("mechanism") or row.get("question") or "mechanism under test"),
        "null_hypothesis": str(row["null_hypothesis"]),
        "baseline": ", ".join(controls) if controls else "registry-declared matched control",
        "ablation": str(row["falsifier"]),
        "metric": str(list(row.get("metrics") or ["missing-metric"])[0]),
        "probe_dependency": {
            "factor": factor,
            "encoder": "structured-form-fixture",
            "atlas_row": "not-applicable: factor labels are explicit fixture state",
            "decodable": "yes",
            "acc_above_chance": 1.0,
        },
        "encoder_scale": "not-applicable",
        "seeds": {
            "n": int(intended_seeds),
            "sem": "not-estimated-at-contract-lock",
            "sign_stability": "not-estimated-at-contract-lock",
        },
        "provenance_tag": "provisional",
        "result": (
            "contract locked for the next canonical run; this card is not evidence of temporal "
            "preregistration for runs that predate the audit"
        ),
        "taxonomy_category": int(row["taxonomy_slot"]),
        "verdict": "DOWNGRADE-TIE",
        "badges": badges,
        "raw_run_id": f"proof/FORM_SUBSTRATE/{evidence_kind}/{row['id']}.json",
        "repro_level": "R0",
    }


def write_null_cards(*, repo_root: Path | str = REPO_ROOT, overwrite: bool = False) -> dict[str, Any]:
    """Write or verify one canonical null card for every F-series registry row."""
    from ..devel.registries import load_experiments

    root = Path(repo_root)
    campaign = load_form_campaign(repo_root=root)
    campaign_problems = validate_form_campaign(campaign)
    by_id = {str(row["id"]): row for row in load_experiments(root / "registry/experiments.yaml")}
    records: list[dict[str, Any]] = []
    minimum_seeds = int(campaign["minimum_seeds"])
    for leg in campaign["legs"]:
        eid = str(leg["id"])
        row = by_id[eid]
        config_path = root / "configs" / "experiment" / f"{eid}.yaml"
        intended = minimum_seeds
        if config_path.exists():
            loaded_config = OmegaConf.load(config_path)
            if isinstance(loaded_config, DictConfig):
                config = _plain(loaded_config)
                intended = len(list(config.get("seeds") or [])) or minimum_seeds
        intended = max(intended, minimum_seeds)
        card = build_null_card(row, intended_seeds=intended)
        card_problems = validate_card(card, strict=True)
        output = root / PROOF_ROOT / "NULL_CARDS" / f"{eid}.md"
        action = "verified"
        if output.exists() and not overwrite:
            # Existing cards are immutable by default.  Verify their exact canonical null instead of
            # silently rewriting history after a result is known.
            from .null_cards import load_card

            existing = load_card(output)
            if existing.get("null_hypothesis") != row.get("null_hypothesis"):
                card_problems.append(
                    f"{eid}: existing card null differs from registry; explicit overwrite needed"
                )
            action = "kept"
        elif not card_problems:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(_render_form_card(card))
            action = "written"
        records.append(
            {
                "experiment_id": eid,
                "path": _display(root, output),
                "action": action,
                "problems": card_problems,
                "all_ok": not card_problems,
            }
        )
    problems = [*campaign_problems, *(p for record in records for p in record["problems"])]
    return {"records": records, "problems": problems, "all_ok": not problems}


def latest_run_dir(experiment_id: str, *, repo_root: Path | str = REPO_ROOT) -> Path | None:
    """Return the newest numeric harness directory, without claiming it is canonical."""
    base = Path(repo_root) / "runs" / experiment_id
    candidates = (
        [path for path in base.iterdir() if path.is_dir() and path.name.isdigit()] if base.exists() else []
    )
    candidates = [path for path in candidates if (path / "manifest.json").exists()]
    return max(candidates, key=lambda path: int(path.name)) if candidates else None


def build_run_receipt(
    experiment_id: str,
    *,
    source_run_dir: Path | str | None = None,
    repo_root: Path | str = REPO_ROOT,
    contract_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a self-contained canonical-candidate receipt from one ignored harness run."""
    from ..devel.registries import load_experiments

    root = Path(repo_root)
    audit = contract_audit or build_contract_audit(repo_root=root, series="F", implemented_only=False)
    records = {str(record["experiment_id"]): record for record in audit.get("records", [])}
    contract = records.get(experiment_id)
    rows = {str(row["id"]): row for row in load_experiments(root / "registry/experiments.yaml")}
    row = rows.get(experiment_id)
    run_dir = (
        Path(source_run_dir) if source_run_dir is not None else latest_run_dir(experiment_id, repo_root=root)
    )
    if run_dir is not None and not run_dir.is_absolute():
        run_dir = root / run_dir
    problems: list[str] = []
    if row is None:
        problems.append(f"{experiment_id}: not present in experiment registry")
    if contract is None:
        problems.append(f"{experiment_id}: no implemented contract-audit record")
    if run_dir is None:
        problems.append(f"{experiment_id}: no harness run found")
    manifest_path = run_dir / "manifest.json" if run_dir is not None else None
    config_path = run_dir / "config.yaml" if run_dir is not None else None
    manifest = _load_json(manifest_path)
    run_config = _load_yaml(config_path)
    if manifest is None:
        problems.append(f"{experiment_id}: missing or invalid source manifest")
    if run_config is None:
        problems.append(f"{experiment_id}: missing or invalid source config snapshot")
    if manifest is not None and manifest.get("status") != "ok":
        problems.append(f"{experiment_id}: source run status is {manifest.get('status')!r}")
    if manifest is not None and run_config is not None and contract is not None:
        problems.extend(validate_manifest_contract(manifest, contract, run_config.get("experiment", {})))

    live_config_path = root / "configs" / "experiment" / f"{experiment_id}.yaml"
    live_config = _load_yaml(live_config_path)
    snapshot_experiment = dict((run_config or {}).get("experiment") or {})
    if live_config is None:
        problems.append(f"{experiment_id}: live experiment config missing")
    elif live_config != snapshot_experiment:
        problems.append(f"{experiment_id}: run config snapshot is not the current canonical default config")

    metrics = dict((manifest or {}).get("metrics") or {})
    declared = list((contract or {}).get("canonical", {}).get("metric") or (row or {}).get("metrics") or [])
    for metric in declared:
        if metric not in metrics:
            problems.append(f"{experiment_id}: declared metric {metric!r} missing from source result")
    if not isinstance(metrics.get("null_supported"), bool):
        problems.append(f"{experiment_id}: result needs a boolean null_supported decision")

    campaign = load_form_campaign(repo_root=root)
    minimum_seeds = int(campaign.get("minimum_seeds", 5))
    intended_seeds = list((live_config or {}).get("seeds") or [])
    observed_seeds = list(metrics.get("seeds") or [])
    if intended_seeds != observed_seeds:
        problems.append(
            f"{experiment_id}: observed seeds {observed_seeds!r} != canonical config seeds {intended_seeds!r}"
        )
    if len(observed_seeds) < minimum_seeds or len(set(observed_seeds)) != len(observed_seeds):
        problems.append(
            f"{experiment_id}: canonical result needs >= {minimum_seeds} unique seeds, got {observed_seeds!r}"
        )
    problems.extend(
        _seed_evidence_problems(
            experiment_id,
            metrics,
            observed_count=len(observed_seeds),
            minimum_seeds=minimum_seeds,
        )
    )

    density = metrics.get("density")
    problems.extend(_density_problems(experiment_id, density, declared))
    oa = extract_oa_components(experiment_id, metrics)
    null_supported = (
        metrics.get("null_supported") if isinstance(metrics.get("null_supported"), bool) else None
    )
    verdict = "DOWNGRADE-TIE" if null_supported is not False else "PUBLISH-POSITIVE"
    created_at = _timestamp_iso((manifest or {}).get("finished"))
    output_path = root / PROOF_ROOT / "RECEIPTS" / f"{experiment_id}.json"
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "created_at": created_at,
        "experiment_id": experiment_id,
        "evidence_scope": "structured synthetic or local mechanics only; not natural-form evidence",
        "canonical": not problems,
        "source": {
            "run_dir": _display(root, run_dir) if run_dir is not None else None,
            "manifest_path": _display(root, manifest_path) if manifest_path is not None else None,
            "manifest_sha256": _sha256(manifest_path),
            "config_path": _display(root, config_path) if config_path is not None else None,
            "config_sha256": _sha256(config_path),
            "git": (manifest or {}).get("git"),
            "device": (manifest or {}).get("device"),
            "platform": (manifest or {}).get("platform"),
            "result_tag": (manifest or {}).get("result_tag"),
        },
        "contract": contract,
        "registry": {
            "status": (row or {}).get("status"),
            "resource_tier": (row or {}).get("resource_tier"),
            "proof": (row or {}).get("proof"),
        },
        "execution": {
            "status": (manifest or {}).get("status"),
            "outer_seed": (manifest or {}).get("seed"),
            "intended_seeds": intended_seeds,
            "observed_seeds": observed_seeds,
            "started": (manifest or {}).get("started"),
            "finished": (manifest or {}).get("finished"),
        },
        "metrics": metrics,
        "declared_metrics": {metric: metrics.get(metric) for metric in declared},
        "null_decision": {
            "null_supported": null_supported,
            "declared_verdict": verdict,
            "positive_requires_independent_verifier": verdict == "PUBLISH-POSITIVE",
            "ledger_ready": False,
        },
        "density": density,
        "operational_awareness": oa,
        "durable_path": _display(root, output_path),
        "problems": problems,
        "all_ok": not problems,
    }
    receipt["receipt_fingerprint"] = _object_sha256(
        {key: value for key, value in receipt.items() if key not in {"receipt_fingerprint", "created_at"}}
    )
    return receipt


def collect_run_receipts(
    *,
    repo_root: Path | str = REPO_ROOT,
    source_dirs: dict[str, Path | str] | None = None,
) -> dict[str, Any]:
    """Collect a durable receipt for every currently implemented F experiment."""
    from ..devel.registries import load_experiments

    root = Path(repo_root)
    audit = build_contract_audit(repo_root=root, series="F", implemented_only=False)
    implemented = [
        str(row["id"])
        for row in load_experiments(root / "registry/experiments.yaml")
        if row.get("series") == "F" and row.get("status") == "implemented"
    ]
    records: list[dict[str, Any]] = []
    for eid in sorted(implemented):
        receipt = build_run_receipt(
            eid,
            source_run_dir=(source_dirs or {}).get(eid),
            repo_root=root,
            contract_audit=audit,
        )
        output = root / PROOF_ROOT / "RECEIPTS" / f"{eid}.json"
        write_json(receipt, output)
        records.append(
            {
                "experiment_id": eid,
                "path": _display(root, output),
                "canonical": receipt["canonical"],
                "problems": receipt["problems"],
            }
        )
    problems = [problem for record in records for problem in record["problems"]]
    return {
        "contract_audit": audit,
        "records": records,
        "summary": {
            "total": len(records),
            "canonical": sum(1 for record in records if record["canonical"]),
            "noncanonical": sum(1 for record in records if not record["canonical"]),
        },
        "problems": problems,
        "all_ok": audit.get("all_ok", False) and not problems,
    }


def run_local_campaign(
    *, repo_root: Path | str = REPO_ROOT, experiment_ids: list[str] | None = None
) -> dict[str, Any]:
    """Run every implemented local F leg once under its exact canonical default config."""
    from .. import config
    from ..devel.registries import load_experiments
    from ..harness.runner import run_experiment
    from ..logging_utils import new_run_dir

    root = Path(repo_root)
    campaign = load_form_campaign(repo_root=root)
    audit = build_contract_audit(repo_root=root, series="F", implemented_only=False)
    if not audit.get("all_ok"):
        raise ValueError(
            "F contract audit must be clean before local campaign execution: " + "; ".join(audit["problems"])
        )
    registry = {str(row["id"]): row for row in load_experiments(root / "registry/experiments.yaml")}
    selected = set(experiment_ids or [])
    unknown = selected - set(registry)
    if unknown:
        raise ValueError(f"unknown --only F experiment ids {sorted(unknown)}")
    runnable = [
        str(leg["id"])
        for leg in campaign.get("legs", [])
        if registry[str(leg["id"])].get("status") == "implemented"
        and (not selected or str(leg["id"]) in selected)
    ]
    records: list[dict[str, Any]] = []
    started = time.time()
    for eid in runnable:
        row = registry[eid]
        if row.get("exp_tier") != "cpu-now":
            records.append(
                {
                    "experiment_id": eid,
                    "status": "refused-nonlocal-tier",
                    "exp_tier": row.get("exp_tier"),
                    "problems": [f"{eid}: implemented local campaign leg has tier {row.get('exp_tier')!r}"],
                    "all_ok": False,
                }
            )
            continue
        run_dir = new_run_dir(eid, root=root / "runs")
        cfg = config.compose(
            [f"experiment={eid}", "device=cpu", "result_tag=structured-synthetic"],
            config_dir=root / "configs",
        )
        t0 = time.perf_counter()
        try:
            metrics = run_experiment(cfg, run_dir=run_dir)
            problems: list[str] = []
            status = "ok"
        except Exception as exc:
            metrics = {}
            problems = [f"{type(exc).__name__}: {exc}"]
            status = "error"
        manifest_path = run_dir / "manifest.json"
        records.append(
            {
                "experiment_id": eid,
                "status": status,
                "run_dir": _display(root, run_dir),
                "manifest_sha256": _sha256(manifest_path),
                "seconds": time.perf_counter() - t0,
                "null_supported": metrics.get("null_supported"),
                "problems": problems,
                "all_ok": not problems,
            }
        )
    summary = {
        "schema": "mop-form-local-run-summary/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "started": started,
        "finished": time.time(),
        "requested_ids": sorted(selected),
        "records": records,
        "summary": {
            "planned": len(runnable),
            "completed": sum(1 for record in records if record["all_ok"]),
            "failed": sum(1 for record in records if not record["all_ok"]),
        },
        "all_ok": len(records) == len(runnable) and all(record["all_ok"] for record in records),
    }
    write_json(summary, root / LOCAL_RUN_SUMMARY_PATH)
    return summary


def run_fail_closed_preflights(*, repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Execute the tiny F8/F16 smoke and prove that scientific mode refuses missing evidence.

    This is intentionally the only campaign helper that runs experiment code.  Both lanes are tiny,
    local, non-scientific mechanics checks.  The scientific invocation is made in a temporary
    directory and must raise ``ScientificExecutionRefused`` while leaving its raw refusal receipt.
    """
    from .. import config, devices
    from ..experiments import get_experiment
    from ..harness.runner import run_experiment
    from ..logging_utils import new_run_dir

    root = Path(repo_root)
    campaign = load_form_campaign(repo_root=root)
    gated = [
        str(leg["id"])
        for leg in campaign.get("legs", [])
        if leg.get("local_requirement") == "fail-closed-preflight-and-smoke"
    ]
    records: list[dict[str, Any]] = []
    for eid in gated:
        cfg = config.compose([f"experiment={eid}", "device=cpu"], config_dir=root / "configs")
        run_dir = new_run_dir(eid, root=root / "runs")
        smoke_metrics = run_experiment(cfg, run_dir=run_dir)
        smoke_manifest_path = run_dir / "manifest.json"
        smoke_raw_path = run_dir / "preflight_receipt.json"
        smoke_manifest = _load_json(smoke_manifest_path) or {}
        smoke_raw = _load_json(smoke_raw_path) or {}
        refusal_observed = False
        refusal_error = ""
        refusal_raw: dict[str, Any] = {}
        with tempfile.TemporaryDirectory(prefix=f"mop_{eid}_refusal_") as tmp:
            refusal_dir = Path(tmp)
            scientific_cfg = config.compose(
                [f"experiment={eid}", "device=cpu", "experiment.execution_mode=scientific"],
                config_dir=root / "configs",
            )
            try:
                get_experiment(eid).run(scientific_cfg, devices.resolve("cpu"), refusal_dir)
            except Exception as exc:  # the exact named refusal is asserted below
                refusal_error = f"{type(exc).__name__}: {exc}"
                refusal_observed = type(exc).__name__ == "ScientificExecutionRefused"
            refusal_raw = _load_json(refusal_dir / "preflight_receipt.json") or {}
        problems: list[str] = []
        if smoke_manifest.get("status") != "ok":
            problems.append(f"{eid}: smoke harness status is {smoke_manifest.get('status')!r}")
        if smoke_raw.get("schema") != "mop-scientific-preflight/v2":
            problems.append(f"{eid}: smoke raw preflight schema is invalid")
        if smoke_metrics.get("execution_status") != "smoke-only":
            problems.append(f"{eid}: default execution is not smoke-only")
        if smoke_metrics.get("promotion_eligible") is not False:
            problems.append(f"{eid}: smoke must be permanently non-promotable")
        if smoke_metrics.get("null_evaluated") is not False:
            problems.append(f"{eid}: smoke must not claim a null evaluation")
        if smoke_metrics.get("smoke_mechanics_pass") is not True:
            problems.append(f"{eid}: smoke mechanics failed")
        if not refusal_observed:
            problems.append(f"{eid}: scientific mode did not fail closed ({refusal_error or 'no error'})")
        if refusal_raw.get("requested_mode") != "scientific":
            problems.append(f"{eid}: scientific refusal receipt missing")
        if refusal_raw.get("evidence_eligible") is not False:
            problems.append(f"{eid}: empty evidence package unexpectedly became eligible")
        output = root / PROOF_ROOT / "PREFLIGHT" / f"{eid}.json"
        wrapper = {
            "schema": PREFLIGHT_SCHEMA,
            "created_at": _timestamp_iso(smoke_manifest.get("finished")),
            "experiment_id": eid,
            "preflight_kind": "mechanics smoke plus fail-closed scientific refusal",
            "scientific_result": False,
            "promotion_eligible": False,
            "null_evaluated": False,
            "smoke_mechanics_pass": smoke_metrics.get("smoke_mechanics_pass"),
            "fail_closed": refusal_observed,
            "source": {
                "run_dir": _display(root, run_dir),
                "manifest_sha256": _sha256(smoke_manifest_path),
                "raw_preflight_sha256": _sha256(smoke_raw_path),
            },
            "smoke_receipt": smoke_raw,
            "scientific_refusal_receipt": refusal_raw,
            "scientific_refusal_error": refusal_error,
            "problems": problems,
            "all_ok": not problems,
        }
        wrapper["receipt_fingerprint"] = _object_sha256(
            {key: value for key, value in wrapper.items() if key not in {"receipt_fingerprint", "created_at"}}
        )
        write_json(wrapper, output)
        records.append(
            {
                "experiment_id": eid,
                "path": _display(root, output),
                "problems": problems,
                "all_ok": not problems,
            }
        )
    problems = [problem for record in records for problem in record["problems"]]
    return {"records": records, "problems": problems, "all_ok": not problems}


def build_form_verdict_gates(*, repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Gate every implemented F receipt; positives require an independent verifier receipt."""
    from ..devel.registries import load_experiments

    root = Path(repo_root)
    implemented = [
        str(row["id"])
        for row in load_experiments(root / "registry/experiments.yaml")
        if row.get("series") == "F" and row.get("status") == "implemented"
    ]
    records: list[dict[str, Any]] = []
    for eid in sorted(implemented):
        card_path = root / PROOF_ROOT / "NULL_CARDS" / f"{eid}.md"
        receipt_path = root / PROOF_ROOT / "RECEIPTS" / f"{eid}.json"
        verifier_path = root / PROOF_ROOT / "VERIFIERS" / f"{eid}.json"
        receipt = _load_json(receipt_path) or {}
        verdict = str((receipt.get("null_decision") or {}).get("declared_verdict") or "DOWNGRADE-TIE")
        gate = build_verdict_gate(
            null_card_path=card_path,
            run_receipt_path=receipt_path,
            verifier_receipt_path=verifier_path if verifier_path.exists() else None,
            declared_verdict=verdict,
            strict_card=True,
        )
        if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("all_ok") is not True:
            gate["problems"].append("form run receipt is missing or noncanonical")
        expected = (
            "DOWNGRADE-TIE"
            if (receipt.get("null_decision") or {}).get("null_supported") is not False
            else "PUBLISH-POSITIVE"
        )
        if verdict != expected:
            gate["problems"].append(
                f"declared verdict {verdict!r} disagrees with receipt null decision {expected!r}"
            )
        gate["all_ok"] = not gate["problems"]
        output = root / PROOF_ROOT / "VERDICT_GATES" / f"{eid}.json"
        write_verdict_gate(gate, output)
        records.append(
            {
                "experiment_id": eid,
                "path": _display(root, output),
                "verdict": verdict,
                "positive": gate["positive"],
                "problems": gate["problems"],
                "all_ok": gate["all_ok"],
            }
        )
    problems = [problem for record in records for problem in record["problems"]]
    return {
        "records": records,
        "summary": {
            "total": len(records),
            "ready": sum(1 for record in records if record["all_ok"]),
            "blocked": sum(1 for record in records if not record["all_ok"]),
            "positive": sum(1 for record in records if record["positive"]),
        },
        "problems": problems,
        "all_ok": not problems,
    }


def extract_oa_components(experiment_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Translate only directly measured F metrics into OA suite inputs; never infer a composite."""
    components: dict[str, dict[str, Any]] = {}
    if "oa1_missing_form_auroc" in metrics:
        components["oa1_missing_form"] = {
            "auroc": metrics["oa1_missing_form_auroc"],
            "chance": 0.5,
            "source": metrics.get("oa1_source", "unspecified"),
        }
    if "oa2_calibration_auroc" in metrics or "absence_ece" in metrics:
        components["oa2_calibration"] = {
            "auroc": metrics.get("oa2_calibration_auroc"),
            "ece": metrics.get("absence_ece"),
            "chance_auroc": 0.5,
        }
    if "oa3_memory_availability_auroc" in metrics:
        components["oa3_memory_availability"] = {
            "auroc": metrics["oa3_memory_availability_auroc"],
            "chance": 0.5,
        }
    if "oa5_compute_value_auroc" in metrics:
        components["oa5_compute_value"] = {
            "auroc": metrics["oa5_compute_value_auroc"],
            "chance": 0.5,
        }
    if "crisis_auroc" in metrics:
        components["oa6_crisis_detection"] = {
            "auroc": metrics.get("crisis_auroc"),
            "raw_error_auroc": metrics.get("raw_error_auroc"),
            "strongest_baseline_auroc": metrics.get("strongest_baseline_auroc"),
            "avoided_compute_measured": metrics.get("avoided_compute_measured"),
            "chance": 0.5,
        }
    if "false_trigger_rate" in metrics and "true_trigger_rate" in metrics:
        components["oa7_rewrite_caution"] = {
            "false_trigger_rate": metrics.get("false_trigger_rate"),
            "true_trigger_rate": metrics.get("true_trigger_rate"),
            "caution_margin": float(metrics["true_trigger_rate"]) - float(metrics["false_trigger_rate"]),
        }
    return {
        "schema": OA_SCHEMA,
        "experiment_id": experiment_id,
        "components": components,
        "components_present": sorted(components),
        "components_missing": sorted(set(OA_COMPONENTS) - set(components)),
        "composite_score": None,
    }


def build_oa_input(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    sources: dict[str, list[dict[str, Any]]] = {component: [] for component in OA_COMPONENTS}
    for receipt in receipts:
        eid = str(receipt.get("experiment_id") or "")
        oa = dict(receipt.get("operational_awareness") or {})
        for component, metrics in dict(oa.get("components") or {}).items():
            if component in sources:
                sources[component].append({"experiment_id": eid, "metrics": metrics})
    present = {component: rows for component, rows in sources.items() if rows}
    return {
        "schema": OA_INPUT_SCHEMA,
        "oa_schema": OA_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "components": present,
        "components_present": sorted(present),
        "components_missing": sorted(set(OA_COMPONENTS) - set(present)),
        "composite_score": None,
        "all_ok": all(isinstance(row.get("metrics"), dict) for rows in present.values() for row in rows),
    }


def build_density_input(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for receipt in receipts:
        eid = str(receipt.get("experiment_id") or "")
        density = receipt.get("density")
        declared = list((receipt.get("contract") or {}).get("canonical", {}).get("metric") or [])
        row_problems = _density_problems(eid, density, declared)
        problems.extend(row_problems)
        rows.append(
            {
                "experiment_id": eid,
                "primary": (density or {}).get("primary") if isinstance(density, dict) else None,
                "capability": (density or {}).get("capability") if isinstance(density, dict) else None,
                "cost": (density or {}).get("cost") if isinstance(density, dict) else None,
                "density": (density or {}).get("density") if isinstance(density, dict) else None,
                "problems": row_problems,
                "all_ok": not row_problems,
            }
        )
    return {
        "schema": DENSITY_INPUT_SCHEMA,
        "density_schema": DENSITY_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "rows": rows,
        "summary": {
            "total": len(rows),
            "valid": sum(1 for row in rows if row["all_ok"]),
            "invalid": sum(1 for row in rows if not row["all_ok"]),
        },
        "problems": problems,
        "all_ok": not problems,
    }


def build_campaign_scorecard(*, repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Build the receipt-backed F campaign scorecard without inventing missing evidence."""
    from ..devel.registries import load_experiments
    from .null_cards import load_card

    root = Path(repo_root)
    campaign = load_form_campaign(repo_root=root)
    campaign_problems = validate_form_campaign(campaign)
    rows = {str(row["id"]): row for row in load_experiments(root / "registry/experiments.yaml")}
    audit = build_contract_audit(repo_root=root, series="F", implemented_only=False)
    audit_by_id = {str(record["experiment_id"]): record for record in audit.get("records", [])}
    states: list[dict[str, Any]] = []
    valid_receipts: list[dict[str, Any]] = []
    for leg in campaign["legs"]:
        eid = str(leg["id"])
        row = rows[eid]
        card_path = root / PROOF_ROOT / "NULL_CARDS" / f"{eid}.md"
        card_problems: list[str] = []
        if not card_path.exists():
            card_problems.append(f"{eid}: canonical null card missing")
        else:
            try:
                card = load_card(card_path)
                card_problems.extend(validate_card(card, strict=True))
                if card.get("null_hypothesis") != row.get("null_hypothesis"):
                    card_problems.append(f"{eid}: canonical card null differs from registry")
            except Exception as exc:
                card_problems.append(f"{eid}: canonical null card cannot be parsed: {exc}")

        receipt_path = root / PROOF_ROOT / "RECEIPTS" / f"{eid}.json"
        preflight_path = root / PROOF_ROOT / "PREFLIGHT" / f"{eid}.json"
        gate_path = root / PROOF_ROOT / "VERDICT_GATES" / f"{eid}.json"
        receipt = _load_json(receipt_path)
        preflight = _load_json(preflight_path)
        gate = _load_json(gate_path)
        contract_ok = bool(audit_by_id.get(eid, {}).get("all_ok"))
        run_ok = bool(receipt and receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("all_ok"))
        preflight_ok = bool(
            preflight
            and preflight.get("schema") == PREFLIGHT_SCHEMA
            and preflight.get("all_ok")
            and (
                leg["local_requirement"] != "fail-closed-preflight-and-smoke"
                or preflight.get("fail_closed") is True
            )
        )
        local_ok = run_ok if row.get("status") == "implemented" else preflight_ok
        gate_ok = bool(gate and gate.get("schema") == "mop-verdict-gate/v1" and gate.get("all_ok"))
        if run_ok and receipt is not None:
            valid_receipts.append(receipt)
        problems = list(card_problems)
        if not contract_ok:
            problems.extend(audit_by_id.get(eid, {}).get("problems") or [f"{eid}: contract not aligned"])
        if not local_ok:
            needed = (
                "canonical run receipt" if row.get("status") == "implemented" else "local preflight receipt"
            )
            problems.append(f"{eid}: {needed} missing or noncanonical")
        states.append(
            {
                "experiment_id": eid,
                "phase": leg["phase"],
                "registry_status": row.get("status"),
                "local_requirement": leg["local_requirement"],
                "scale_boundary": leg["scale_boundary"],
                "null_card_ok": not card_problems,
                "contract_ok": contract_ok,
                "local_evidence_ok": local_ok,
                "verdict_gate_ok": gate_ok,
                "null_supported": (receipt or {}).get("null_decision", {}).get("null_supported"),
                "problems": problems,
                "all_ok": not problems,
            }
        )
    oa_input = build_oa_input(valid_receipts)
    density_input = build_density_input(valid_receipts)
    local_exhausted = not campaign_problems and all(state["all_ok"] for state in states)
    ledger_ready = all(
        state["verdict_gate_ok"] for state in states if state["registry_status"] == "implemented"
    )
    problems = [*campaign_problems, *(p for state in states for p in state["problems"])]
    return {
        "schema": SCORECARD_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "campaign_schema": campaign.get("schema"),
        "contract_audit": audit,
        "legs": states,
        "operational_awareness_input": oa_input,
        "density_input": density_input,
        "summary": {
            "total_legs": len(states),
            "local_ready": sum(1 for state in states if state["all_ok"]),
            "local_pending": sum(1 for state in states if not state["all_ok"]),
            "canonical_run_receipts": len(valid_receipts),
            "verdict_gates_ready": sum(1 for state in states if state["verdict_gate_ok"]),
        },
        "local_obligations_exhausted": local_exhausted,
        "scientific_ledger_ready": ledger_ready,
        "problems": problems,
        "all_ok": local_exhausted and density_input["all_ok"],
    }


def write_scorecard_inputs(scorecard: dict[str, Any], *, repo_root: Path | str = REPO_ROOT) -> None:
    root = Path(repo_root)
    write_json(scorecard["operational_awareness_input"], root / OA_INPUT_PATH)
    write_json(scorecard["density_input"], root / DENSITY_INPUT_PATH)
    write_json(scorecard, root / SCORECARD_PATH)


def write_json(data: dict[str, Any], path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, default=str) + "\n")


def _density_problems(experiment_id: str, density: Any, declared: list[str]) -> list[str]:
    if not isinstance(density, dict):
        return [f"{experiment_id}: density block missing"]
    problems: list[str] = []
    if density.get("schema") != DENSITY_SCHEMA:
        problems.append(f"{experiment_id}: density schema is {density.get('schema')!r}")
    primary = density.get("primary")
    if primary not in declared:
        problems.append(f"{experiment_id}: density primary {primary!r} is not a declared metric {declared!r}")
    capability = density.get("capability")
    if not isinstance(capability, dict) or primary not in capability:
        problems.append(f"{experiment_id}: density capability omits primary {primary!r}")
    cost = density.get("cost")
    if not isinstance(cost, dict) or not cost:
        problems.append(f"{experiment_id}: density cost is empty")
    elif not any(_finite_positive(value) for value in cost.values()):
        problems.append(f"{experiment_id}: density cost has no finite positive measurement")
    ratios = density.get("density")
    if not isinstance(ratios, dict) or not ratios:
        problems.append(f"{experiment_id}: density ratios are empty")
    return problems


def _seed_evidence_problems(
    experiment_id: str,
    metrics: dict[str, Any],
    *,
    observed_count: int,
    minimum_seeds: int,
) -> list[str]:
    problems: list[str] = []
    interval = metrics.get("seed_ci")
    if not isinstance(interval, dict):
        problems.append(f"{experiment_id}: seed_ci receipt missing")
        interval_n = 0
    else:
        interval_n = int(interval.get("n") or 0)
        # Cross-seed stability tests may form adjacent seed pairs, so S seed models legitimately
        # yield S-1 paired deltas. Require at least three deltas and never more than the seed count.
        if interval_n < 3 or interval_n > observed_count:
            problems.append(
                f"{experiment_id}: seed_ci.n {interval.get('n')!r} must be in [3, {observed_count}] "
                f"for {minimum_seeds}-seed campaign evidence"
            )
        for field in ("mean", "lo", "hi"):
            if not _finite_number(interval.get(field)):
                problems.append(f"{experiment_id}: seed_ci {field!r} must be finite")
    signs = metrics.get("sign_flip_report") or metrics.get("sign_flip")
    if not isinstance(signs, dict):
        problems.append(f"{experiment_id}: sign-flip receipt missing")
    elif int(signs.get("n") or 0) != interval_n:
        problems.append(f"{experiment_id}: sign-flip n {signs.get('n')!r} != seed_ci.n {interval_n}")
    return problems


def _render_form_card(card: dict[str, Any]) -> str:
    """Render an F card without the legacy frozen-encoder assumptions in ``render_card``."""
    # Reuse the canonical YAML serializer/parser contract, replacing only the generic prose wrapper.
    rendered = render_card(card)
    yaml_start = rendered.index("```yaml")
    yaml_block = rendered[yaml_start:].rstrip()
    return "\n".join(
        [
            f"# Canonical null card: {card['exp_id']}",
            "",
            "Locked from `registry/experiments.yaml` for the durable F-series campaign. This card",
            "defines the null and controls; it is not a result receipt. Historical runs that predate",
            "the audit do not acquire retrospective preregistration status. A positive verdict still",
            "requires an independent verifier and verdict-gate receipt.",
            "",
            "## Claim Under Test",
            "",
            str(card["title"]),
            "",
            "## Machine-Readable Card",
            "",
            yaml_block,
            "",
        ]
    )


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except Exception:
        return False
    return math.isfinite(number) and number > 0


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _dependency_cycle_problems(by_id: dict[str, dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    stack: set[str] = set()
    problems: list[str] = []

    def visit(eid: str) -> None:
        if eid in seen:
            return
        if eid in stack:
            problems.append(f"campaign dependency cycle at {eid}")
            return
        stack.add(eid)
        for dependency in by_id[eid].get("depends_on") or []:
            if dependency in by_id:
                visit(str(dependency))
        stack.remove(eid)
        seen.add(eid)

    for eid in by_id:
        visit(eid)
    return problems


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _load_yaml(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception:
        return None
    return {str(key): value for key, value in data.items()} if isinstance(data, dict) else None


def _plain(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        data = OmegaConf.to_container(config, resolve=True)
        return {str(key): value for key, value in data.items()} if isinstance(data, dict) else {}
    return dict(config)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _object_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _timestamp_iso(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    except Exception:
        return datetime.now(UTC).isoformat()


def _display(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
