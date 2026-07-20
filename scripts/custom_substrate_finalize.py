#!/usr/bin/env python
"""Freeze, attest, independently verify, and compose the completed CM7 receipt chain."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from mop.config import REPO_ROOT
from mop.evidence import atomic_write_bytes, atomic_write_json, json_bytes, sha256_file
from mop.substrate.custom_workbench import audit_requirements

RAW_SCHEMA = "mop-custom-substrate-workbench/v1"
ATTEST_SCHEMA = "mop-custom-substrate-current-evidence-attestation/v1"
ENV_SCHEMA = "mop-custom-substrate-environment-receipt/v1"
VERIFIER_SCHEMA = "mop-custom-substrate-cm7-independent-verifier/v1"
CHAIN_SCHEMA = "mop-custom-substrate-receipt-chain/v1"
CANDIDATES = ("predictive", "invariance", "reconstruction")
CONTROLS = ("random_target", "frozen_random")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_copy_exact(
    source: Path,
    target: Path,
    expected_sha256: str,
    *,
    replace_existing: bool = False,
    previously_bound_sha256: str | None = None,
) -> None:
    raw = source.read_bytes()
    actual = _sha_bytes(raw)
    if actual != expected_sha256:
        raise RuntimeError(f"receipt-chain source hash drift at {source}: {actual} != {expected_sha256}")
    if target.exists():
        existing = sha256_file(target)
        if existing == expected_sha256:
            return
        if not replace_existing or existing != previously_bound_sha256:
            raise RuntimeError(f"durable receipt-chain target already exists with a different hash: {target}")
    atomic_write_bytes(target, raw)


def materialize_proof_chain(
    run_dir: Path,
    proof_path: Path,
    *,
    replace_existing: bool = False,
) -> dict[str, Any]:

    run_composite_path = run_dir / "workbench_receipt.json"
    composite = json.loads(run_composite_path.read_text())
    if composite.get("schema") != RAW_SCHEMA or composite.get("receipt_chain_schema") != CHAIN_SCHEMA:
        raise RuntimeError("run receipt is not a finalized CM7 composite")
    links = composite.get("receipt_chain")
    if not isinstance(links, dict) or not links:
        raise RuntimeError("finalized CM7 composite has no receipt chain")

    proof = proof_path if proof_path.is_absolute() else REPO_ROOT / proof_path
    chain_dir = proof.parent / f"{proof.stem}_CHAIN"
    prior_links: dict[str, dict[str, str]] = {}
    if replace_existing:
        if not proof.is_file():
            raise RuntimeError("cannot replace a durable chain without its existing composite")
        prior = json.loads(proof.read_text())
        if prior.get("receipt_chain_schema") != CHAIN_SCHEMA:
            raise RuntimeError("existing durable composite has no valid receipt chain")
        prior_links = prior.get("receipt_chain") or {}
    durable_links: dict[str, dict[str, str]] = {}
    for role, raw_link in links.items():
        if not isinstance(raw_link, dict):
            raise RuntimeError(f"invalid receipt-chain link for {role}")
        filename = Path(str(raw_link.get("path") or "")).name
        expected = str(raw_link.get("sha256") or "")
        if not filename or filename in {".", ".."} or len(expected) != 64:
            raise RuntimeError(f"unsafe or incomplete receipt-chain link for {role}")
        source = run_dir / filename
        target = chain_dir / filename
        previously_bound_sha256: str | None = None
        if replace_existing:
            prior_link = prior_links.get(str(role))
            if not isinstance(prior_link, dict):
                raise RuntimeError(f"existing durable composite has no link for {role}")
            prior_path = Path(str(prior_link.get("path") or ""))
            if not prior_path.is_absolute():
                prior_path = REPO_ROOT / prior_path
            if prior_path.resolve() != target.resolve():
                raise RuntimeError(f"existing durable composite path drift for {role}")
            previously_bound_sha256 = str(prior_link.get("sha256") or "")
        _atomic_copy_exact(
            source,
            target,
            expected,
            replace_existing=replace_existing,
            previously_bound_sha256=previously_bound_sha256,
        )
        display_path = str(target.relative_to(REPO_ROOT) if target.is_relative_to(REPO_ROOT) else target)
        durable_links[str(role)] = {"path": display_path, "sha256": expected}

    durable = dict(composite)
    durable["receipt_chain"] = durable_links
    atomic_write_json(proof, durable)
    return durable


def freeze_raw(run_dir: Path, *, allow_finalized: bool = False) -> tuple[dict[str, Any], str]:
    current_path = run_dir / "workbench_receipt.json"
    current = json.loads(current_path.read_text())
    embedded = current.pop("evidence_attestation", None)
    if embedded is not None and not isinstance(embedded, dict):
        raise RuntimeError("legacy embedded attestation is malformed")
    if current.get("receipt_chain_schema") == CHAIN_SCHEMA:
        if not allow_finalized:
            raise RuntimeError("existing receipt is already a finalized CM7 composite")
        links = current.get("receipt_chain") or {}
        raw_link = links.get("raw_training_receipt")
        if not isinstance(raw_link, dict):
            raise RuntimeError("existing receipt has no bound raw training receipt")
        raw_path = run_dir / Path(str(raw_link.get("path") or "")).name
        expected = str(raw_link.get("sha256") or "")
        if not raw_path.is_file() or len(expected) != 64 or sha256_file(raw_path) != expected:
            raise RuntimeError("existing raw training receipt does not match the finalized binding")
        raw_bytes = raw_path.read_bytes()
        raw = json.loads(raw_bytes)
        reconstructed = {
            key: value
            for key, value in current.items()
            if key not in {"receipt_chain_schema", "receipt_chain", "authoritative_promotion"}
        }
        if json_bytes(reconstructed) != raw_bytes:
            raise RuntimeError("finalized composite does not reconstruct its bound raw receipt")
        if raw.get("schema") != RAW_SCHEMA or not raw.get("complete"):
            raise RuntimeError("bound raw receipt is not a complete workbench result")
        return raw, expected
    raw_bytes = json_bytes(current)
    raw_hash = _sha_bytes(raw_bytes)
    if isinstance(embedded, dict):
        expected = str(embedded.get("original_receipt_sha256", ""))
        if raw_hash != expected:
            raise RuntimeError(f"raw reconstruction hash {raw_hash} does not match embedded {expected}")
    raw_path = run_dir / "raw_workbench_receipt.json"
    if raw_path.exists() and sha256_file(raw_path) != raw_hash:
        raise RuntimeError("immutable raw receipt already exists with a different hash")
    if not raw_path.exists():
        atomic_write_bytes(raw_path, raw_bytes)
    if current.get("schema") != RAW_SCHEMA or not current.get("complete"):
        raise RuntimeError("raw receipt is not a complete workbench result")
    return current, raw_hash


def build_attestation(run_dir: Path, raw_hash: str, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    start_path = run_dir / "requirements_audit.json"
    config = json.loads((run_dir / "resolved_config.json").read_text())
    start = json.loads(start_path.read_text())
    current = audit_requirements(config["requirements_ledger"], repo_root=repo_root)
    current_path = run_dir / "requirements_current_audit.json"
    atomic_write_json(current_path, current)
    problems: list[str] = []
    implementation_path = run_dir / "implementation_manifest.json"
    implementation = json.loads(implementation_path.read_text())
    start_rows = [source for requirement in start["requirements"] for source in requirement["sources"]]
    snapshot_checks: list[dict[str, Any]] = []
    implementation_checks: list[dict[str, Any]] = []
    for rows, path_key, label, checks in (
        (start_rows, "path", "requirements", snapshot_checks),
        (implementation["files"], "source_path", "core implementation", implementation_checks),
    ):
        for row in rows:
            snapshot = repo_root / row["snapshot_path"]
            actual = sha256_file(snapshot) if snapshot.is_file() else None
            expected = row["snapshot_sha256"]
            ok = actual is not None and actual == expected
            checks.append(
                {
                    "source_path": row[path_key],
                    "snapshot_path": row["snapshot_path"],
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "ok": ok,
                }
            )
            if not ok:
                problems.append(f"invalid {label} snapshot: {row[path_key]}")

    start_sources = {str(source["path"]): source for source in start_rows}
    live_sources = {
        str(source["path"]): source
        for requirement in current["requirements"]
        for source in requirement["sources"]
    }
    drift: list[dict[str, Any]] = []
    for path, old in start_sources.items():
        live = live_sources.get(path)
        if live is None:
            problems.append(f"current source missing: {path}")
            continue
        old_schema = (old.get("observation") or {}).get("schema")
        new_schema = (live.get("observation") or {}).get("schema")
        compatible = old_schema == new_schema
        if not compatible:
            problems.append(f"source schema drift: {path}: {old_schema!r} -> {new_schema!r}")
        if old.get("sha256") != live.get("sha256"):
            drift.append(
                {
                    "path": path,
                    "start_sha256": old.get("sha256"),
                    "current_sha256": live.get("sha256"),
                    "schema_compatible": compatible,
                }
            )
    semantic_same = start.get("ledger_sha256") == current.get("ledger_sha256") and start.get(
        "requirement_ids"
    ) == current.get("requirement_ids")
    if not semantic_same:
        problems.append("requirements ledger semantics changed")
    if not current.get("all_ok"):
        problems.append("current requirements audit is not clean")
    return {
        "schema": ATTEST_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "raw_training_receipt_path": "raw_workbench_receipt.json",
        "raw_training_receipt_sha256": raw_hash,
        "start_audit_path": "requirements_audit.json",
        "start_audit_sha256": sha256_file(start_path),
        "current_audit_path": "requirements_current_audit.json",
        "current_audit_sha256": sha256_file(current_path),
        "implementation_manifest_path": "implementation_manifest.json",
        "implementation_manifest_sha256": sha256_file(implementation_path),
        "implementation_snapshot_scope": (
            "core experiment, generator, runner, experiment wrapper, config, and requirements only; "
            "transitive runtime dependencies are recorded separately, not claimed as source snapshots"
        ),
        "training_design_snapshot_self_verifies": all(row["ok"] for row in snapshot_checks),
        "implementation_snapshot_self_verifies": all(row["ok"] for row in implementation_checks),
        "requirements_semantics_unchanged": semantic_same,
        "current_evidence_all_ok": current.get("all_ok"),
        "source_drift": drift,
        "snapshot_checks": snapshot_checks,
        "implementation_checks": implementation_checks,
        "problems": problems,
        "scientifically_current": not problems,
        "all_ok": not problems,
    }


def _hash_command(command: list[str]) -> tuple[str, int]:
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    digest = hashlib.sha256()
    assert process.stdout is not None
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    _, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"{command} failed: {stderr.decode(errors='replace')}")
    return digest.hexdigest(), process.returncode


def build_environment(run_dir: Path, raw_hash: str) -> dict[str, Any]:
    implementation_path = run_dir / "implementation_manifest.json"
    implementation = json.loads(implementation_path.read_text())
    lock_paths = [REPO_ROOT / "pyproject.toml", REPO_ROOT / "scaffolding/requirements.freeze.txt"]
    diff_hash, _ = _hash_command(["git", "diff", "HEAD", "--binary", "--no-ext-diff"])
    status_hash, _ = _hash_command(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    packages = {}
    for name in ("torch", "numpy", "omegaconf", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema": ENV_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "raw_training_receipt_sha256": raw_hash,
        "implementation_manifest_sha256": sha256_file(implementation_path),
        "implementation_aggregate_sha256": implementation["aggregate_sha256"],
        "source_inventory_sha256": implementation["aggregate_sha256"],
        "snapshot_scope": "six core files; transitive dependencies are recorded, not snapshotted",
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "torch": torch.__version__,
            "packages": packages,
        },
        "host": {
            "platform": platform.platform(),
            "mac_ver": platform.mac_ver()[0],
            "machine": platform.machine(),
            "processor": platform.processor(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "package_locks": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in lock_paths
        ],
        "git": {
            "head": head,
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT)),
            "tracked_diff_sha256": diff_hash,
            "status_inventory_sha256": status_hash,
        },
        "finalizer_sha256": sha256_file(Path(__file__)),
        "free_disk_bytes": shutil.disk_usage(REPO_ROOT).free,
        "all_ok": True,
    }


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / max(abs(d), 1e-300) * (1.0 if d >= 0 else -1.0)
    h = d
    for m in range(1, 401):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(value: float, df: int) -> float:
    x = df / (df + value * value)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - tail if value >= 0 else tail


def student_t_ppf(probability: float, df: int) -> float:
    low, high = -64.0, 64.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if student_t_cdf(middle, df) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _comparison(values: list[float], margin: float) -> dict[str, Any]:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
    se = sd / math.sqrt(n)
    if se == 0:
        statistic = math.inf if mean > margin else -math.inf
        p_value = 0.0 if mean > margin else 1.0
    else:
        statistic = (mean - margin) / se
        p_value = 1.0 - student_t_cdf(statistic, n - 1)
    return {
        "n": n,
        "paired_deltas": values,
        "mean_delta": mean,
        "sd": sd,
        "se": se,
        "margin": margin,
        "t_statistic": statistic,
        "raw_one_sided_p": p_value,
    }


def build_verifier(
    raw: dict[str, Any],
    *,
    raw_hash: str,
    attestation_hash: str,
    environment_hash: str,
    evidence_ok: bool,
    environment_ok: bool,
) -> dict[str, Any]:
    margin, alpha = 0.03, 0.05
    seeds = sorted(raw["seed_results"], key=int)
    scores = {
        arm: [float(raw["seed_results"][seed][arm]["evaluation"]["heldout_combo_score"]) for seed in seeds]
        for arm in (*CANDIDATES, *CONTROLS)
    }
    means = {arm: sum(values) / len(values) for arm, values in scores.items()}
    raw_winner = max(CANDIDATES, key=lambda arm: means[arm])
    comparisons: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        opponents = (*CONTROLS, *(arm for arm in CANDIDATES if arm != candidate))
        for opponent in opponents:
            key = f"{candidate}_vs_{opponent}"
            comparisons[key] = _comparison(
                [left - right for left, right in zip(scores[candidate], scores[opponent], strict=True)],
                margin,
            )
            comparisons[key].update({"candidate": candidate, "opponent": opponent})
    family_size = len(comparisons)
    critical = student_t_ppf(1.0 - alpha / family_size, len(seeds) - 1)
    ordered = sorted(comparisons, key=lambda key: comparisons[key]["raw_one_sided_p"])
    previous = 0.0
    for rank, key in enumerate(ordered):
        adjusted = min(1.0, (family_size - rank) * comparisons[key]["raw_one_sided_p"])
        previous = max(previous, adjusted)
        comparisons[key]["holm_adjusted_p"] = previous
    for row in comparisons.values():
        row["simultaneous_lower_bound"] = row["mean_delta"] - critical * row["se"]
        row["clears_margin"] = row["simultaneous_lower_bound"] > margin and row["holm_adjusted_p"] < alpha
    winner_keys = [key for key, row in comparisons.items() if row["candidate"] == raw_winner]
    gates = {
        "raw_training_complete": bool(raw.get("complete")),
        "five_complete_seeds": len(seeds) == 5
        and all(
            raw["seed_results"][seed][arm]["training"]["complete"]
            for seed in seeds
            for arm in CANDIDATES + ("random_target",)
        ),
        "compute_match": bool(raw["compute_match"]["all_ok"]),
        "off_ceiling": means[raw_winner] < 0.98,
        "winner_clears_all_corrected_comparisons": all(
            comparisons[key]["clears_margin"] for key in winner_keys
        ),
        "current_evidence": evidence_ok,
        "environment_receipt": environment_ok,
    }
    promotion = all(gates.values())
    problems = [name for name, ok in gates.items() if not ok]
    verification_complete = all(
        ok for name, ok in gates.items() if name != "winner_clears_all_corrected_comparisons"
    )
    return {
        "schema": VERIFIER_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "bindings": {
            "raw_training_receipt_sha256": raw_hash,
            "current_evidence_attestation_sha256": attestation_hash,
            "environment_receipt_sha256": environment_hash,
        },
        "selection": {
            "candidate_objectives": list(CANDIDATES),
            "raw_winner": raw_winner,
            "candidate_means": {arm: means[arm] for arm in CANDIDATES},
            "selection_status": "familywise-corrected",
            "family_size": family_size,
            "correction": "Holm one-sided tests plus simultaneous Bonferroni Student-t lower bounds",
            "alpha": alpha,
            "df": len(seeds) - 1,
            "simultaneous_t_critical": critical,
        },
        "paired_comparisons": comparisons,
        "gates": gates,
        "verdict": "promote-local-objective-lever" if promotion else "not-promoted",
        "promotion": promotion,
        "verification_complete": verification_complete,
        "null_valid": verification_complete and not promotion,
        "problems": problems,
        "all_ok": not problems,
    }


def _write_chain_document(run_dir: Path, filename: str, payload: Any) -> dict[str, str]:
    path = run_dir / filename
    atomic_write_json(path, payload)
    return {"path": filename, "sha256": sha256_file(path)}


def finalize(
    run_dir: Path,
    proof_path: Path,
    *,
    allow_finalized: bool = False,
    replace_durable_chain: bool = False,
) -> dict[str, Any]:
    raw, raw_hash = freeze_raw(run_dir, allow_finalized=allow_finalized)
    attestation = build_attestation(run_dir, raw_hash)
    attestation_link = _write_chain_document(run_dir, "current_evidence_attestation.json", attestation)
    environment = build_environment(run_dir, raw_hash)
    environment_link = _write_chain_document(run_dir, "environment_receipt.json", environment)
    verifier = build_verifier(
        raw,
        raw_hash=raw_hash,
        attestation_hash=attestation_link["sha256"],
        environment_hash=environment_link["sha256"],
        evidence_ok=attestation["all_ok"],
        environment_ok=environment["all_ok"],
    )
    verifier_link = _write_chain_document(run_dir, "independent_verifier.json", verifier)
    links = {
        "raw_training_receipt": {"path": "raw_workbench_receipt.json", "sha256": raw_hash},
        "current_evidence_attestation": attestation_link,
        "environment_receipt": environment_link,
        "independent_verifier": verifier_link,
    }
    gates = {
        "raw_training_complete": bool(raw["complete"]),
        "evidence_current": bool(attestation["scientifically_current"]),
        "environment_all_ok": bool(environment["all_ok"]),
        "independent_verifier_promotes": bool(verifier["promotion"]),
    }
    authoritative = {
        "cm7_local_objective_lever_promotable": all(gates.values()),
        "cm8_custom_build_promotable": False,
        "verdict": "promote-local-objective-lever" if all(gates.values()) else "not-promoted",
        "raw_promotion_is_preliminary": True,
        "gates": gates,
        "reasons": verifier["problems"],
        "scope_boundary": (
            "authoritative only for deterministic programmatic-video objective selection; never "
            "natural-video, intelligence, sentience, or general-capability evidence"
        ),
    }
    composite = dict(raw)
    composite.update(
        {
            "receipt_chain_schema": CHAIN_SCHEMA,
            "receipt_chain": links,
            "authoritative_promotion": authoritative,
        }
    )
    atomic_write_json(run_dir / "workbench_receipt.json", composite)
    proof = proof_path if proof_path.is_absolute() else REPO_ROOT / proof_path
    materialize_proof_chain(run_dir, proof, replace_existing=replace_durable_chain)
    return composite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--proof", type=Path, default=Path("proof/CUSTOM_SUBSTRATE_PILOT.json"))
    parser.add_argument(
        "--materialize-existing",
        action="store_true",
        help="materialize the durable chain from an already-finalized run without recomputing it",
    )
    parser.add_argument(
        "--refinalize-existing",
        action="store_true",
        help="recompute a finalized run's generated chain while preserving its bound raw receipt",
    )
    args = parser.parse_args()
    if abs(student_t_ppf(0.95, 4) - 2.131846786) > 1e-6:
        raise RuntimeError("dependency-free Student-t implementation failed its df=4 reference")
    if args.materialize_existing and args.refinalize_existing:
        parser.error("--materialize-existing and --refinalize-existing are mutually exclusive")
    if args.materialize_existing:
        composite = materialize_proof_chain(args.run_dir, args.proof)
    elif args.refinalize_existing:
        composite = finalize(
            args.run_dir,
            args.proof,
            allow_finalized=True,
            replace_durable_chain=True,
        )
    else:
        composite = finalize(args.run_dir, args.proof)
    print(json.dumps(composite["authoritative_promotion"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
