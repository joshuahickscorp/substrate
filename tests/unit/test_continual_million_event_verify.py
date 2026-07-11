from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from mop.studies.continual_million_event import ContinualSmokeProfile, run_smoke_arm
from mop.studies.continual_million_event_verify import (
    ARMS,
    CLAIM_SCOPE,
    RECORD,
    RECORD_CORE,
    SCHEDULES,
    TIE_RULE,
    VERIFIER_SCHEMA,
    _canonical_bytes,
    _canonical_sha256,
    _decision,
    _embedded_preflight_authority,
    _next_rung_authority,
    _recompute_cell_metrics,
    _replay_stream,
    _stream_composite_sha,
    build_verification_receipt,
    write_verification_receipt,
)
from mop.substrate.continual_stream import (
    ContinualStreamSpec,
    TransitionSchedule,
    materialize_stream,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload) + b"\n")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hash(payload: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(payload)
    output["payload_sha256"] = _canonical_sha256(output)
    return output


def _fixture_repo(tmp_path: Path, *, rung: int = 10_000) -> Path:
    for relative, content in {
        "src/mop/studies/continual_million_event_verify.py": "# verifier fixture\n",
        "scripts/verify_continual_million_event_rung.py": "# verifier driver fixture\n",
        "scripts/continual_million_event_rung.py": "# rung runner fixture\n",
        "src/mop/studies/continual_fixture.py": "VALUE = 1\n",
        "proof/EXPANSION_WAVE0.json": "{}\n",
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    preflight_config = {
        "schema": "mop-continual-million-event-preflight-config/v1",
        "claim_scope": CLAIM_SCOPE,
    }
    preflight_config_path = tmp_path / "configs/experiment/continual_million_event_preflight.yaml"
    preflight_config_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_config_path.write_text(yaml.safe_dump(preflight_config, sort_keys=False))
    preflight_impl = tmp_path / "src/mop/studies/continual_fixture.py"
    wave_path = tmp_path / "proof/EXPANSION_WAVE0.json"
    preflight = _self_hash(
        {
            "schema": "mop-continual-million-event-preflight/v1",
            "claim_scope": CLAIM_SCOPE,
            "status": "mechanics-pass",
            "all_mechanics_ok": True,
            "config": {
                "path": "configs/experiment/continual_million_event_preflight.yaml",
                "sha256": _sha256_file(preflight_config_path),
                "payload": preflight_config,
                "profile_sha256": _canonical_sha256(preflight_config),
            },
            "implementation": [
                {
                    "path": "configs/experiment/continual_million_event_preflight.yaml",
                    "sha256": _sha256_file(preflight_config_path),
                },
                {
                    "path": "src/mop/studies/continual_fixture.py",
                    "sha256": _sha256_file(preflight_impl),
                },
            ],
            "wave_e0": {
                "path": "proof/EXPANSION_WAVE0.json",
                "sha256": _sha256_file(wave_path),
            },
        }
    )
    preflight_path = tmp_path / "proof/CONTINUAL_MILLION_EVENT_PREFLIGHT.json"
    _write_json(preflight_path, preflight)

    seeds = [101, 102, 103, 104, 105]
    profile_config: dict[str, Any] = {
        "replay_capacity": 128,
        "future_accuracy_threshold": 0.5,
        "matched_updates_per_event": 2,
        "minimum_chunk_events": 64,
        "chunks_per_stream": 100,
        "checkpoint_every_events": 1250,
        "future_window_events": 48,
        "threshold_window_events": 16,
        "gradual_width_divisor": 12,
        "deletion_numerator": 3,
        "deletion_denominator": 8,
    }
    rung_config = {
        "schema": "mop-continual-progressive-rungs-config/v1",
        "claim_scope": CLAIM_SCOPE,
        "replication": {
            "rungs": [10_000, 100_000, 1_000_000],
            "seeds": seeds,
            "schedules": list(SCHEDULES),
            "arms": list(ARMS),
            "minimum_independent_seeds": 5,
            "independent_metric_verifier_required": True,
        },
        "profile": profile_config,
    }
    rung_config_path = tmp_path / "configs/experiment/continual_million_event_rungs.yaml"
    rung_config_path.write_text(yaml.safe_dump(rung_config, sort_keys=False))
    runner_path = tmp_path / "scripts/continual_million_event_rung.py"
    source_authority, authority_problems = _embedded_preflight_authority(preflight, repo_root=tmp_path)
    assert authority_problems == []

    profile: dict[str, Any] = {
        "checkpoint_every": profile_config["checkpoint_every_events"],
        "replay_capacity": profile_config["replay_capacity"],
        "future_window_events": profile_config["future_window_events"],
        "threshold_window_events": profile_config["threshold_window_events"],
        "future_accuracy_threshold": profile_config["future_accuracy_threshold"],
        "matched_updates_per_event": profile_config["matched_updates_per_event"],
    }
    stream_profile: dict[str, int] = {
        "chunk_events": max(
            profile_config["minimum_chunk_events"], rung // profile_config["chunks_per_stream"]
        ),
        "n_domains": 4,
        "n_classes": 4,
        "gradual_width_events": max(1, rung // profile_config["gradual_width_divisor"]),
        "deletion_event": (
            rung * profile_config["deletion_numerator"] // profile_config["deletion_denominator"]
        ),
    }
    plan_cells = [
        {"seed": seed, "schedule": schedule, "arm": arm}
        for seed in seeds
        for schedule in SCHEDULES
        for arm in ARMS
    ]
    plan = {
        "mode": "replication",
        "rung": rung,
        "seeds": seeds,
        "schedules": list(SCHEDULES),
        "arms": list(ARMS),
        "cells": plan_cells,
        "expected_cells": 30,
        "stream": stream_profile,
        "profile": profile,
    }
    identity = {
        "config_sha256": _sha256_file(rung_config_path),
        "runner_sha256": _sha256_file(runner_path),
        "source_preflight_file_sha256": _sha256_file(preflight_path),
        "source_preflight_payload_sha256": preflight["payload_sha256"],
        "source_live_bindings_sha256": source_authority["bindings_sha256"],
        "plan": plan,
        "claim_scope": CLAIM_SCOPE,
    }
    work_root = tmp_path / "runs/continual_million_event/rung"
    cells: dict[str, dict[str, Any]] = {}
    smoke_profile = ContinualSmokeProfile(**profile)
    for seed in seeds:
        for schedule in SCHEDULES:
            stream_root = work_root / "streams" / f"seed_{seed}" / schedule
            spec = ContinualStreamSpec(
                seed=seed,
                total_events=rung,
                chunk_events=stream_profile["chunk_events"],
                n_domains=stream_profile["n_domains"],
                n_classes=stream_profile["n_classes"],
                transition_schedule=TransitionSchedule(schedule),
                gradual_width_events=stream_profile["gradual_width_events"],
                deletion_event=stream_profile["deletion_event"],
            )
            manifest = materialize_stream(stream_root, spec)
            for arm in ARMS:
                checkpoint_path = work_root / "checkpoints" / f"seed_{seed}" / schedule / f"{arm}.json"
                result = run_smoke_arm(
                    stream_root=stream_root,
                    spec=spec,
                    arm=arm,
                    profile=smoke_profile,
                    checkpoint_path=checkpoint_path,
                )
                key = f"seed_{seed}/{schedule}/{arm}"
                cells[key] = {
                    "seed": seed,
                    "schedule": schedule,
                    "arm": arm,
                    "stream_identity_sha256": spec.identity_sha256,
                    "stream_sha256": manifest["stream_sha256"],
                    "checkpoint_sha256": _sha256_file(checkpoint_path),
                    "state_sha256": result["state_sha256"],
                    "metrics": result["metrics"],
                    "controls": result["controls"],
                    "all_mechanics_ok": result["all_mechanics_ok"],
                    "resumed_from_atomic_checkpoint": result["resumed_from_atomic_checkpoint"],
                }
    progress = {
        "schema": "mop-continual-progressive-rung-progress/v1",
        "identity": identity,
        "identity_sha256": _canonical_sha256(identity),
        "cells": cells,
        "complete": True,
    }
    progress_path = work_root / "progress.json"
    _write_json(progress_path, progress)
    source_receipt = _self_hash(
        {
            "schema": "mop-continual-progressive-rung/v1",
            "claim_scope": CLAIM_SCOPE,
            "identity": identity,
            "identity_sha256": _canonical_sha256(identity),
            "source_live_authority": source_authority,
            "mode": "replication",
            "rung": rung,
            "plan": plan,
            "progress": {
                "path": str(progress_path.relative_to(tmp_path)),
                "sha256": _sha256_file(progress_path),
                "resumed_existing_progress": False,
                "completed_cells": 30,
                "expected_cells": 30,
            },
            "cells": cells,
            "resource_measurement": {
                "max_rss_bytes": 100_000_000,
                "measured_after_complete": True,
            },
            "all_mechanics_ok": True,
            "replication_execution_complete": True,
            "independent_metric_verifier_complete": False,
            "scientific_promotion": False,
            "claim_boundary": "execution mechanics only",
        }
    )
    source_path = tmp_path / f"proof/P6_CONTINUAL_{rung}.json"
    _write_json(source_path, source_receipt)
    return source_path


@pytest.fixture(scope="module")
def base_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _fixture_repo(tmp_path_factory.mktemp("continual-verifier-valid"))


def _copy_source(base_source: Path, destination: Path) -> Path:
    root = base_source.parents[1]
    shutil.copytree(root, destination, dirs_exist_ok=True)
    return destination / base_source.relative_to(root)


def _work_root(source: Path) -> tuple[Path, dict[str, Any]]:
    payload = json.loads(source.read_text())
    root = source.parents[1]
    return root / payload["progress"]["path"].rsplit("/", 1)[0], payload


def _reseal_checkpoint_source(source: Path, *, key: str, checkpoint: dict[str, Any]) -> None:
    work_root, payload = _work_root(source)
    row = payload["cells"][key]
    checkpoint_path = (
        work_root / "checkpoints" / f"seed_{row['seed']}" / row["schedule"] / f"{row['arm']}.json"
    )
    _write_json(checkpoint_path, checkpoint)
    result = checkpoint["result"]
    row.update(
        {
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "state_sha256": checkpoint["state_sha256"],
            "metrics": result["metrics"],
            "controls": result["controls"],
            "all_mechanics_ok": result["all_mechanics_ok"],
            "resumed_from_atomic_checkpoint": result["resumed_from_atomic_checkpoint"],
        }
    )
    root = source.parents[1]
    progress_path = root / payload["progress"]["path"]
    progress = json.loads(progress_path.read_text())
    progress["cells"] = copy.deepcopy(payload["cells"])
    _write_json(progress_path, progress)
    payload["progress"]["sha256"] = _sha256_file(progress_path)
    payload["cells"] = copy.deepcopy(progress["cells"])
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = _canonical_sha256(payload)
    _write_json(source, payload)


def _reseal_checkpoint_cadence_plan(source: Path, *, checkpoint_every: int) -> None:
    work_root, payload = _work_root(source)
    payload["plan"]["profile"]["checkpoint_every"] = checkpoint_every
    payload["identity"]["plan"] = copy.deepcopy(payload["plan"])
    payload["identity_sha256"] = _canonical_sha256(payload["identity"])
    for row in payload["cells"].values():
        checkpoint_path = (
            work_root / "checkpoints" / f"seed_{row['seed']}" / row["schedule"] / f"{row['arm']}.json"
        )
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint["identity"]["profile"] = copy.deepcopy(payload["plan"]["profile"])
        checkpoint["identity_sha256"] = _canonical_sha256(checkpoint["identity"])
        _write_json(checkpoint_path, checkpoint)
        row["checkpoint_sha256"] = _sha256_file(checkpoint_path)

    root = source.parents[1]
    progress_path = root / payload["progress"]["path"]
    progress = json.loads(progress_path.read_text())
    progress["identity"] = copy.deepcopy(payload["identity"])
    progress["identity_sha256"] = payload["identity_sha256"]
    progress["cells"] = copy.deepcopy(payload["cells"])
    _write_json(progress_path, progress)
    payload["progress"]["sha256"] = _sha256_file(progress_path)
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = _canonical_sha256(payload)
    _write_json(source, payload)


def _rewrite_stream_with_valid_chain_wrong_cue(stream_root: Path) -> None:
    manifest_path = stream_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    identity = str(manifest["identity_sha256"])
    previous = bytes(32)
    changed = False
    for row in manifest["chunks"]:
        path = stream_root / row["path"]
        raw = path.read_bytes()
        rewritten = bytearray()
        row["chain_start_sha256"] = previous.hex()
        for offset in range(int(row["count"])):
            unpacked = RECORD.unpack(raw[offset * RECORD.size : (offset + 1) * RECORD.size])
            fields = [int(value) for value in unpacked[:7]]
            if not changed:
                fields[4] = (fields[4] + 1) % int(manifest["spec"]["n_classes"])
                active_domain = fields[2] if fields[6] & 1 else fields[1]
                fields[5] = (fields[4] + active_domain) % int(manifest["spec"]["n_classes"])
                changed = True
            digest = hashlib.sha256(bytes.fromhex(identity) + previous + RECORD_CORE.pack(*fields)).digest()
            rewritten.extend(RECORD.pack(*fields, digest))
            previous = digest
        raw = bytes(rewritten)
        path.write_bytes(raw)
        row["bytes"] = len(raw)
        row["sha256"] = hashlib.sha256(raw).hexdigest()
        row["chain_head_sha256"] = previous.hex()
    manifest["chain_head_sha256"] = previous.hex()
    manifest["stream_sha256"] = _stream_composite_sha(manifest)
    _write_json(manifest_path, manifest)


def test_full_rung_verifier_recomputes_metrics_controls_and_mutation_suite(tmp_path: Path, base_source: Path):
    source = _copy_source(base_source, tmp_path)
    receipt = build_verification_receipt(source, repo_root=tmp_path)

    assert receipt["schema"] == VERIFIER_SCHEMA
    assert receipt["verification_complete"] is True
    assert receipt["errors"] == []
    assert receipt["independent_recompute"]["cell_count"] == 30
    assert receipt["independent_recompute"]["checkpoint_state_recomputed"] is True
    assert receipt["independent_recompute"]["decision"]["verdict"] == "null"
    assert receipt["mutation_suite"]["count"] == 12
    assert receipt["mutation_suite"]["rejected"] == 12
    assert receipt["mutation_suite"]["all_rejected"] is True
    assert all(receipt["checks"].values())
    assert receipt["prerequisite"] == {
        "source_rung": 10_000,
        "source_rung_file_sha256": _sha256_file(source),
        "source_identity_sha256": json.loads(source.read_text())["identity_sha256"],
        "verification_complete": True,
        "valid_controls": True,
        "tie_is_null": True,
        "mutation_suite_all_rejected": True,
        "next_rung": 100_000,
        "next_rung_allowed": False,
        "next_rung_reason": "verified tie, null, invalid evidence, or final rung does not admit scaling",
    }
    assert receipt["scientific_promotion"] is False


def test_one_tied_seed_is_a_null_even_when_paired_means_are_positive():
    seeds = [101, 102, 103, 104, 105]
    cells: dict[str, dict[str, Any]] = {}
    for schedule in SCHEDULES:
        for seed in seeds:
            replay_value = 0.5 if seed == seeds[0] else 1.0
            for arm, value in (("replay", replay_value), ("no-replay", 0.5), ("fresh-init", 0.25)):
                cells[f"seed_{seed}/{schedule}/{arm}"] = {
                    "metrics": {
                        "retention": {"domain_zero_final_accuracy": value},
                        "future_learnability": {"first_window_accuracy": value},
                    }
                }
    decision = _decision(cells, {"seeds": seeds})

    assert decision["verdict"] == "null"
    assert decision["null_supported"] is True
    assert decision["aggregate_tie_count"] == 2
    assert decision["tie_rule"] == TIE_RULE
    tied_pairs = [
        pair
        for contrast in decision["contrasts"]
        for pair in contrast["paired_seed_deltas"]
        if pair["seed"] == seeds[0] and contrast["control"] == "no-replay"
    ]
    assert tied_pairs and all(pair["tie_is_null"] for pair in tied_pairs)
    assert decision["strict_joint_gain_all_schedules_and_controls"] is False


def test_one_negative_seed_endpoint_is_a_null_even_when_paired_means_are_positive():
    seeds = [101, 102, 103, 104, 105]
    cells: dict[str, dict[str, Any]] = {}
    for schedule in SCHEDULES:
        for seed in seeds:
            replay_retention = 0.4 if seed == seeds[0] else 1.0
            for arm, retention in (
                ("replay", replay_retention),
                ("no-replay", 0.5),
                ("fresh-init", 0.25),
            ):
                cells[f"seed_{seed}/{schedule}/{arm}"] = {
                    "metrics": {
                        "retention": {"domain_zero_final_accuracy": retention},
                        "future_learnability": {"first_window_accuracy": 1.0 if arm == "replay" else 0.5},
                    }
                }
    decision = _decision(cells, {"seeds": seeds})

    assert decision["verdict"] == "null"
    negative_pairs = [
        pair
        for contrast in decision["contrasts"]
        for pair in contrast["paired_seed_deltas"]
        if pair["seed"] == seeds[0] and contrast["control"] == "no-replay"
    ]
    assert negative_pairs and all(pair["retention_delta"] < 0 for pair in negative_pairs)
    assert all(pair["tie_is_null"] is False for pair in negative_pairs)
    assert all(pair["nonpositive_is_null"] is True for pair in negative_pairs)


def test_final_one_million_rung_never_authorizes_a_next_rung_even_if_favorable():
    authority = _next_rung_authority(
        rung=1_000_000,
        source_file_sha256="a" * 64,
        source_identity_sha256="b" * 64,
        verification_complete=True,
        valid_controls=True,
        mutations_all_rejected=True,
        favorable=True,
    )

    assert authority["next_rung"] is None
    assert authority["next_rung_allowed"] is False
    assert authority["next_rung_reason"] == (
        "verified tie, null, invalid evidence, or final rung does not admit scaling"
    )


def test_verifier_receipt_is_payload_sealed_and_bound_to_source(tmp_path: Path, base_source: Path):
    source = _copy_source(base_source, tmp_path)
    output = tmp_path / "proof/P6_CONTINUAL_10K_INDEPENDENT_VERIFICATION.json"
    receipt = write_verification_receipt(source, output, repo_root=tmp_path)

    assert output.is_file()
    assert json.loads(output.read_text()) == receipt
    assert receipt["verification_complete"] is True
    assert receipt["prerequisite"]["source_rung"] == 10_000
    assert receipt["prerequisite"]["next_rung"] == 100_000
    assert receipt["prerequisite"]["next_rung_allowed"] is False
    core = dict(receipt)
    declared = core.pop("payload_sha256")
    assert declared == _canonical_sha256(core)


def test_verifier_rejects_checkpoint_metric_and_live_dependency_mutations(tmp_path: Path, base_source: Path):
    source = _copy_source(base_source, tmp_path / "metric")
    payload = json.loads(source.read_text())
    first_key = sorted(payload["cells"])[0]
    payload["cells"][first_key]["metrics"]["retention"]["domain_zero_final_accuracy"] = 0.125
    payload["payload_sha256"] = _canonical_sha256(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )
    _write_json(source, payload)
    metric_mutation = build_verification_receipt(source, repo_root=tmp_path / "metric")
    assert metric_mutation["verification_complete"] is False
    assert any(
        "progress authority" in error or "independently recomputed" in error
        for error in metric_mutation["errors"]
    )

    source = _copy_source(base_source, tmp_path / "dependency")
    runner = tmp_path / "dependency/scripts/continual_million_event_rung.py"
    runner.write_text("# drifted runner\n")
    dependency_mutation = build_verification_receipt(source, repo_root=tmp_path / "dependency")
    assert dependency_mutation["verification_complete"] is False
    assert any("runner_sha256" in error for error in dependency_mutation["errors"])


def test_verifier_rejects_fully_resealed_plan_drift_before_raw_replay(
    tmp_path: Path,
    base_source: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _copy_source(base_source, tmp_path)
    _reseal_checkpoint_cadence_plan(source, checkpoint_every=1)

    def unexpected_replay(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("raw replay must not begin for a noncanonical plan")

    monkeypatch.setattr(
        "mop.studies.continual_million_event_verify._replay_stream",
        unexpected_replay,
    )
    receipt = build_verification_receipt(source, repo_root=tmp_path)

    assert receipt["verification_complete"] is False
    assert receipt["independent_recompute"]["cell_count"] == 0
    assert any("exact live-config plan drift" in error for error in receipt["errors"])


def test_verifier_rejects_short_stream_bytes(tmp_path: Path, base_source: Path):
    source = _copy_source(base_source, tmp_path)
    work_root, payload = _work_root(source)
    seed = payload["plan"]["seeds"][0]
    stream_root = work_root / "streams" / f"seed_{seed}" / "abrupt"
    chunk = stream_root / "chunk_000000.bin"
    chunk.write_bytes(chunk.read_bytes()[:-1])

    receipt = build_verification_receipt(source, repo_root=tmp_path)

    assert receipt["verification_complete"] is False
    assert any("fixed-width byte count drift" in error for error in receipt["errors"])


def test_verifier_rejects_valid_digest_chain_with_wrong_deterministic_fields(
    tmp_path: Path, base_source: Path
):
    source = _copy_source(base_source, tmp_path)
    work_root, payload = _work_root(source)
    seed = payload["plan"]["seeds"][0]
    stream_root = work_root / "streams" / f"seed_{seed}" / "abrupt"
    _rewrite_stream_with_valid_chain_wrong_cue(stream_root)

    _, problems = _replay_stream(
        stream_root,
        plan=payload["plan"],
        seed=seed,
        schedule="abrupt",
    )

    assert any("deterministic record field drift" in problem for problem in problems)
    assert not any("record digest chain drift" in problem for problem in problems)


def test_verifier_rejects_fully_resealed_invented_favorable_state(tmp_path: Path, base_source: Path):
    source = _copy_source(base_source, tmp_path)
    work_root, payload = _work_root(source)
    key = next(key for key in sorted(payload["cells"]) if key.endswith("/replay"))
    row = payload["cells"][key]
    checkpoint_path = (
        work_root / "checkpoints" / f"seed_{row['seed']}" / row["schedule"] / f"{row['arm']}.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text())
    state = checkpoint["state"]
    state["counts"] = [[10_000 if cue == label else 0 for label in range(4)] for cue in range(4)]
    state["correct"] = state["total"]
    state["domain_correct"] = list(state["domain_total"])
    state["future_outcomes"] = [1] * 48
    state["future_events_to_threshold"] = 16
    state["future_rolling"] = [1] * 16
    metrics, controls, problems = _recompute_cell_metrics(
        state,
        arm=row["arm"],
        plan=payload["plan"],
        stream_disk_bytes=checkpoint["result"]["metrics"]["resources"]["stream_disk_bytes"],
        lifecycle_deleted=True,
    )
    assert problems == []
    checkpoint["state_sha256"] = _canonical_sha256(state)
    checkpoint["result"]["state_sha256"] = checkpoint["state_sha256"]
    checkpoint["result"]["metrics"] = metrics
    checkpoint["result"]["controls"] = controls
    _reseal_checkpoint_source(source, key=key, checkpoint=checkpoint)

    receipt = build_verification_receipt(source, repo_root=tmp_path)

    assert receipt["verification_complete"] is False
    assert any("state differs from independent raw-event replay" in error for error in receipt["errors"])


def test_verifier_rejects_fully_resealed_lifecycle_drift(tmp_path: Path, base_source: Path):
    source = _copy_source(base_source, tmp_path)
    work_root, payload = _work_root(source)
    key = sorted(payload["cells"])[0]
    row = payload["cells"][key]
    checkpoint_path = (
        work_root / "checkpoints" / f"seed_{row['seed']}" / row["schedule"] / f"{row['arm']}.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text())
    lifecycle = checkpoint["result"]["lifecycle"]
    lifecycle["entries"][0]["reason"] = "fabricated but hash-consistent reason"
    previous: str | None = None
    for entry in lifecycle["entries"]:
        entry["previous_entry_sha256"] = previous
        entry.pop("entry_sha256", None)
        entry["entry_sha256"] = _canonical_sha256(entry)
        previous = entry["entry_sha256"]
    lifecycle["head_sha256"] = previous
    checkpoint["result"]["lifecycle_sha256"] = _canonical_sha256(lifecycle)
    _reseal_checkpoint_source(source, key=key, checkpoint=checkpoint)

    receipt = build_verification_receipt(source, repo_root=tmp_path)

    assert receipt["verification_complete"] is False
    assert any("result differs from independent full recompute" in error for error in receipt["errors"])
