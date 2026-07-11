from __future__ import annotations

import json
from pathlib import Path

import pytest
import scripts.continual_million_event_rung as rung_runner
from scripts.continual_million_event_rung import (
    PROGRESS_SCHEMA,
    RESULT_SCHEMA,
    _progress,
    _revalidate_live_identity,
    _sha256_file,
    _update_peak_rss,
    _validate_source_live_bindings,
    build_plan,
    load_config,
    run_rung,
)

from mop.substrate.events import canonical_sha256


def test_progressive_config_supports_exact_three_rungs_and_full_replication_matrix():
    config = load_config()
    assert config["null_hypothesis"].startswith("replay does not improve preregistered retention")
    assert config["replication"]["rungs"] == [10_000, 100_000, 1_000_000]
    for rung in config["replication"]["rungs"]:
        plan = build_plan(config, rung)
        assert plan["expected_cells"] == 2 * 3 * 5
        assert plan["schedules"] == ["abrupt", "gradual"]
        assert plan["arms"] == ["replay", "no-replay", "fresh-init"]
        assert len(plan["seeds"]) == 5
        assert plan["profile"] == {
            "checkpoint_every": 1250,
            "replay_capacity": 128,
            "future_window_events": 48,
            "threshold_window_events": 16,
            "future_accuracy_threshold": 0.5,
            "matched_updates_per_event": 2,
        }


def test_resource_probe_is_only_the_first_canonical_10k_cell():
    config = load_config()
    plan = build_plan(
        config,
        10_000,
        resource_probe=True,
        seed_count=1,
        schedules=("abrupt",),
        arms=("replay",),
    )
    assert plan["mode"] == "resource-probe"
    assert plan["cells"] == [{"seed": 20260710, "schedule": "abrupt", "arm": "replay"}]
    with pytest.raises(ValueError, match="exactly 10k"):
        build_plan(
            config,
            100_000,
            resource_probe=True,
            seed_count=1,
            schedules=("abrupt",),
            arms=("replay",),
        )


def test_full_rung_refuses_underreplication_or_missing_control():
    config = load_config()
    with pytest.raises(ValueError, match="at least five seeds"):
        build_plan(config, 10_000, seed_count=4)
    with pytest.raises(ValueError, match="every schedule and arm"):
        build_plan(config, 10_000, arms=("replay", "no-replay"))


def test_progress_receipt_resumes_only_under_exact_identity(tmp_path):
    path = tmp_path / "progress.json"
    identity = {"rung": 10_000, "mode": "resource-probe"}
    first, resumed = _progress(path, identity)
    assert resumed is False
    assert first["schema"] == PROGRESS_SCHEMA
    first["cells"]["seed_20260710/abrupt/replay"] = {"all_mechanics_ok": True}
    path.write_text(json.dumps(first))
    second, resumed = _progress(path, identity)
    assert resumed is True
    assert second["cells"] == first["cells"]
    with pytest.raises(ValueError, match="identity drift"):
        _progress(path, {"rung": 100_000, "mode": "replication"})


def test_progress_persists_maximum_rss_across_resumes(tmp_path):
    path = tmp_path / "progress.json"
    identity = {"rung": 10_000, "mode": "resource-probe"}
    progress, _ = _progress(path, identity)
    assert _update_peak_rss(progress, observed_bytes=300_000_000) == 300_000_000
    path.write_text(json.dumps(progress))

    resumed, resumed_existing = _progress(path, identity)
    assert resumed_existing is True
    assert _update_peak_rss(resumed, observed_bytes=100_000_000) == 300_000_000
    assert _update_peak_rss(resumed, observed_bytes=400_000_000) == 400_000_000
    assert resumed["resource_measurement"] == {
        "max_rss_bytes": 400_000_000,
        "rss_scope": "maximum runner-process RSS across persisted invocations",
    }


def test_resumed_rung_publishes_persisted_peak_not_finalization_only_rss(tmp_path, monkeypatch):
    plan = {
        "mode": "resource-probe",
        "rung": 10_000,
        "seeds": [7],
        "schedules": ["abrupt"],
        "arms": ["replay"],
        "cells": [{"seed": 7, "schedule": "abrupt", "arm": "replay"}],
        "expected_cells": 1,
        "stream": {
            "chunk_events": 100,
            "n_domains": 4,
            "n_classes": 4,
            "gradual_width_events": 833,
            "deletion_event": 3_750,
        },
        "profile": {
            "checkpoint_every": 1_250,
            "replay_capacity": 128,
            "future_window_events": 48,
            "threshold_window_events": 16,
            "future_accuracy_threshold": 0.5,
            "matched_updates_per_event": 2,
        },
    }
    source_authority = {"bindings_sha256": "b" * 64}
    config = {
        "_config_sha256": "a" * 64,
        "_source_preflight_payload_sha256": "c" * 64,
        "_source_live_authority": source_authority,
        "source_preflight": {"file_sha256": "d" * 64},
        "replication": {
            "minimum_independent_seeds": 5,
            "schedules": ["abrupt", "gradual"],
            "arms": ["replay", "no-replay", "fresh-init"],
        },
    }
    identity = {
        "config_sha256": config["_config_sha256"],
        "runner_sha256": _sha256_file(Path(rung_runner.__file__).resolve()),
        "source_preflight_file_sha256": config["source_preflight"]["file_sha256"],
        "source_preflight_payload_sha256": config["_source_preflight_payload_sha256"],
        "source_live_bindings_sha256": source_authority["bindings_sha256"],
        "plan": plan,
        "claim_scope": rung_runner.CLAIM_SCOPE,
    }
    work_root = tmp_path / "runs/rung"
    progress_path = work_root / "progress.json"
    progress, _ = _progress(progress_path, identity)
    _update_peak_rss(progress, observed_bytes=300_000_000)
    progress_path.write_text(json.dumps(progress))
    result = {
        "stream_identity_sha256": "e" * 64,
        "stream_sha256": "f" * 64,
        "checkpoint_sha256": "1" * 64,
        "state_sha256": "2" * 64,
        "metrics": {},
        "controls": {},
        "all_mechanics_ok": True,
        "resumed_from_atomic_checkpoint": False,
    }

    monkeypatch.setattr(rung_runner, "load_config", lambda _path: config)
    monkeypatch.setattr(rung_runner, "build_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(rung_runner, "_revalidate_live_identity", lambda *_args: None)
    monkeypatch.setattr(rung_runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rung_runner, "materialize_stream", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(rung_runner, "verify_stream", lambda *_args, **_kwargs: {"verified": True})
    monkeypatch.setattr(rung_runner, "run_smoke_arm", lambda **_kwargs: result)
    monkeypatch.setattr(rung_runner, "_max_rss_bytes", lambda: 100_000_000)

    receipt = run_rung(
        tmp_path / "config.yaml",
        work_root,
        tmp_path / "proof/result.json",
        rung=10_000,
        resource_probe=True,
        seed_count=1,
        schedules=("abrupt",),
        arms=("replay",),
    )

    assert receipt["resource_measurement"]["current_invocation_max_rss_bytes"] == 100_000_000
    assert receipt["resource_measurement"]["max_rss_bytes"] == 300_000_000
    persisted = json.loads(progress_path.read_text())
    assert persisted["resource_measurement"]["max_rss_bytes"] == 300_000_000
    sealed = dict(receipt)
    declared = sealed.pop("payload_sha256")
    assert declared == canonical_sha256(sealed)


def test_existing_output_never_bypasses_live_progress_and_checkpoint_revalidation(tmp_path, monkeypatch):
    plan = {
        "mode": "resource-probe",
        "rung": 10_000,
        "seeds": [7],
        "schedules": ["abrupt"],
        "arms": ["replay"],
        "cells": [{"seed": 7, "schedule": "abrupt", "arm": "replay"}],
        "expected_cells": 1,
        "stream": {
            "chunk_events": 100,
            "n_domains": 4,
            "n_classes": 4,
            "gradual_width_events": 833,
            "deletion_event": 3_750,
        },
        "profile": {
            "checkpoint_every": 1_250,
            "replay_capacity": 128,
            "future_window_events": 48,
            "threshold_window_events": 16,
            "future_accuracy_threshold": 0.5,
            "matched_updates_per_event": 2,
        },
    }
    source_authority = {"bindings_sha256": "b" * 64}
    config = {
        "_config_sha256": "a" * 64,
        "_source_preflight_payload_sha256": "c" * 64,
        "_source_live_authority": source_authority,
        "source_preflight": {"file_sha256": "d" * 64},
        "replication": {
            "minimum_independent_seeds": 5,
            "schedules": ["abrupt", "gradual"],
            "arms": ["replay", "no-replay", "fresh-init"],
        },
    }
    identity = {
        "config_sha256": config["_config_sha256"],
        "runner_sha256": _sha256_file(Path(rung_runner.__file__).resolve()),
        "source_preflight_file_sha256": config["source_preflight"]["file_sha256"],
        "source_preflight_payload_sha256": config["_source_preflight_payload_sha256"],
        "source_live_bindings_sha256": source_authority["bindings_sha256"],
        "plan": plan,
        "claim_scope": rung_runner.CLAIM_SCOPE,
    }
    work_root = tmp_path / "runs/rung"
    progress_path = work_root / "progress.json"
    progress, _ = _progress(progress_path, identity)
    progress["cells"]["seed_7/abrupt/replay"] = {"all_mechanics_ok": True}
    progress["complete"] = True
    progress_path.write_text(json.dumps(progress))
    stream_root = work_root / "streams/seed_7/abrupt"
    stream_root.mkdir(parents=True)
    checkpoint = work_root / "checkpoints/seed_7/abrupt/replay.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}")
    output = tmp_path / "proof/result.json"
    output.parent.mkdir(parents=True)
    existing = {
        "schema": RESULT_SCHEMA,
        "identity": identity,
        "identity_sha256": canonical_sha256(identity),
        "source_live_authority": source_authority,
        "all_mechanics_ok": True,
    }
    existing["payload_sha256"] = canonical_sha256(existing)
    output.write_text(json.dumps(existing))

    monkeypatch.setattr(rung_runner, "load_config", lambda _path: config)
    monkeypatch.setattr(rung_runner, "build_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(rung_runner, "_revalidate_live_identity", lambda *_args: None)
    monkeypatch.setattr(rung_runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rung_runner, "verify_stream", lambda *_args, **_kwargs: {"verified": True})

    def reject_stale_checkpoint(**_kwargs):
        raise ValueError("stale checkpoint reached")

    monkeypatch.setattr(rung_runner, "run_smoke_arm", reject_stale_checkpoint)
    with pytest.raises(ValueError, match="stale checkpoint reached"):
        run_rung(
            tmp_path / "config.yaml",
            work_root,
            output,
            rung=10_000,
            resource_probe=True,
            seed_count=1,
            schedules=("abrupt",),
            arms=("replay",),
        )


def test_source_preflight_live_bindings_are_revalidated_before_resume(tmp_path):
    config_path = tmp_path / "configs/preflight.yaml"
    implementation_path = tmp_path / "src/continual.py"
    wave_path = tmp_path / "proof/wave.json"
    config_path.parent.mkdir(parents=True)
    implementation_path.parent.mkdir(parents=True)
    wave_path.parent.mkdir(parents=True)
    config_payload = {"schema": "fixture/v1", "value": 1}
    config_path.write_text("schema: fixture/v1\nvalue: 1\n")
    implementation_path.write_text("VALUE = 1\n")
    wave_path.write_text("{}\n")
    receipt = {
        "config": {
            "path": "configs/preflight.yaml",
            "sha256": _sha256_file(config_path),
            "payload": config_payload,
            "profile_sha256": canonical_sha256(config_payload),
        },
        "implementation": [
            {"path": "configs/preflight.yaml", "sha256": _sha256_file(config_path)},
            {"path": "src/continual.py", "sha256": _sha256_file(implementation_path)},
        ],
        "wave_e0": {"path": "proof/wave.json", "sha256": _sha256_file(wave_path)},
    }

    authority = _validate_source_live_bindings(receipt, repo_root=tmp_path)
    assert authority["bindings_sha256"] == canonical_sha256(
        {key: value for key, value in authority.items() if key != "bindings_sha256"}
    )

    implementation_path.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="live implementation drift"):
        _validate_source_live_bindings(receipt, repo_root=tmp_path)


def test_loaded_rung_config_carries_validated_preflight_authority_into_identity():
    config = load_config()
    authority = config["_source_live_authority"]
    assert config["_source_preflight_payload_sha256"] == config["source_preflight"]["payload_sha256"]
    assert authority["bindings_sha256"] == canonical_sha256(
        {key: value for key, value in authority.items() if key != "bindings_sha256"}
    )
    assert authority["config"]["path"] == "configs/experiment/continual_million_event_preflight.yaml"

    plan = build_plan(
        config,
        10_000,
        resource_probe=True,
        seed_count=1,
        schedules=("abrupt",),
        arms=("replay",),
    )
    identity = {
        "config_sha256": config["_config_sha256"],
        "runner_sha256": _sha256_file(Path(rung_runner.__file__).resolve()),
        "source_preflight_file_sha256": config["source_preflight"]["file_sha256"],
        "source_preflight_payload_sha256": config["_source_preflight_payload_sha256"],
        "source_live_bindings_sha256": authority["bindings_sha256"],
        "plan": plan,
        "claim_scope": config["claim_scope"],
    }
    _revalidate_live_identity(config, identity)
    identity["source_live_bindings_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="live source identity drift"):
        _revalidate_live_identity(config, identity)
