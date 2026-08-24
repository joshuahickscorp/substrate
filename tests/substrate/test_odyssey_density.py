"""Density contract tests: gateway pin, parity, scheduling, preprocess, receipts."""

from __future__ import annotations

from pathlib import Path

import pytest

from substrate import odyssey_density as density
from substrate.odyssey_tools import assert_budget_parity, budget_for_frontier


def test_pinned_parallel_slots_is_positive_int() -> None:
    assert density.PINNED_OLLAMA_NUM_PARALLEL == 8
    assert density.OLLAMA_NUM_PARALLEL_ENV == "OLLAMA_NUM_PARALLEL"
    assert f"={density.PINNED_OLLAMA_NUM_PARALLEL}" in density.GATEWAY_REVISION


def test_gateway_pin_document_closed_shape() -> None:
    pin = density.gateway_pin_document(artifact_sha256="a" * 64)
    assert set(pin) == {"id", "revision", "artifact_sha256", "stateless"}
    assert pin["stateless"] is True
    assert pin["revision"] == density.GATEWAY_REVISION


def test_gateway_runtime_identity_binds_parallel_pin() -> None:
    identity = density.gateway_runtime_identity(cli="ollama version x", api_version="0.32.1")
    assert identity[density.OLLAMA_NUM_PARALLEL_ENV] == density.PINNED_OLLAMA_NUM_PARALLEL
    assert "cli" in identity and "api" in identity


def test_assert_pin_matches_live_service_or_skips() -> None:
    """On this host the live service is expected to match; otherwise refuse cleanly."""
    observed = density.read_running_ollama_num_parallel()
    if observed is None:
        with pytest.raises(density.DensityRefused, match="cannot verify"):
            density.assert_ollama_num_parallel_pinned(require_running=True)
        status = density.assert_ollama_num_parallel_pinned(require_running=False)
        assert status["status"] == "service_unobserved"
        return
    if observed == density.PINNED_OLLAMA_NUM_PARALLEL:
        status = density.assert_ollama_num_parallel_pinned(require_running=True)
        assert status["matched"] is True
        assert status["pinned"] == 8
    else:
        with pytest.raises(density.DensityRefused, match="does not match"):
            density.assert_ollama_num_parallel_pinned(require_running=True)


def test_resource_classes_cover_all_frontiers_and_pair_parity() -> None:
    assert set(density.FRONTIER_RESOURCE_CLASS) == set("ABCDEFGH")
    assert density.resource_class_for_frontier("B") == "light"
    assert density.resource_class_for_frontier("C") == "light"
    assert density.resource_class_for_frontier("E") == "light"
    assert density.resource_class_for_frontier("A") == "medium"
    assert density.resource_class_for_frontier("D") == "medium"
    assert density.resource_class_for_frontier("H") == "medium"
    assert density.resource_class_for_frontier("F") == "heavy"
    assert density.resource_class_for_frontier("G") == "heavy"
    for frontier in "ABCDEFGH":
        assert density.assert_pair_resource_parity(frontier, frontier) == density.resource_class_for_frontier(frontier)
        cand = budget_for_frontier(frontier)
        ctrl = budget_for_frontier(frontier)
        assert assert_budget_parity(cand, ctrl) == cand.budget_sha256()
    with pytest.raises(density.DensityRefused):
        density.assert_pair_resource_parity("A", "B")


def test_budget_for_frontier_differs_across_classes_not_within_pair() -> None:
    light = budget_for_frontier("B").to_dict()
    heavy = budget_for_frontier("G").to_dict()
    assert light != heavy
    assert budget_for_frontier("B").to_dict() == budget_for_frontier("C").to_dict()


def test_deadline_aware_order_keeps_pairs_atomic_and_prefers_heavy() -> None:
    now = 1000.0
    deadline = now + 1800.0
    ordered = density.order_frontier_entries_for_phase(
        list("ABCDEFGH"),
        phase_deadline_monotonic=deadline,
        checkpoint_criticality=1,
        now=now,
    )
    assert set(ordered) == set("ABCDEFGH")
    assert len(ordered) == 8
    # With shared deadline, less slack (heavier estimated runtime) sorts first.
    # Heavy F/G should appear before light B/C/E.
    heavy_positions = [ordered.index(f) for f in ("F", "G")]
    light_positions = [ordered.index(f) for f in ("B", "C", "E")]
    assert max(heavy_positions) < min(light_positions)
    # Deterministic for equal inputs.
    again = density.order_frontier_entries_for_phase(
        list("ABCDEFGH"),
        phase_deadline_monotonic=deadline,
        checkpoint_criticality=1,
        now=now,
    )
    assert ordered == again


def test_shared_preprocess_is_byte_identical_for_both_arms(tmp_path: Path) -> None:
    store = density.SharedPreprocessStore(tmp_path / "preprocess")
    raw = b"%PDF-1.4 shared document bytes for both arms"
    a = density.preprocess_document_fingerprint(store, raw, producer="candidate-prep")
    b = density.preprocess_document_fingerprint(store, raw, producer="control-prep")
    assert a.output_sha256 == b.output_sha256
    assert a.byte_length == b.byte_length
    cand_path = tmp_path / "cand" / "doc.json"
    ctrl_path = tmp_path / "ctrl" / "doc.json"
    store.materialize_readonly(a.output_sha256, cand_path)
    store.materialize_readonly(b.output_sha256, ctrl_path)
    assert cand_path.read_bytes() == ctrl_path.read_bytes()
    assert cand_path.read_bytes() == store.get(a.output_sha256).read_bytes()
    stats = store.stats()
    assert stats["hits"] >= 1
    assert stats["reuse_rate"] > 0


def test_preprocess_rejects_unknown_recipe(tmp_path: Path) -> None:
    store = density.SharedPreprocessStore(tmp_path / "preprocess")
    with pytest.raises(density.DensityRefused, match="allowlist"):
        store.put_bytes(
            b"x",
            recipe_id="answer.precompute",
            input_sha256="0" * 64,
            media_type="text/plain",
            producer="evil",
            timing_seconds=0.0,
        )


def test_compact_receipt_has_no_payload() -> None:
    receipt = density.compact_artifact_receipt(
        digest_hex="ab" * 32,
        size=12,
        media_type="image/png",
        producer="three_d.render:blender",
        path="artifacts/out.png",
        timing_seconds=0.94,
        resource_use={"elapsed_seconds": 0.94},
    )
    assert "digest" in receipt and "size" in receipt and "type" in receipt
    assert "payload" not in receipt and "stdout" not in receipt and "bytes" not in receipt
    assert receipt["sha256"] == density.digest({k: v for k, v in receipt.items() if k != "sha256"})


def test_overlap_ledger_reports_cpu_gpu_overlap() -> None:
    ledger = density.OverlapLedger()
    # Model 0..10, tool 5..15 → 5s overlap.
    ledger.record(density.OverlapSample("A", "candidate", "model", 0.0, 10.0))
    ledger.record(density.OverlapSample("B", "control", "tool", 5.0, 15.0))
    summary = ledger.summary()
    assert summary["cpu_gpu_overlap_seconds"] == pytest.approx(5.0)
    assert summary["model_busy_seconds"] == pytest.approx(10.0)
    assert summary["tool_busy_seconds"] == pytest.approx(10.0)
    assert summary["gpu_idle_seconds_estimated"] == pytest.approx(5.0)


def test_warm_pool_policy_documents_process_per_op_tools() -> None:
    assert "lean" in density.PROCESS_PER_OP_TOOLS
    assert "blender" in density.PROCESS_PER_OP_TOOLS
    assert "z3" in density.WARM_CAPABLE_TOOLS
    assert "ffmpeg" in density.WARM_CAPABLE_TOOLS


def test_parity_proof_document() -> None:
    proof = density.parity_proof_for_frontiers()
    assert proof["candidate_control_profiles_byte_identical"] is True
    assert proof["pairs_never_split"] is True
    assert len(proof["rows"]) == 8


def test_density_metrics_bundle() -> None:
    metrics = density.DensityMetrics(
        useful_work_units=16,
        wall_seconds=60.0,
        phase_deadline_seconds=1800.0,
        phase_used_seconds=900.0,
        cache_hits=3,
        cache_misses=1,
        gateway_pin={"matched": True},
        parity_proof=density.parity_proof_for_frontiers(),
    )
    body = metrics.to_dict()
    assert body["useful_work_per_minute"] == pytest.approx(16.0)
    assert body["phase_deadline_utilization"] == pytest.approx(0.5)
    assert body["cache_reuse_rate"] == pytest.approx(0.75)
    assert body["pinned_ollama_num_parallel"] == 8
