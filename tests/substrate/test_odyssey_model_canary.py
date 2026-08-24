"""Unit coverage for the inert, public-only Odyssey model screen."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from substrate import odyssey_model_canary as canary


def _template() -> dict:
    return {
        "candidate_aliases": ["one", "two", "three"],
        "visibility": "public_only",
        "model_service_cap_bytes": 24 * canary.GIB,
        "required_concurrent_clients": 8,
        "selection_rule": "highest_public_canary_score_then_lower_service_peak_then_lower_median_latency_then_lexical_weight_digest",
        "case_set": [
            {"id": "A1", "frontier": "A", "seed": 1, "prompt": "one", "answer": "one"},
            {"id": "A2", "frontier": "A", "seed": 2, "prompt": "two", "answer": "two"},
        ],
    }


def _base(name: str) -> dict[str, str]:
    return {
        "id": name,
        "revision": f"revision-{name}",
        "weight_sha256": canary.digest({"weight": name}),
        "tokenizer_sha256": canary.digest({"tokenizer": name}),
        "runtime_sha256": canary.digest({"runtime": "fixture"}),
        "quantization": "Q4_K_M",
    }


def _candidate(name: str, *, eligible: bool, score: int, service_peak: int = 10, latency: float = 1.0) -> dict:
    return {
        "base_model": _base(name),
        "model_size_bytes": service_peak,
        "service_peak_bytes": service_peak,
        "swap_pageout_delta_bytes": 0,
        "width_eight": {"requests": 8, "completed": 8, "all_responses_valid": True},
        "canary": {"total": 2, "passed": score, "median_latency_ms": latency, "case_results": []},
        "errors": [] if eligible else ["fixture unavailable"],
        "eligible": eligible,
    }


def test_pageout_counter_is_normalized_to_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        canary.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["vm_stat"],
            returncode=0,
            stdout="Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPageouts: 7.\n",
            stderr="",
        ),
    )
    assert canary._pageout_bytes() == 7 * 16384


def test_winner_uses_score_then_service_then_latency_then_digest() -> None:
    rows = [
        _candidate("one", eligible=True, score=2, service_peak=11, latency=1.0),
        _candidate("two", eligible=True, score=2, service_peak=10, latency=9.0),
        _candidate("three", eligible=True, score=1, service_peak=1, latency=0.1),
    ]
    assert canary._winner(rows)["base_model"]["id"] == "two"


def test_run_refuses_a_subset_of_the_frozen_candidate_cohort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template = _template()
    monkeypatch.setattr(canary, "_template", lambda _root: ({"sha256": "f" * 64}, template))
    with pytest.raises(canary.Refused, match="exact frozen candidate order"):
        canary.run(tmp_path, tmp_path / "receipt.json", ["one"])


def test_run_fails_closed_when_any_configured_candidate_is_ineligible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    template = _template()
    monkeypatch.setattr(canary, "_template", lambda _root: ({"sha256": "f" * 64}, template))
    monkeypatch.setattr(canary, "file_digest", lambda _path: "t" * 64)
    monkeypatch.setattr(canary, "_runtime", lambda: {"id": "ollama", "version": "fixture", "sha256": canary.digest({"runtime": "fixture"})})
    rows = {
        "one": _candidate("one", eligible=True, score=2),
        "two": _candidate("two", eligible=False, score=0),
        "three": _candidate("three", eligible=True, score=1),
    }
    monkeypatch.setattr(canary, "_candidate", lambda name, **_kwargs: rows[name])
    receipt = canary.run(tmp_path, tmp_path / "receipt.json", None)
    assert receipt["all_pass"] is False
    assert receipt["checks"]["all_configured_candidates_eligible"] is False
    assert receipt["selected_base_model"]["id"] == "one"
    assert (tmp_path / "receipt.json").is_file()
