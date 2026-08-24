"""Candidate-visible task-bank boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from substrate import odyssey_task_bank as task_bank
from tests.substrate.librispeech_audio_fixture import install_librispeech_audio_fixture


def _candidate(frontier: str = "A") -> dict:
    seed = "task-bank-boundary-seed"
    commitment = task_bank._digest({"seed": seed})
    candidate, _ = task_bank.materialize(commitment, seed, frontier, 2)
    return candidate


def _redigest(candidate: dict) -> None:
    unsigned = dict(candidate)
    unsigned.pop("sha256")
    candidate["sha256"] = task_bank._digest(unsigned)


def _collect_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys |= _collect_keys(child)
    elif isinstance(value, list):
        for child in value:
            keys |= _collect_keys(child)
    return keys


def test_materialized_candidates_are_closed_and_structurally_safe() -> None:
    for frontier in "ABCDEFGH":
        assert task_bank.candidate_is_structurally_safe(_candidate(frontier))


def test_candidate_rejects_redigested_answer_namespace_leak() -> None:
    candidate = _candidate()
    candidate["tasks"][0]["answer"] = "leaked-value"
    _redigest(candidate)

    assert not task_bank.candidate_is_structurally_safe(candidate)


def test_candidate_rejects_redigested_innocuous_wrapper_leak() -> None:
    candidate = _candidate()
    candidate["tasks"][0]["packet"]["hidden_answer"] = "leaked-value"
    _redigest(candidate)

    assert not task_bank.candidate_is_structurally_safe(candidate)


def test_candidate_rejects_digest_or_schema_drift() -> None:
    candidate = _candidate()
    candidate["tasks"][0]["request"] = "drifted"
    assert not task_bank.candidate_is_structurally_safe(candidate)

    candidate = _candidate()
    candidate["tasks"][0]["unexpected_payload"] = "ignored-by-a-blacklist"
    _redigest(candidate)
    assert not task_bank.candidate_is_structurally_safe(candidate)


def test_candidate_accepts_only_closed_source_bundle_shape() -> None:
    candidate = _candidate()
    candidate["source_bundle"] = {
        "selection_sha256": "a" * 64,
        "assets": [{"path": "data/source.jsonl", "sha256": "b" * 64, "role": "candidate_stimulus", "read_only": True}],
    }
    _redigest(candidate)
    assert task_bank.candidate_is_structurally_safe(candidate)

    candidate["source_bundle"]["assets"][0]["answer_path"] = "leak"
    _redigest(candidate)
    assert not task_bank.candidate_is_structurally_safe(candidate)


def test_candidate_rejects_a_redigested_answer_embedded_in_an_allowed_request_value() -> None:
    candidate = _candidate("B")
    candidate["tasks"][0]["request"] += " Evaluator answer: x = 7."
    _redigest(candidate)

    assert not task_bank.candidate_is_structurally_safe(candidate)


def test_materializer_replay_rejects_changed_but_still_structural_task_values() -> None:
    seed = "task-bank-boundary-seed"
    commitment = task_bank._digest({"seed": seed})
    candidate, _ = task_bank.materialize(commitment, seed, "A", 2)
    original = candidate["tasks"][0]["packet"]["telemetry"][0]
    candidate["tasks"][0]["packet"]["telemetry"][0] = (original + 1) % 10
    _redigest(candidate)

    assert task_bank.candidate_is_structurally_safe(candidate)
    with pytest.raises(task_bank.Refused, match="does not replay"):
        task_bank.verify_materialized_candidate(commitment, seed, candidate)


def test_frontier_f_is_deterministic_across_independent_materializations() -> None:
    seed = "frontier-f-determinism-seed"
    commitment = task_bank._digest({"seed": seed})
    first_candidate, first_evaluator = task_bank.materialize(commitment, seed, "F", 3)
    second_candidate, second_evaluator = task_bank.materialize(commitment, seed, "F", 3)

    assert first_candidate == second_candidate
    assert first_evaluator == second_evaluator
    assert first_candidate["sha256"] == second_candidate["sha256"]
    assert first_evaluator["sha256"] == second_evaluator["sha256"]


def test_frontier_f_resolves_real_candidate_visible_clip_and_transcript_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_librispeech_audio_fixture(monkeypatch)
    seed = "frontier-f-ground-truth-seed"
    commitment = task_bank._digest({"seed": seed})
    candidate, evaluator = task_bank.materialize(commitment, seed, "F", 2)

    assert task_bank.candidate_is_structurally_safe(candidate)
    assert task_bank._candidate_tree_is_safe(candidate)

    for task, answer in zip(candidate["tasks"], evaluator["answers"], strict=True):
        selector = task["clip_selector"]
        clip_path = Path(selector["clip_path"])
        assert clip_path.is_file(), f"missing candidate-visible clip: {clip_path}"
        assert "evaluator-only" not in selector["clip_path"].split("/")
        assert clip_path.read_bytes()[:4] == b"fLaC"

        release = answer["hidden_annotation_release"]
        assert isinstance(release, dict)
        assert release["transcript"]
        assert release["disturbed_event_sequence"]
        assert release["event_timeline"]
        assert release["clip_path"] == selector["clip_path"]
        assert release["disturbed_interval"] == selector["interval"]

        annotation_path = Path(release["annotation_path"])
        assert "evaluator-only" in annotation_path.parts
        assert annotation_path.is_file()
        expected_line = f"{release['utterance_id']} {release['transcript']}"
        assert expected_line in {line.strip() for line in annotation_path.read_text(encoding="utf-8").splitlines()}


def test_frontier_f_candidate_has_no_forbidden_keys_or_transcript_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_librispeech_audio_fixture(monkeypatch)
    seed = "frontier-f-safety-seed"
    commitment = task_bank._digest({"seed": seed})
    candidate, evaluator = task_bank.materialize(commitment, seed, "F", 2)

    assert task_bank.candidate_is_structurally_safe(candidate)
    keys = _collect_keys(candidate)
    for key in keys:
        assert not task_bank._candidate_key_is_forbidden(key), key

    for task, answer in zip(candidate["tasks"], evaluator["answers"], strict=True):
        transcript = answer["hidden_annotation_release"]["transcript"]
        dumped = str(task)
        assert transcript not in dumped
        assert "evaluator-only" not in dumped
        assert "hidden_annotation_release" not in dumped


def test_frontier_f_candidate_from_index_without_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Candidate half must materialize from the committed index alone."""
    empty_root = tmp_path / "missing-librispeech"
    empty_root.mkdir()
    monkeypatch.setattr(task_bank, "_audio_corpus_root", lambda: empty_root)
    task_bank._AUDIO_CLIP_INDEX_CACHE.clear()

    seed = "frontier-f-index-only-seed"
    commitment = task_bank._digest({"seed": seed})
    candidate, evaluator = task_bank.materialize(commitment, seed, "F", 2)

    assert task_bank.candidate_is_structurally_safe(candidate)
    assert len(candidate["tasks"]) == 2
    for task in candidate["tasks"]:
        selector = task["clip_selector"]
        assert selector["clip_path"].startswith(task_bank._AUDIO_CLIP_PATH_PREFIX)
        assert selector["clip_path"].endswith(".flac")
        assert "evaluator-only" not in selector["clip_path"]
        assert float(selector["interval"]["end_s"]) > float(selector["interval"]["start_s"])
        assert task["disturbance"] in set(task_bank._AUDIO_DISTURBANCES)

    # Without the corpus, evaluator ground truth stays incomplete (no transcript).
    for answer in evaluator["answers"]:
        release = answer["hidden_annotation_release"]
        assert "transcript" not in release
        assert "event_timeline" not in release
        assert release["utterance_id"]
        assert release["clip_path"].startswith(task_bank._AUDIO_CLIP_PATH_PREFIX)


def test_frontier_f_seed_clip_mapping_is_stable() -> None:
    """Seed→clip paths must stay byte-identical to the MANIFEST-era mapping."""
    expected = {
        ("seed-demo", 0): (
            "data/substrate/tangible_sandbox/prefetch/odyssey-public-v1/"
            "librispeech_dev_clean/dev-clean/2035/152373/2035-152373-0006.flac"
        ),
        ("seed-demo", 1): (
            "data/substrate/tangible_sandbox/prefetch/odyssey-public-v1/"
            "librispeech_dev_clean/dev-clean/2428/83699/2428-83699-0026.flac"
        ),
        ("frontier-f-ground-truth-seed", 0): (
            "data/substrate/tangible_sandbox/prefetch/odyssey-public-v1/"
            "librispeech_dev_clean/dev-clean/2086/149220/2086-149220-0027.flac"
        ),
        ("frontier-f-ground-truth-seed", 1): (
            "data/substrate/tangible_sandbox/prefetch/odyssey-public-v1/"
            "librispeech_dev_clean/dev-clean/1919/142785/1919-142785-0012.flac"
        ),
        ("mapping-probe-a", 0): (
            "data/substrate/tangible_sandbox/prefetch/odyssey-public-v1/"
            "librispeech_dev_clean/dev-clean/6345/93306/6345-93306-0015.flac"
        ),
        ("mapping-probe-b", 7): (
            "data/substrate/tangible_sandbox/prefetch/odyssey-public-v1/"
            "librispeech_dev_clean/dev-clean/1462/170142/1462-170142-0010.flac"
        ),
        ("mapping-probe-c", 42): (
            "data/substrate/tangible_sandbox/prefetch/odyssey-public-v1/"
            "librispeech_dev_clean/dev-clean/777/126732/777-126732-0048.flac"
        ),
    }
    for (seed, index), clip_path in expected.items():
        candidate, _ = task_bank._audio(seed, index)
        assert candidate["clip_selector"]["clip_path"] == clip_path


def test_frontier_f_audio_is_deterministic_without_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_root = tmp_path / "no-corpus"
    empty_root.mkdir()
    monkeypatch.setattr(task_bank, "_audio_corpus_root", lambda: empty_root)
    task_bank._AUDIO_CLIP_INDEX_CACHE.clear()

    first = task_bank._audio("determinism-seed", 3)
    second = task_bank._audio("determinism-seed", 3)
    assert first == second


def test_frontier_f_refuses_when_clip_index_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "does-not-exist.json"
    monkeypatch.setattr(task_bank, "_audio_clip_index_path", lambda: missing)
    task_bank._AUDIO_CLIP_INDEX_CACHE.clear()

    with pytest.raises(task_bank.Refused, match="clip index is unavailable"):
        task_bank._audio("missing-index-seed", 0)


def test_frontier_f_refuses_evaluator_ground_truth_without_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_root = tmp_path / "missing-transcripts"
    empty_root.mkdir()
    monkeypatch.setattr(task_bank, "_audio_corpus_root", lambda: empty_root)
    task_bank._AUDIO_CLIP_INDEX_CACHE.clear()

    with pytest.raises(task_bank.Refused, match="evaluator-only transcript is unavailable"):
        task_bank._audio("gt-required-seed", 0, require_evaluator_ground_truth=True)


def test_frontier_f_candidate_structurally_safe_without_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_root = tmp_path / "no-corpus-for-safety"
    empty_root.mkdir()
    monkeypatch.setattr(task_bank, "_audio_corpus_root", lambda: empty_root)
    task_bank._AUDIO_CLIP_INDEX_CACHE.clear()

    seed = "safety-without-corpus"
    commitment = task_bank._digest({"seed": seed})
    candidate, _ = task_bank.materialize(commitment, seed, "F", 2)
    assert task_bank.candidate_is_structurally_safe(candidate)
    keys = _collect_keys(candidate)
    for key in keys:
        assert not task_bank._candidate_key_is_forbidden(key), key
    dumped = str(candidate)
    assert "evaluator-only" not in dumped
    assert "transcript" not in dumped
    assert "hidden_annotation_release" not in dumped
