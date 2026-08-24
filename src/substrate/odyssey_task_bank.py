"""Deterministic, custody-separated Odyssey task-bank generators."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

PROGRAM = "substrate-odyssey-7d-v1"

# Candidate-visible LibriSpeech dev-clean tree. Transcripts live under
# ``evaluator-only/transcripts/`` beside it and must never appear on the
# candidate half. Paths are root-relative to the repository working tree.
_AUDIO_CORPUS_ROOT = Path(
    "data/substrate/tangible_sandbox/prefetch/odyssey-public-v1/librispeech_dev_clean"
)
_AUDIO_CLIP_PATH_PREFIX = (
    "data/substrate/tangible_sandbox/prefetch/odyssey-public-v1/librispeech_dev_clean/"
)
# Committed clip index: seed→clip + duration mapping without the 90 GB corpus.
# Order matches the sealed corpus MANIFEST.sha256 FLAC listing. Transcripts are
# intentionally absent — they are the evaluator answer.
_AUDIO_CLIP_INDEX_PATH = Path("plans/substrate/tangible_next_launch/LIBRISPEECH_CLIP_INDEX.json")
_AUDIO_LICENSED_BANK = "LibriSpeech_or_FSD50K"
_AUDIO_DISTURBANCES = ("masked_interval", "timestamp_shift", "signal_dropout")
# Cached committed index rows: (clip_rel, duration_s, utterance_id).
_AUDIO_CLIP_INDEX_CACHE: dict[str, tuple[tuple[str, float, str], ...]] = {}


class Refused(ValueError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _rng(seed: str, frontier: str, task_index: int) -> random.Random:
    if frontier not in set("ABCDEFGH") or task_index < 0:
        raise Refused("unsupported frontier or task index")
    return random.Random(int.from_bytes(hashlib.sha256(f"{seed}|{frontier}|{task_index}".encode()).digest(), "big"))


def _audio_corpus_root() -> Path:
    """Return the LibriSpeech corpus root (overridable in tests)."""
    return _AUDIO_CORPUS_ROOT


def _audio_clip_index_path() -> Path:
    """Return the committed clip-index path (overridable in tests)."""
    return _AUDIO_CLIP_INDEX_PATH


def _load_committed_clip_index() -> tuple[tuple[str, float, str], ...]:
    """Load the committed LibriSpeech clip index (path, duration, utterance id).

    Manifest generation only needs this index — not the audio bytes. Order is
    the sealed corpus MANIFEST acquisition order so seed→clip mapping stays
    byte-identical to the previous MANIFEST-backed implementation.
    """
    path = _audio_clip_index_path()
    cache_key = str(path)
    cached = _AUDIO_CLIP_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not path.is_file():
        raise Refused("LibriSpeech clip index is unavailable")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"LibriSpeech clip index is unreadable: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != "SUBSTRATE_LIBRISPEECH_CLIP_INDEX/v1":
        raise Refused("LibriSpeech clip index has an unexpected schema")
    rows = payload.get("clips")
    if not isinstance(rows, list) or not rows:
        raise Refused("LibriSpeech clip index has no clips")
    clips: list[tuple[str, float, str]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 3:
            raise Refused("LibriSpeech clip index row is malformed")
        rel, duration_s, utterance_id = row
        if not isinstance(rel, str) or not rel.endswith(".flac"):
            raise Refused("LibriSpeech clip index contains a non-FLAC path")
        if rel.startswith("/") or ".." in rel.split("/"):
            raise Refused("LibriSpeech clip index contains an unsafe path")
        if "evaluator-only" in rel.split("/"):
            raise Refused("LibriSpeech clip index leaked an evaluator-only path")
        if not isinstance(duration_s, (int, float)) or isinstance(duration_s, bool) or float(duration_s) <= 0:
            raise Refused("LibriSpeech clip index has an invalid duration")
        if not isinstance(utterance_id, str) or not utterance_id:
            raise Refused("LibriSpeech clip index has an invalid utterance id")
        clips.append((rel, float(duration_s), utterance_id))
    ordered = tuple(clips)
    _AUDIO_CLIP_INDEX_CACHE[cache_key] = ordered
    return ordered


def _flac_duration_seconds(path: Path) -> float:
    """Read STREAMINFO total-samples / sample-rate without third-party codecs."""
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"fLaC":
                raise Refused(f"audio clip is not a FLAC file: {path}")
            while True:
                header = handle.read(4)
                if len(header) < 4:
                    raise Refused(f"audio clip lacks STREAMINFO: {path}")
                is_last = bool(header[0] & 0x80)
                block_type = header[0] & 0x7F
                size = int.from_bytes(header[1:], "big")
                payload = handle.read(size)
                if len(payload) != size:
                    raise Refused(f"audio clip STREAMINFO truncated: {path}")
                if block_type == 0:
                    if len(payload) < 18:
                        raise Refused(f"audio clip STREAMINFO too short: {path}")
                    sample_rate = (payload[10] << 12) | (payload[11] << 4) | (payload[12] >> 4)
                    total_samples = ((payload[13] & 0x0F) << 32) | int.from_bytes(payload[14:18], "big")
                    if sample_rate <= 0 or total_samples <= 0:
                        raise Refused(f"audio clip has invalid duration metadata: {path}")
                    return total_samples / sample_rate
                if is_last:
                    raise Refused(f"audio clip lacks STREAMINFO: {path}")
    except OSError as error:
        raise Refused(f"audio clip is unreadable: {path}: {error}") from error


def _read_librispeech_transcript(root: Path, clip_rel: str, utterance_id: str) -> tuple[str, str]:
    """Load the evaluator-only transcript line for *utterance_id*.

    Returns ``(transcript_text, annotation_path_root_relative)``.
    Transcripts are never committed into the candidate-visible clip index; they
    remain under the corpus ``evaluator-only/transcripts/`` tree.
    """
    parent = Path(clip_rel).parent.as_posix()
    chapter_id = "-".join(utterance_id.split("-")[:2])
    annotation_rel = f"evaluator-only/transcripts/{parent}/{chapter_id}.trans.txt"
    annotation_path = root / annotation_rel
    if not annotation_path.is_file():
        raise Refused(f"LibriSpeech evaluator-only transcript is unavailable for {utterance_id}")
    try:
        lines = annotation_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Refused(f"LibriSpeech evaluator-only transcript is unreadable for {utterance_id}: {error}") from error
    prefix = f"{utterance_id} "
    for line in lines:
        if line.startswith(prefix):
            text = line[len(prefix) :].strip()
            if not text:
                raise Refused(f"LibriSpeech transcript for {utterance_id} is empty")
            # Root-relative path including the corpus prefix used by the rest of the tree.
            full_annotation = f"{_AUDIO_CLIP_PATH_PREFIX}{annotation_rel}"
            return text, full_annotation
    raise Refused(f"LibriSpeech transcript has no line for {utterance_id}")


def _equal_word_timeline(transcript: str, duration_s: float) -> list[dict[str, Any]]:
    """Deterministic word timeline when the corpus has no forced alignments.

    LibriSpeech supplies utterance text only. Equal spacing is an explicit,
    seed-stable proxy so timestamp-tolerance scoring has a concrete sequence
    to compare against; it is not a claim of true word onsets.
    """
    tokens = transcript.split()
    if not tokens:
        raise Refused("cannot build event timeline from empty transcript")
    if duration_s <= 0:
        raise Refused("cannot build event timeline from non-positive duration")
    step = duration_s / len(tokens)
    events: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        events.append(
            {
                "index": index,
                "token": token,
                "start_s": round(index * step, 4),
                "end_s": round((index + 1) * step, 4),
            }
        )
    return events


def _events_overlapping(events: list[dict[str, Any]], start_s: float, end_s: float) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if float(event["end_s"]) > start_s and float(event["start_s"]) < end_s
    ]


def _build_audio_candidate(
    task_index: int,
    *,
    slot: int,
    disturbance: str,
    clip_rel: str,
    duration_s: float,
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    """Assemble the candidate-visible audio task (index-only inputs)."""
    candidate_clip_path = f"{_AUDIO_CLIP_PATH_PREFIX}{clip_rel}"
    return {
        "schema": "SUBSTRATE_ODYSSEY_AUDIO_TASK/v1",
        "program": PROGRAM,
        "activation": False,
        "task_id": f"F-{task_index:04d}",
        "frontier": "F",
        "family": "timestamped_audio_reconstruction",
        "clip_selector": {
            "licensed_bank": _AUDIO_LICENSED_BANK,
            "slot": slot,
            "clip_path": candidate_clip_path,
            "interval": {"start_s": start_s, "end_s": end_s},
        },
        "disturbance": disturbance,
        "request": "Recover the timestamped event sequence and state uncertainty for the disturbed segment.",
        "required_receipt": ["event_timeline", "recovery_method", "uncertainty"],
    }


def _build_audio_answer(
    *,
    task_id: str,
    utterance_id: str,
    clip_rel: str,
    candidate_clip_path: str,
    duration_s: float,
    disturbance: str,
    start_s: float,
    end_s: float,
    require_evaluator_ground_truth: bool,
) -> dict[str, Any]:
    """Build the evaluator answer, loading transcripts only from the corpus.

    Candidate generation never enters here for its required fields. When the
    corpus evaluator-only tree is absent, either refuse (if ground truth is
    required) or return an answer that carries only index-derived metadata so
    mutation fixtures and candidate manifests can still materialize.
    """
    release: dict[str, Any] = {
        "utterance_id": utterance_id,
        "clip_path": candidate_clip_path,
        "clip_duration_s": round(duration_s, 4),
        "disturbance": disturbance,
        "disturbed_interval": {"start_s": start_s, "end_s": end_s},
        "timeline_model": "equal_word_spacing",
    }
    try:
        transcript, annotation_path = _read_librispeech_transcript(_audio_corpus_root(), clip_rel, utterance_id)
    except Refused:
        if require_evaluator_ground_truth:
            raise
        return {
            "schema": "SUBSTRATE_ODYSSEY_AUDIO_ANSWER/v1",
            "activation": False,
            "task_id": task_id,
            "hidden_annotation_release": release,
            "scoring": "timestamp tolerance and recovery calibration",
        }

    event_timeline = _equal_word_timeline(transcript, duration_s)
    disturbed_events = _events_overlapping(event_timeline, start_s, end_s)
    if not disturbed_events:
        # Degenerate float edge: force at least the middle token into scope.
        # The candidate interval is sealed from the index alone and is not
        # rewritten here, so candidate generation stays corpus-independent.
        mid = event_timeline[len(event_timeline) // 2]
        disturbed_events = [mid]
    release.update(
        {
            "transcript": transcript,
            "annotation_path": annotation_path,
            "event_timeline": event_timeline,
            "disturbed_event_sequence": disturbed_events,
        }
    )
    return {
        "schema": "SUBSTRATE_ODYSSEY_AUDIO_ANSWER/v1",
        "activation": False,
        "task_id": task_id,
        "hidden_annotation_release": release,
        "scoring": "timestamp tolerance and recovery calibration",
    }


def _resolve_audio_clip(
    task_index: int,
    rng: random.Random,
    *,
    require_evaluator_ground_truth: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map seed+index to a clip via the committed index; GT from corpus only.

    The candidate half is fully determined by the committed clip index and the
    seeded RNG — no audio bytes required. Evaluator ground truth still reads
    ``evaluator-only/transcripts/`` under the corpus root.
    """
    slot = rng.randrange(100000)
    disturbance = rng.choice(list(_AUDIO_DISTURBANCES))
    # Interval placement draws must follow slot/disturbance so the seed stream
    # stays fully deterministic for a fixed generator version.
    start_frac = rng.random() * 0.55
    length_frac = 0.12 + rng.random() * 0.28

    clips = _load_committed_clip_index()
    clip_rel, duration_s, utterance_id = clips[slot % len(clips)]

    start_s = round(start_frac * duration_s, 4)
    end_s = round(min(duration_s, start_s + length_frac * duration_s), 4)
    if end_s <= start_s:
        end_s = round(min(duration_s, start_s + max(0.05, duration_s * 0.05)), 4)
    if end_s <= start_s:
        raise Refused(f"audio clip duration too short to disturb: {clip_rel}")

    candidate = _build_audio_candidate(
        task_index,
        slot=slot,
        disturbance=disturbance,
        clip_rel=clip_rel,
        duration_s=duration_s,
        start_s=start_s,
        end_s=end_s,
    )
    answer = _build_audio_answer(
        task_id=candidate["task_id"],
        utterance_id=utterance_id,
        clip_rel=clip_rel,
        candidate_clip_path=candidate["clip_selector"]["clip_path"],
        duration_s=duration_s,
        disturbance=disturbance,
        start_s=start_s,
        end_s=end_s,
        require_evaluator_ground_truth=require_evaluator_ground_truth,
    )
    return candidate, answer


def build_librispeech_clip_index(corpus_root: Path, output_path: Path) -> dict[str, Any]:
    """Regenerate the committed clip index from a local LibriSpeech corpus.

    Reads ``MANIFEST.sha256`` for stable ordering and each FLAC STREAMINFO for
    duration. Never writes transcripts. Records the MANIFEST digest so drift
    against the sealed corpus is detectable.
    """
    root = corpus_root.expanduser()
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise Refused(f"corpus MANIFEST.sha256 is missing under {root}")
    try:
        manifest_bytes = manifest.read_bytes()
    except OSError as error:
        raise Refused(f"corpus MANIFEST.sha256 is unreadable: {error}") from error
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    clips: list[list[Any]] = []
    for line in manifest_bytes.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            continue
        rel = parts[1].replace("\\", "/")
        if not rel.endswith(".flac"):
            continue
        if rel.startswith("/") or ".." in rel.split("/"):
            raise Refused(f"corpus MANIFEST contains an unsafe path: {rel}")
        if "evaluator-only" in rel.split("/"):
            raise Refused(f"corpus MANIFEST leaked an evaluator-only path: {rel}")
        flac_path = root / rel
        if not flac_path.is_file():
            raise Refused(f"corpus FLAC is missing: {rel}")
        duration_s = _flac_duration_seconds(flac_path)
        clips.append([rel, duration_s, Path(rel).stem])
    if not clips:
        raise Refused("corpus MANIFEST has no FLAC clips")
    payload: dict[str, Any] = {
        "schema": "SUBSTRATE_LIBRISPEECH_CLIP_INDEX/v1",
        "corpus_root": _AUDIO_CORPUS_ROOT.as_posix(),
        "manifest_sha256": manifest_sha256,
        "clip_count": len(clips),
        "clips": clips,
    }
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n", encoding="utf-8")
    _AUDIO_CLIP_INDEX_CACHE.clear()
    return payload


def _logic(seed: str, task_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = _rng(seed, "C", task_index)
    a, b, c = rng.sample(["A", "B", "C", "D", "E"], 3)
    satisfiable = bool(rng.getrandbits(1))
    if satisfiable:
        # The all-false assignment is an explicit valid witness for this
        # implication/constraint set.
        rules = [f"{a} -> {b}", f"{b} -> {c}", f"not ({a} and {c})"]
        witness = {a: False, b: False, c: False}
        core = None
    else:
        # These three visible formulae are jointly inconsistent: A forces B,
        # while both A and not-B are asserted.
        rules = [a, f"{a} -> {b}", f"not {b}"]
        witness = None
        core = rules
    candidate = {
        "schema": "SUBSTRATE_ODYSSEY_LOGIC_TASK/v1",
        "program": PROGRAM,
        "activation": False,
        "task_id": f"C-{task_index:04d}",
        "frontier": "C",
        "family": "satisfiability_and_countermodel",
        "rules": rules,
        "request": "Determine satisfiability. If satisfiable, give a satisfying assignment; otherwise derive a minimal contradiction.",
        "required_receipt": ["derivation", "claimed_status", "claimed_assignment_or_core"],
    }
    answer = {
        "schema": "SUBSTRATE_ODYSSEY_LOGIC_ANSWER/v1",
        "activation": False,
        "task_id": candidate["task_id"],
        "satisfiable": satisfiable,
        "witness_assignment": witness,
        "minimal_unsat_core": core,
        "scoring": "verify all visible rules under submitted assignment; compare the visible unsat core only if unsatisfiable",
    }
    return candidate, answer


def _philosophy(seed: str, task_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = _rng(seed, "E", task_index)
    domain = ["scientific testimony", "historical attribution", "policy forecast", "instrument calibration"][rng.randrange(4)]
    initial = rng.choice(["accept", "withhold", "reject"])
    evidence = [
        {"id": "e1", "reliability": "medium", "claim": "A primary observation supports proposition P."},
        {"id": "e2", "reliability": "high", "claim": "A later calibration reveals a possible confound."},
        {"id": "e3", "reliability": "medium", "claim": "An independent source partially replicates P under changed conditions."},
    ]
    candidate = {
        "schema": "SUBSTRATE_ODYSSEY_EPISTEMIC_TASK/v1",
        "program": PROGRAM,
        "activation": False,
        "task_id": f"E-{task_index:04d}",
        "frontier": "E",
        "family": "epistemic_commitment_revision",
        "domain": domain,
        "prior_commitment": initial,
        "evidence": evidence,
        "request": (
            "Maintain an explicit commitment ledger. Map support/conflict, state what "
            "changed, and give a calibrated revised position with unresolved alternatives."
        ),
        "required_receipt": ["commitment_ledger", "argument_map", "revision_rationale", "residual_uncertainty"],
    }
    answer = {
        "schema": "SUBSTRATE_ODYSSEY_EPISTEMIC_ANSWER/v1",
        "activation": False,
        "task_id": candidate["task_id"],
        "expected_revision_class": "revise_to_withhold" if initial == "accept" else "revise_with_qualified_acceptance",
        "required_conflict": "e2 conflicts with an unqualified acceptance of e1",
        "scoring": "credit explicit evidence tracking, calibration, and preserved uncertainty; do not score a preferred philosophical doctrine",
    }
    return candidate, answer


def _project(seed: str, task_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = _rng(seed, "A", task_index)
    candidate = {
        "schema": "SUBSTRATE_ODYSSEY_CONTINUITY_TASK/v1",
        "program": PROGRAM,
        "activation": False,
        "task_id": f"A-{task_index:04d}",
        "frontier": "A",
        "family": "cross_modal_project_reconstruction",
        "packet": {
            "document": f"project-note-{rng.randrange(1000)}",
            "telemetry": [rng.randrange(10) for _ in range(4)],
            "state_delta": {"revision": task_index},
        },
        "request": "Reconstruct current project state and produce a repair plan after an interruption.",
        "required_receipt": ["state_reconstruction", "uncertainty", "repair_plan"],
    }
    return candidate, {
        "schema": "SUBSTRATE_ODYSSEY_CONTINUITY_ANSWER/v1",
        "activation": False,
        "task_id": candidate["task_id"],
        "expected_state_digest": _digest(candidate["packet"]),
        "scoring": "state fields and recovery ordering",
    }


def _math(seed: str, task_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = _rng(seed, "B", task_index)
    left, right = rng.randrange(11, 99), rng.randrange(2, 10)
    candidate = {
        "schema": "SUBSTRATE_ODYSSEY_MATH_TASK/v1",
        "program": PROGRAM,
        "activation": False,
        "task_id": f"B-{task_index:04d}",
        "frontier": "B",
        "family": "derivation_continuation",
        "source_selector": {"dataset": "MATH", "task_index": task_index},
        "request": f"Use a checkable derivation to simplify ({left} * {right}) + x = {left * right + right}. Report x and the derivation invariant.",
        "required_receipt": ["derivation", "final_answer", "verification_trace"],
    }
    return candidate, {
        "schema": "SUBSTRATE_ODYSSEY_MATH_ANSWER/v1",
        "activation": False,
        "task_id": candidate["task_id"],
        "expected_answer": right,
        "scoring": "symbolic substitution and derivation consistency",
    }


def _code(seed: str, task_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = _rng(seed, "D", task_index)
    candidate = {
        "schema": "SUBSTRATE_ODYSSEY_CODE_TASK/v1",
        "program": PROGRAM,
        "activation": False,
        "task_id": f"D-{task_index:04d}",
        "frontier": "D",
        "family": "issue_history_reconstruction",
        "repository_selector": {"allowlist_slot": rng.randrange(32), "base_commit_slot": task_index},
        "issue_history": ["initial failing test", "partial attempted repair", "later regression report"],
        "request": "Propose a minimal patch plan and a reproducible test sequence without changing the pinned base commit.",
        "required_receipt": ["patch_plan", "test_sequence", "rollback_condition"],
    }
    return candidate, {
        "schema": "SUBSTRATE_ODYSSEY_CODE_ANSWER/v1",
        "activation": False,
        "task_id": candidate["task_id"],
        "hidden_test_class": "regression_and_backward_compatibility",
        "scoring": "pinned-base and hidden-test trace",
    }


def _audio(
    seed: str,
    task_index: int,
    *,
    require_evaluator_ground_truth: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Frontier-F generator.

    Candidate tasks resolve from the committed clip index alone. Evaluator
    ground truth still requires the corpus transcript tree; set
    ``require_evaluator_ground_truth=True`` to refuse when that tree is absent
    rather than returning index-only evaluator metadata.
    """
    rng = _rng(seed, "F", task_index)
    return _resolve_audio_clip(
        task_index,
        rng,
        require_evaluator_ground_truth=require_evaluator_ground_truth,
    )


def _embodied(seed: str, task_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = _rng(seed, "G", task_index)
    action = [round(rng.uniform(-1, 1), 3) for _ in range(3)]
    candidate = {
        "schema": "SUBSTRATE_ODYSSEY_EMBODIED_TASK/v1",
        "program": PROGRAM,
        "activation": False,
        "task_id": f"G-{task_index:04d}",
        "frontier": "G",
        "family": "action_conditioned_counterfactual",
        "scene_selector": {"movi_or_kubric_seed_slot": rng.randrange(100000)},
        "action": action,
        "observations": ["rgb", "depth", "flow", "object_state"],
        "request": "Predict the next physical state and specify an action-conditioned recovery after a sensor change.",
        "required_receipt": ["state_prediction", "counterfactual", "recovery_action"],
    }
    return candidate, {
        "schema": "SUBSTRATE_ODYSSEY_EMBODIED_ANSWER/v1",
        "activation": False,
        "task_id": candidate["task_id"],
        "hidden_future_state": [round(rng.uniform(-2, 2), 3) for _ in range(3)],
        "scoring": "state error, contact-event and recovery criteria",
    }


def _science(seed: str, task_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = _rng(seed, "H", task_index)
    candidate = {
        "schema": "SUBSTRATE_ODYSSEY_SCIENCE_TASK/v1",
        "program": PROGRAM,
        "activation": False,
        "task_id": f"H-{task_index:04d}",
        "frontier": "H",
        "family": "telemetry_hypothesis_revision",
        "evidence": {
            "arc_slot": rng.randrange(7787),
            "telemetry": [rng.randrange(100) for _ in range(5)],
            "instrument_note": "calibration changed after sample two",
        },
        "request": "State competing causal hypotheses, identify discriminating evidence, and revise the current hypothesis.",
        "required_receipt": ["hypotheses", "causal_graph", "revision", "next_measurement"],
    }
    return candidate, {
        "schema": "SUBSTRATE_ODYSSEY_SCIENCE_ANSWER/v1",
        "activation": False,
        "task_id": candidate["task_id"],
        "hidden_generator_class": rng.choice(["confound", "measurement_error", "causal_shift"]),
        "scoring": "causal discrimination and calibrated revision",
    }


def materialize(
    seed_commitment: str,
    secret_seed: str,
    frontier: str,
    task_count: int,
    *,
    require_evaluator_ground_truth: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one frontier's candidate/evaluator pair from the custodian seed.

    ``require_evaluator_ground_truth`` is opt-in because mutation fixtures and
    clean clones legitimately materialize candidate manifests without the
    gitignored corpus.  A real launch materialization passes it so a frontier
    can never seal with a placeholder standing in for a scored answer.
    """
    if task_count < 1 or _digest({"seed": secret_seed}) != seed_commitment:
        raise Refused("invalid task count or custodian seed commitment")
    generator = {"A": _project, "B": _math, "C": _logic, "D": _code, "E": _philosophy, "F": _audio, "G": _embodied, "H": _science}.get(frontier)
    if generator is None:
        raise Refused("unsupported frozen frontier")
    if frontier == "F":
        pairs = [
            generator(secret_seed, index, require_evaluator_ground_truth=require_evaluator_ground_truth)
            for index in range(task_count)
        ]
    else:
        pairs = [generator(secret_seed, index) for index in range(task_count)]
    tasks, answers = zip(*pairs, strict=True)
    candidate = {
        "schema": "SUBSTRATE_ODYSSEY_CANDIDATE_TASK_MANIFEST/v1",
        "activation": False,
        "frontier": frontier,
        "seed_commitment": seed_commitment,
        "tasks": list(tasks),
    }
    evaluator = {
        "schema": "SUBSTRATE_ODYSSEY_EVALUATOR_ANSWER_MANIFEST/v1",
        "activation": False,
        "frontier": frontier,
        "seed_commitment": seed_commitment,
        "answers": list(answers),
    }
    candidate["sha256"] = _digest(candidate)
    evaluator["sha256"] = _digest(evaluator)
    return candidate, evaluator


_TASK_SCHEMAS = {
    "A": ("SUBSTRATE_ODYSSEY_CONTINUITY_TASK/v1", {"packet"}),
    "B": ("SUBSTRATE_ODYSSEY_MATH_TASK/v1", {"source_selector"}),
    "C": ("SUBSTRATE_ODYSSEY_LOGIC_TASK/v1", {"rules"}),
    "D": ("SUBSTRATE_ODYSSEY_CODE_TASK/v1", {"repository_selector", "issue_history"}),
    "E": ("SUBSTRATE_ODYSSEY_EPISTEMIC_TASK/v1", {"domain", "prior_commitment", "evidence"}),
    "F": ("SUBSTRATE_ODYSSEY_AUDIO_TASK/v1", {"clip_selector", "disturbance"}),
    "G": ("SUBSTRATE_ODYSSEY_EMBODIED_TASK/v1", {"scene_selector", "action", "observations"}),
    "H": ("SUBSTRATE_ODYSSEY_SCIENCE_TASK/v1", {"evidence"}),
}
_TASK_COMMON_KEYS = {"schema", "program", "activation", "task_id", "frontier", "family", "request", "required_receipt"}
_FIXED_REQUESTS = {
    "A": "Reconstruct current project state and produce a repair plan after an interruption.",
    "C": "Determine satisfiability. If satisfiable, give a satisfying assignment; otherwise derive a minimal contradiction.",
    "D": "Propose a minimal patch plan and a reproducible test sequence without changing the pinned base commit.",
    "E": (
        "Maintain an explicit commitment ledger. Map support/conflict, state what "
        "changed, and give a calibrated revised position with unresolved alternatives."
    ),
    "F": "Recover the timestamped event sequence and state uncertainty for the disturbed segment.",
    "G": "Predict the next physical state and specify an action-conditioned recovery after a sensor change.",
    "H": "State competing causal hypotheses, identify discriminating evidence, and revise the current hypothesis.",
}
_EXPECTED_FAMILIES = {
    "A": "cross_modal_project_reconstruction",
    "B": "derivation_continuation",
    "C": "satisfiability_and_countermodel",
    "D": "issue_history_reconstruction",
    "E": "epistemic_commitment_revision",
    "F": "timestamped_audio_reconstruction",
    "G": "action_conditioned_counterfactual",
    "H": "telemetry_hypothesis_revision",
}
_EXPECTED_RECEIPTS = {
    "A": ["state_reconstruction", "uncertainty", "repair_plan"],
    "B": ["derivation", "final_answer", "verification_trace"],
    "C": ["derivation", "claimed_status", "claimed_assignment_or_core"],
    "D": ["patch_plan", "test_sequence", "rollback_condition"],
    "E": ["commitment_ledger", "argument_map", "revision_rationale", "residual_uncertainty"],
    "F": ["event_timeline", "recovery_method", "uncertainty"],
    "G": ["state_prediction", "counterfactual", "recovery_action"],
    "H": ["hypotheses", "causal_graph", "revision", "next_measurement"],
}
_MATH_REQUEST = re.compile(
    r"Use a checkable derivation to simplify \((?P<left>\d+) \* (?P<right>\d+)\) \+ x = (?P<total>\d+)\. "
    r"Report x and the derivation invariant\."
)
_FORBIDDEN_CANDIDATE_KEY_PARTS = {
    "answer",
    "answers",
    "countermodel",
    "evaluator",
    "evaluation",
    "expected",
    "ground_truth",
    "hidden",
    "key",
    "label",
    "score",
    "scoring",
    "solution",
    "target",
    "truth",
    "witness",
}


def _candidate_key_is_forbidden(key: object) -> bool:
    if not isinstance(key, str):
        return True
    normalized = key.casefold().replace("-", "_").split("_")
    return any(part in _FORBIDDEN_CANDIDATE_KEY_PARTS for part in normalized)


def _candidate_tree_is_safe(value: Any) -> bool:
    if isinstance(value, dict):
        return all(not _candidate_key_is_forbidden(key) and _candidate_tree_is_safe(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_candidate_tree_is_safe(item) for item in value)
    return True


def _task_index(task_id: str, frontier: str) -> int:
    match = re.fullmatch(rf"{re.escape(frontier)}-(\d{{4}})", task_id)
    if match is None:
        raise Refused("candidate task identifier must be exactly frontier-indexed")
    return int(match.group(1))


def _bounded_integer(value: Any, *, minimum: int, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _validate_task_values(task: dict[str, Any], *, frontier: str, task_index: int) -> None:
    """Restrict candidate content to the fixed public generator grammar.

    Closed field names alone cannot stop an answer being slipped into an
    otherwise permitted string.  These constraints limit every generated
    field to the public task grammar.  They do *not* establish semantic
    blindness: only the separate custodian/isolation gates can establish that
    an evaluator's answers were never accessible to the candidate.
    """
    if task["family"] != _EXPECTED_FAMILIES[frontier] or task["required_receipt"] != _EXPECTED_RECEIPTS[frontier]:
        raise Refused("candidate task family or receipt grammar drifted")
    if frontier == "B":
        match = _MATH_REQUEST.fullmatch(task["request"])
        selector = task["source_selector"]
        if (
            match is None
            or not isinstance(selector, dict)
            or selector != {"dataset": "MATH", "task_index": task_index}
            or not (11 <= int(match["left"]) <= 98 and 2 <= int(match["right"]) <= 9)
            or int(match["total"]) != int(match["left"]) * int(match["right"]) + int(match["right"])
        ):
            raise Refused("math candidate task is outside the public generator grammar")
        return
    if task["request"] != _FIXED_REQUESTS[frontier]:
        raise Refused("candidate task request is outside the public generator grammar")
    if frontier == "A":
        packet = task["packet"]
        if (
            not isinstance(packet, dict)
            or set(packet) != {"document", "telemetry", "state_delta"}
            or not isinstance(packet["document"], str)
            or re.fullmatch(r"project-note-\d{1,3}", packet["document"]) is None
            or not isinstance(packet["telemetry"], list)
            or len(packet["telemetry"]) != 4
            or not all(_bounded_integer(value, minimum=0, maximum=9) for value in packet["telemetry"])
            or packet["state_delta"] != {"revision": task_index}
        ):
            raise Refused("continuity candidate task is outside the public generator grammar")
    elif frontier == "C":
        rules = task["rules"]
        if not isinstance(rules, list) or len(rules) != 3 or not all(isinstance(rule, str) for rule in rules):
            raise Refused("logic candidate task has invalid public rules")
        variables = "[ABCDE]"
        satisfiable = re.fullmatch(rf"({variables}) -> ({variables})", rules[0])
        if satisfiable is not None:
            first, second = satisfiable.groups()
            third = re.fullmatch(rf"({variables}) -> ({variables})", rules[1])
            expected = f"not ({first} and {third.group(2)})" if third is not None else None
            if third is None or third.group(1) != second or len({first, second, third.group(2)}) != 3 or rules[2] != expected:
                raise Refused("logic satisfiable task is outside the public generator grammar")
        else:
            if (
                re.fullmatch(variables, rules[0]) is None
                or re.fullmatch(rf"{rules[0]} -> ({variables})", rules[1]) is None
                or rules[2] != f"not {rules[1][-1]}"
                or rules[0] == rules[1][-1]
            ):
                raise Refused("logic unsatisfiable task is outside the public generator grammar")
    elif frontier == "D":
        selector = task["repository_selector"]
        if (
            not isinstance(selector, dict)
            or selector.get("base_commit_slot") != task_index
            or not _bounded_integer(selector.get("allowlist_slot"), minimum=0, maximum=31)
            or set(selector) != {"allowlist_slot", "base_commit_slot"}
            or task["issue_history"] != ["initial failing test", "partial attempted repair", "later regression report"]
        ):
            raise Refused("code candidate task is outside the public generator grammar")
    elif frontier == "E":
        expected_evidence = [
            {"id": "e1", "reliability": "medium", "claim": "A primary observation supports proposition P."},
            {"id": "e2", "reliability": "high", "claim": "A later calibration reveals a possible confound."},
            {"id": "e3", "reliability": "medium", "claim": "An independent source partially replicates P under changed conditions."},
        ]
        valid_domain = task["domain"] in {
            "scientific testimony",
            "historical attribution",
            "policy forecast",
            "instrument calibration",
        }
        valid_commitment = task["prior_commitment"] in {"accept", "withhold", "reject"}
        if not valid_domain or not valid_commitment or task["evidence"] != expected_evidence:
            raise Refused("epistemic candidate task is outside the public generator grammar")
    elif frontier == "F":
        selector = task["clip_selector"]
        interval = selector.get("interval") if isinstance(selector, dict) else None
        clip_path = selector.get("clip_path") if isinstance(selector, dict) else None
        valid_interval = (
            isinstance(interval, dict)
            and set(interval) == {"start_s", "end_s"}
            and isinstance(interval.get("start_s"), (int, float))
            and not isinstance(interval.get("start_s"), bool)
            and isinstance(interval.get("end_s"), (int, float))
            and not isinstance(interval.get("end_s"), bool)
            and float(interval["start_s"]) >= 0
            and float(interval["end_s"]) > float(interval["start_s"])
        )
        valid_path = (
            isinstance(clip_path, str)
            and clip_path.startswith(_AUDIO_CLIP_PATH_PREFIX)
            and clip_path.endswith(".flac")
            and not clip_path.startswith("/")
            and ".." not in clip_path.split("/")
            and "evaluator-only" not in clip_path.split("/")
        )
        if (
            not isinstance(selector, dict)
            or set(selector) != {"licensed_bank", "slot", "clip_path", "interval"}
            or selector.get("licensed_bank") != _AUDIO_LICENSED_BANK
            or not _bounded_integer(selector.get("slot"), minimum=0, maximum=99999)
            or not valid_path
            or not valid_interval
            or task["disturbance"] not in set(_AUDIO_DISTURBANCES)
        ):
            raise Refused("audio candidate task is outside the public generator grammar")
    elif frontier == "G":
        selector = task["scene_selector"]
        action = task["action"]
        if (
            not isinstance(selector, dict)
            or set(selector) != {"movi_or_kubric_seed_slot"}
            or not _bounded_integer(selector.get("movi_or_kubric_seed_slot"), minimum=0, maximum=99999)
            or not isinstance(action, list)
            or len(action) != 3
            or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and -1 <= float(value) <= 1 for value in action)
            or task["observations"] != ["rgb", "depth", "flow", "object_state"]
        ):
            raise Refused("embodied candidate task is outside the public generator grammar")
    elif frontier == "H":
        evidence = task["evidence"]
        if (
            not isinstance(evidence, dict)
            or set(evidence) != {"arc_slot", "telemetry", "instrument_note"}
            or not _bounded_integer(evidence.get("arc_slot"), minimum=0, maximum=7786)
            or not isinstance(evidence.get("telemetry"), list)
            or len(evidence["telemetry"]) != 5
            or not all(_bounded_integer(value, minimum=0, maximum=99) for value in evidence["telemetry"])
            or evidence.get("instrument_note") != "calibration changed after sample two"
        ):
            raise Refused("science candidate task is outside the public generator grammar")


def verify_materialized_candidate(seed_commitment: str, secret_seed: str, candidate: dict[str, Any]) -> None:
    """Require a custodian's just-materialized tasks to replay exactly.

    This is useful at the only point where the secret seed is available.  The
    candidate artifact does not carry the seed, so later readers can validate
    structural safety but must rely on G04/G10 for real answer custody.
    """
    verify_candidate_manifest(candidate)
    frontier = candidate["frontier"]
    expected, _ = materialize(seed_commitment, secret_seed, frontier, len(candidate["tasks"]))
    if candidate["seed_commitment"] != seed_commitment or candidate["tasks"] != expected["tasks"]:
        raise Refused("candidate task manifest does not replay from the custodian seed")


def verify_candidate_manifest(candidate: dict[str, Any]) -> None:
    """Reject malformed or structurally unsafe candidate-visible task banks.

    A token blacklist is insufficient here: an answer can be placed under any
    invented field name.  The candidate format is therefore closed at both the
    manifest and per-frontier task levels, and recursively rejects evaluator
    namespace fields.  Evaluator manifests intentionally use a different
    schema and are never accepted by this validator.
    """
    if not isinstance(candidate, dict):
        raise Refused("candidate manifest must be an object")
    required = {"schema", "activation", "frontier", "seed_commitment", "tasks", "sha256"}
    allowed = required | {"source_bundle"}
    if not required.issubset(candidate) or set(candidate) - allowed:
        raise Refused("candidate manifest has undeclared or missing fields")
    if candidate.get("schema") != "SUBSTRATE_ODYSSEY_CANDIDATE_TASK_MANIFEST/v1" or candidate.get("activation") is not False:
        raise Refused("candidate manifest has the wrong identity or activation state")
    frontier = candidate.get("frontier")
    if frontier not in _TASK_SCHEMAS:
        raise Refused("candidate manifest has an unsupported frontier")
    commitment = candidate.get("seed_commitment")
    if not isinstance(commitment, str) or len(commitment) != 64 or any(character not in "0123456789abcdef" for character in commitment):
        raise Refused("candidate manifest has an invalid seed commitment")
    source_bundle = candidate.get("source_bundle")
    if source_bundle is not None:
        if not isinstance(source_bundle, dict) or set(source_bundle) != {"selection_sha256", "assets"}:
            raise Refused("candidate source bundle has undeclared or missing fields")
        selection_sha256 = source_bundle.get("selection_sha256")
        assets = source_bundle.get("assets")
        if not isinstance(selection_sha256, str) or len(selection_sha256) != 64 or any(character not in "0123456789abcdef" for character in selection_sha256):
            raise Refused("candidate source bundle has an invalid selection digest")
        if not isinstance(assets, list) or not assets:
            raise Refused("candidate source bundle needs at least one asset")
        paths: set[str] = set()
        roles: set[str] = set()
        for asset in assets:
            if not isinstance(asset, dict) or set(asset) != {"path", "sha256", "role", "read_only"}:
                raise Refused("candidate source asset has undeclared or missing fields")
            path = asset.get("path")
            source_sha256 = asset.get("sha256")
            role = asset.get("role")
            if not isinstance(path, str) or not path or path.startswith("/") or ".." in path.split("/"):
                raise Refused("candidate source asset path must be root-relative")
            if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(character not in "0123456789abcdef" for character in source_sha256):
                raise Refused("candidate source asset has an invalid digest")
            if not isinstance(role, str) or not role or _candidate_key_is_forbidden(role):
                raise Refused("candidate source asset role is invalid")
            if asset.get("read_only") is not True:
                raise Refused("candidate source asset must declare read_only true")
            if path in paths or role in roles:
                raise Refused("candidate source assets must have unique paths and roles")
            paths.add(path)
            roles.add(role)
    claimed = candidate.get("sha256")
    unsigned = dict(candidate)
    unsigned.pop("sha256")
    if not isinstance(claimed, str) or claimed != _digest(unsigned):
        raise Refused("candidate manifest self-digest is invalid")
    tasks = candidate.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise Refused("candidate manifest needs at least one task")
    expected_schema, extra_keys = _TASK_SCHEMAS[frontier]
    expected_keys = _TASK_COMMON_KEYS | extra_keys
    task_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != expected_keys:
            raise Refused("candidate task has undeclared or missing fields")
        if task.get("schema") != expected_schema or task.get("program") != PROGRAM or task.get("activation") is not False:
            raise Refused("candidate task has the wrong identity or activation state")
        if task.get("frontier") != frontier:
            raise Refused("candidate task frontier does not match its manifest")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id.startswith(f"{frontier}-") or task_id in task_ids:
            raise Refused("candidate task identifiers must be unique and frontier-bound")
        task_ids.add(task_id)
        if not isinstance(task.get("family"), str) or not task["family"] or not isinstance(task.get("request"), str) or not task["request"].strip():
            raise Refused("candidate task family or request is invalid")
        receipt = task.get("required_receipt")
        if not isinstance(receipt, list) or not receipt or not all(isinstance(item, str) and item for item in receipt):
            raise Refused("candidate task receipt contract is invalid")
        if not _candidate_tree_is_safe(task):
            raise Refused("candidate task exposes evaluator-only namespace")
        _validate_task_values(task, frontier=frontier, task_index=_task_index(task_id, frontier))


def candidate_is_structurally_safe(candidate: dict[str, Any]) -> bool:
    """Return whether public syntax/content constraints are met.

    This is intentionally not a semantic blindness certification.  The
    evaluator-answer custody claim is established only by the later G04/G10
    human-attested isolation controls.
    """
    try:
        verify_candidate_manifest(candidate)
    except Refused:
        return False
    return True


def candidate_is_blind(candidate: dict[str, Any]) -> bool:
    """Deprecated compatibility alias; use ``candidate_is_structurally_safe``.

    Retaining the alias avoids silently changing a caller's behavior while
    making its limitation explicit in the implementation and callers.
    """
    return candidate_is_structurally_safe(candidate)


def _main(argv: list[str] | None = None) -> int:
    """CLI: regenerate the committed LibriSpeech clip index from a local corpus."""
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate plans/substrate/tangible_next_launch/LIBRISPEECH_CLIP_INDEX.json "
            "from a local LibriSpeech corpus (MANIFEST order + FLAC STREAMINFO durations). "
            "Transcripts are never written."
        )
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=_AUDIO_CORPUS_ROOT,
        help="Path to librispeech_dev_clean corpus root (default: repo-relative sealed path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_AUDIO_CLIP_INDEX_PATH,
        help="Output path for the committed clip index JSON",
    )
    args = parser.parse_args(argv)
    try:
        payload = build_librispeech_clip_index(args.corpus_root, args.output)
    except Refused as error:
        print(f"Refused: {error}", file=sys.stderr)
        return 1
    print(
        f"wrote {args.output} clips={payload['clip_count']} "
        f"manifest_sha256={payload['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
