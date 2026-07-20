"""X0: charged event formation at the raw asynchronous boundary.

This module is a deterministic, generated-data study scaffold for Experiment 0 in
``18_event_sourced_coalition_substrate.md``.  It deliberately separates policy-visible packets and
public delayed consequences from evaluator-only event labels.  A favorable result remains a Gate-A
candidate pattern: scientific promotion is always blocked.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import stat
import statistics
import tempfile
import unicodedata
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable
from mop.substrate.events import canonical_bytes, canonical_sha256

ENVELOPE_SCHEMA = "mop-escs-x0-envelope/v1"
CONFIG_SCHEMA = "mop-escs-x0-config/v1"
AUTHORITY_SCHEMA = "mop-escs-x0-config-authority/v1"
IMPLEMENTATION_AUTHORITY_SCHEMA = "mop-escs-x0-implementation-authority/v1"
CHECKPOINT_SCHEMA = "mop-escs-x0-checkpoint/v1"
RECEIPT_SCHEMA = "mop-escs-x0-receipt/v1"
VERIFICATION_SCHEMA = "mop-escs-x0-verification/v1"
ROW_SCHEMA = "mop-escs-x0-seed-row/v1"
CLAIM_SCOPE = "generated-raw-event-formation-gate-a-candidate-only"
OFFICIAL_CONTRACT_ID = "escs-x0-v1-2026-07-12"
OFFICIAL_CONFIG_AUTHORITY_SHA256 = "75d7d998e2f4979febedbf65e8b79fdfccf83f3413b7656334ab888fdf138a1e"
OFFICIAL_IMPLEMENTATION_REVIEW_STATUS = "preregistered-scaffold-unexecuted"

ARM_NAMES = (
    "learned_event_former",
    "fixed_raw_delta",
    "periodic",
    "always_on",
    "header_only",
    "novelty",
    "uncertainty",
    "shuffled_rate_matched",
    "oracle_semantic_nonpromotable",
)
PROMOTABLE_COMPARATORS = (
    "fixed_raw_delta",
    "periodic",
    "shuffled_rate_matched",
    "novelty",
    "uncertainty",
)
EVIDENCE_STANDING = {
    "learned_event_former": "candidate_unverified",
    "fixed_raw_delta": "control_only",
    "periodic": "control_only",
    "always_on": "control_only",
    "header_only": "diagnostic_control_only",
    "novelty": "control_only",
    "uncertainty": "control_only",
    "shuffled_rate_matched": "control_only",
    "oracle_semantic_nonpromotable": "oracle_nonpromotable",
}
VISIBLE_PACKET_FIELDS = frozenset(
    {
        "packet_id",
        "world_token",
        "sensor_id",
        "capture_tick",
        "arrival_tick",
        "clock_reading_milli",
        "clock_uncertainty_milli",
        "signal_milli",
        "previous_signal_milli",
        "identity_token",
        "identity_confidence_milli",
        "payload_bytes",
        "header_delta_milli",
        "header_dominant_channel",
        "header_identity_confidence_milli",
        "header_clock_uncertainty_milli",
    }
)
EVALUATOR_FIELDS = frozenset(
    {
        "useful",
        "event_id",
        "event_type",
        "target_action",
        "referent_id",
        "irreducible_noise",
        "irrelevant_change",
        "storm",
        "deadline_tick",
        "consequence_tick",
    }
)
PUBLIC_CONSEQUENCE_FIELDS = frozenset(
    {"packet_id", "delivered_tick", "realized_action_value", "charged_work"}
)
WORK_COMPONENTS = (
    "raw_transport_and_adapters",
    "idle_polling",
    "event_formation",
    "header_construction",
    "indexing_and_queue",
    "downstream_cognition",
    "learning_and_feedback",
    "retained_state_byte_quanta",
    "serialization_and_receipts",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/experiment/escs_x0_event_formation.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "proof/ESCS_X0_EVENT_FORMATION.json"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "proof/ESCS_X0_EVENT_FORMATION.checkpoint.json"
DEFAULT_VERIFICATION_OUTPUT_PATH = REPO_ROOT / "proof/ESCS_X0_EVENT_FORMATION.verification.json"
DEFAULT_IMPLEMENTATION_AUTHORITY_PATH = (
    REPO_ROOT / "proof/ESCS_X0_EVENT_FORMATION.implementation-authority.json"
)
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_SCOPED_FILE_BYTES = 64 * 1024 * 1024


def _stable_int(*parts: Any, modulus: int = 2**63 - 1) -> int:
    return int.from_bytes(hashlib.sha256(canonical_bytes(list(parts))).digest()[:8], "big") % modulus


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    actual = set(value)
    wanted = set(expected)
    _require(
        actual == wanted,
        f"{label} keys differ: missing={sorted(wanted - actual)} extra={sorted(actual - wanted)}",
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_regular_file(path: Path, max_bytes: int, label: str) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        _require(before.st_size <= max_bytes, f"{label} exceeds its byte envelope")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    _require(identity_before == identity_after, f"{label} changed during read")
    raw = b"".join(chunks)
    _require(len(raw) == before.st_size, f"{label} size changed during read")
    return raw


def _file_receipt(path: Path) -> dict[str, Any]:
    raw = _read_regular_file(path, MAX_SCOPED_FILE_BYTES, f"scoped file {path}")
    try:
        label = str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        label = str(path.resolve())
    return {"path": label, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _require_distinct_paths(paths: Mapping[str, Path | str]) -> None:
    resolved = {label: Path(path).resolve() for label, path in paths.items()}
    logical: dict[str, list[str]] = defaultdict(list)
    inodes: dict[tuple[int, int], list[str]] = defaultdict(list)
    for label, path in resolved.items():
        logical[unicodedata.normalize("NFC", str(path)).casefold()].append(label)
        try:
            metadata = path.stat()
        except FileNotFoundError:
            continue
        _require(stat.S_ISREG(metadata.st_mode), f"artifact path {label!r} is not a regular file")
        inodes[(int(metadata.st_dev), int(metadata.st_ino))].append(label)
    collisions = [labels for labels in [*logical.values(), *inodes.values()] if len(labels) > 1]
    _require(not collisions, f"artifact path collision: {collisions}")


def _runtime_identity() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


def _load_envelope_snapshot(path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path).resolve()
    raw = _read_regular_file(source, MAX_ARTIFACT_BYTES, "X0 configuration")
    envelope = json.loads(raw)
    _require(isinstance(envelope, dict), "X0 configuration must be a mapping")
    _require_exact_keys(envelope, ("schema", "authority", "payload"), "X0 configuration envelope")
    _require(envelope["schema"] == ENVELOPE_SCHEMA, "unexpected X0 envelope schema")
    _require(isinstance(envelope["authority"], dict), "X0 config authority missing")
    _require(isinstance(envelope["payload"], dict), "X0 config payload missing")
    receipt = {
        "path": str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT) else str(source),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return envelope, receipt


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema") == CONFIG_SCHEMA, "unexpected X0 config schema")
    _require(config.get("claim_scope") == CLAIM_SCOPE, "X0 claim scope drift")
    _require(tuple(config.get("arms", ())) == ARM_NAMES, "X0 arm set or order drift")
    seed_value = config.get("seeds")
    fresh_value = config.get("fresh_verifier_seeds")
    _require(
        isinstance(seed_value, list) and len(seed_value) == 5,
        "X0 v1 requires exactly five paired seeds",
    )
    _require(
        isinstance(fresh_value, list) and len(fresh_value) == 5,
        "X0 v1 requires exactly five fresh verifier seeds",
    )
    seeds = cast(list[int], seed_value)
    fresh = cast(list[int], fresh_value)
    _require(
        all(isinstance(seed, int) and not isinstance(seed, bool) for seed in [*seeds, *fresh]), "bad seed"
    )
    _require(len(set(seeds)) == len(seeds), "paired seeds must be unique")
    _require(len(set(fresh)) == len(fresh), "fresh verifier seeds must be unique")
    _require(set(seeds).isdisjoint(fresh), "fresh verifier seeds overlap producer seeds")
    split_value = config.get("splits")
    _require(isinstance(split_value, dict), "X0 split declarations missing")
    splits = cast(dict[str, dict[str, Any]], split_value)
    _require(set(splits) == {"tune", "gate", "heldout", "fresh_verifier"}, "X0 split set drift")
    for name, split in splits.items():
        _require(int(split["episodes"]) > 0, f"{name} episodes must be positive")
        _require(bool(split["event_types"]), f"{name} event families missing")
        _require(bool(split["clock_families"]), f"{name} clock families missing")
    tune_events = set(splits["tune"]["event_types"])
    tune_clocks = set(splits["tune"]["clock_families"])
    _require(tune_events.isdisjoint(splits["heldout"]["event_types"]), "heldout event types must be unseen")
    _require(tune_clocks.isdisjoint(splits["heldout"]["clock_families"]), "heldout clocks must be unseen")
    _require(tune_events.isdisjoint(splits["fresh_verifier"]["event_types"]), "fresh events must be unseen")
    _require(
        tune_clocks.isdisjoint(splits["fresh_verifier"]["clock_families"]), "fresh clocks must be unseen"
    )
    world = cast(Mapping[str, Any], config["world"])
    _require(int(world["horizon_ticks"]) >= 32, "X0 horizon is too short to exercise all regimes")
    _require(int(world["sensor_count"]) >= 2, "X0 requires asynchronous multi-sensor input")
    _require(int(world["signal_channels"]) >= 4, "X0 raw signal is too narrow")
    fractions = [float(value) for value in world["useful_event_fractions"]]
    _require(fractions == sorted(set(fractions)), "useful event fractions must be sorted and unique")
    _require(all(0.02 < value < 0.98 for value in fractions), "useful event fraction outside horizon")
    _require(
        0 < float(world["idle_start_fraction"]) < float(world["idle_end_fraction"]) < 1, "bad idle block"
    )
    _require(
        0 < float(world["noisy_tv_start_fraction"]) < float(world["noisy_tv_end_fraction"]) < 1,
        "bad noisy-TV block",
    )
    required_clocks = {
        str(family) for row in splits.values() for family in cast(Sequence[Any], row["clock_families"])
    }
    _require(set(world["clock_specs"]) >= required_clocks, "clock spec missing")
    _require(set(config["visible_schema"]) == VISIBLE_PACKET_FIELDS, "visible packet schema drift")
    _require(set(config["evaluator_only_schema"]) == EVALUATOR_FIELDS, "evaluator-only schema drift")
    _require(
        set(config["public_consequence_schema"]) == PUBLIC_CONSEQUENCE_FIELDS,
        "public consequence schema drift",
    )
    _require(
        not set(config["visible_schema"]).intersection(config["evaluator_only_schema"]), "leakage in schemas"
    )
    _require(set(config["abstract_work"]["weights"]) == set(WORK_COMPONENTS), "work component set drift")
    _require(all(int(value) > 0 for value in config["abstract_work"]["weights"].values()), "bad work weight")
    criteria = config["criteria"]
    _require(float(criteria["max_utility_loss_vs_always_on"]) == 0.01, "utility SESOI drift")
    _require(float(criteria["min_work_saving_vs_always_on"]) == 0.25, "work SESOI drift")
    _require(bool(criteria["require_all_seed_directions"]), "paired direction requirement disabled")
    _require(config["verdict"]["scientific_promotion"] == "blocked", "promotion must remain blocked")
    resources = config["resources"]
    _require(resources["cpu_only"] and int(resources["worker_count"]) == 1, "X0 must be serial CPU-only")
    _require(
        not resources["allow_downloads"] and not resources["allow_external_data"], "external input forbidden"
    )
    _require(int(resources["max_packets_per_episode"]) > 0, "packet cap missing")
    _require(int(resources["max_queue_depth"]) > 0, "queue cap missing")
    _require(int(resources["max_policy_retained_state_bytes"]) > 0, "policy state cap missing")
    _require(int(config["work_costs"]["queue_entry_bytes"]) > 0, "queue byte charge missing")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH, *, exploratory: bool = False) -> dict[str, Any]:
    source = Path(path).resolve()
    envelope, _ = _load_envelope_snapshot(source)
    authority = envelope["authority"]
    _require_exact_keys(
        authority,
        ("schema", "mode", "contract_id", "payload_sha256"),
        "X0 config authority",
    )
    _require(authority["schema"] == AUTHORITY_SCHEMA, "X0 config-authority schema mismatch")
    payload = envelope["payload"]
    digest = canonical_sha256(payload)
    _require(authority["payload_sha256"] == digest, "X0 config payload hash mismatch")
    if not exploratory:
        _require(source == DEFAULT_CONFIG_PATH.resolve(), "official X0 execution requires repository config")
        _require(authority["mode"] == "official-preregistered", "official X0 authority mode required")
        _require(authority["contract_id"] == OFFICIAL_CONTRACT_ID, "X0 contract id mismatch")
        _require(digest == OFFICIAL_CONFIG_AUTHORITY_SHA256, "X0 config is not frozen official authority")
    _validate_config(payload)
    return copy.deepcopy(payload)


@dataclass(frozen=True, slots=True)
class VisiblePacket:

    packet_id: str
    world_token: str
    sensor_id: str
    capture_tick: int
    arrival_tick: int
    clock_reading_milli: int
    clock_uncertainty_milli: int
    signal_milli: tuple[int, ...]
    previous_signal_milli: tuple[int, ...]
    identity_token: str
    identity_confidence_milli: int
    payload_bytes: int
    header_delta_milli: int
    header_dominant_channel: int
    header_identity_confidence_milli: int
    header_clock_uncertainty_milli: int

    def __post_init__(self) -> None:
        _require(
            {field.name for field in dataclasses.fields(self)} == VISIBLE_PACKET_FIELDS,
            "visible schema drift",
        )
        _require(self.packet_id.startswith("packet:"), "packet identity namespace missing")
        _require(self.arrival_tick >= self.capture_tick, "packet arrived before capture")
        _require(len(self.signal_milli) == len(self.previous_signal_milli), "raw signal width changed")
        _require(self.payload_bytes > 0, "raw packet must charge payload bytes")
        _require(0 <= self.identity_confidence_milli <= 1000, "identity confidence out of range")


@dataclass(frozen=True, slots=True)
class EvaluatorTruth:

    useful: bool
    event_id: str | None
    event_type: str | None
    target_action: int | None
    referent_id: str | None
    irreducible_noise: bool
    irrelevant_change: bool
    storm: bool
    deadline_tick: int | None
    consequence_tick: int

    def __post_init__(self) -> None:
        _require({field.name for field in dataclasses.fields(self)} == EVALUATOR_FIELDS, "truth schema drift")
        _require(self.useful == (self.event_id is not None), "useful/event identity mismatch")
        if self.useful:
            _require(
                self.target_action is not None and self.deadline_tick is not None, "useful event incomplete"
            )


@dataclass(frozen=True, slots=True)
class PacketCase:
    visible: VisiblePacket
    evaluator: EvaluatorTruth


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    split: str
    seed: int
    episode: int
    clock_family: str
    event_types: tuple[str, ...]
    horizon_ticks: int
    packets: tuple[PacketCase, ...]
    idle_wall_ticks: int


@dataclass(frozen=True, slots=True)
class PublicConsequence:
    packet_id: str
    delivered_tick: int
    realized_action_value: float
    charged_work: int

    def __post_init__(self) -> None:
        _require(
            {field.name for field in dataclasses.fields(self)} == PUBLIC_CONSEQUENCE_FIELDS,
            "public consequence schema drift",
        )
        _require(math.isfinite(self.realized_action_value), "nonfinite public value")
        _require(self.charged_work > 0, "public feedback work must be charged")


@dataclass(frozen=True, slots=True)
class TrainingObservation:
    packet: VisiblePacket
    consequence: PublicConsequence


@dataclass(frozen=True, slots=True)
class PolicyDescriptor:
    policy_id: str
    evidence_standing: str
    oracle_access: bool


@runtime_checkable
class EventPolicy(Protocol):
    @property
    def descriptor(self) -> PolicyDescriptor: ...

    @property
    def retained_state_bytes(self) -> int: ...

    def fit(
        self,
        packets: Sequence[VisiblePacket],
        feedback: Sequence[TrainingObservation],
    ) -> Mapping[str, Any]: ...

    def score(self, packet: VisiblePacket) -> float: ...


def _signal_delta(packet: VisiblePacket) -> int:
    return sum(
        abs(left - right)
        for left, right in zip(packet.signal_milli, packet.previous_signal_milli, strict=True)
    )


def _feature_vector(packet: VisiblePacket) -> tuple[float, ...]:
    ranked = sorted(packet.signal_milli, reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
    scale = max(1, len(packet.signal_milli) * 3000)
    return (
        1.0,
        min(2.0, _signal_delta(packet) / scale),
        max(-2.0, min(2.0, margin / 2000.0)),
        packet.identity_confidence_milli / 1000.0,
        1.0 - min(1.0, packet.clock_uncertainty_milli / 1000.0),
        -1.0 if packet.signal_milli[-1] < 0 else 1.0,
    )


class DelayedConsequenceEventPolicy:

    def __init__(self, config: Mapping[str, Any], seed: int) -> None:
        learned = config["learned_policy"]
        self._seed = int(seed)
        self._epochs = int(learned["epochs"])
        self._learning_rate = float(learned["learning_rate"])
        self._target_admission_rate = float(learned["target_admission_rate"])
        self._weights = [0.0] * 6
        self._threshold = 0.5
        self._fitted = False

    @property
    def descriptor(self) -> PolicyDescriptor:
        return PolicyDescriptor(
            "builtin:delayed-consequence-linear-v1",
            "candidate_unverified",
            False,
        )

    @property
    def retained_state_bytes(self) -> int:
        return 8 * (len(self._weights) + 1)

    def _probability(self, packet: VisiblePacket) -> float:
        value = sum(
            weight * feature for weight, feature in zip(self._weights, _feature_vector(packet), strict=True)
        )
        value = max(-40.0, min(40.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def fit(
        self,
        packets: Sequence[VisiblePacket],
        feedback: Sequence[TrainingObservation],
    ) -> Mapping[str, Any]:
        _require(bool(packets), "learned policy received no visible tune packets")
        _require(bool(feedback), "learned policy received no public delayed feedback")
        for observation in feedback:
            _require(
                observation.packet.packet_id == observation.consequence.packet_id, "feedback join mismatch"
            )
        ordered = sorted(feedback, key=lambda row: (row.consequence.delivered_tick, row.packet.packet_id))
        operations = 0
        for epoch in range(self._epochs):
            epoch_rows = sorted(
                ordered,
                key=lambda row: _stable_int(self._seed, epoch, row.packet.packet_id),
            )
            for row in epoch_rows:
                features = _feature_vector(row.packet)
                prediction = self._probability(row.packet)
                target = 1.0 if row.consequence.realized_action_value > 0 else 0.0
                error = target - prediction
                for index, feature in enumerate(features):
                    self._weights[index] += self._learning_rate * error * feature
                operations += 5 * len(features)
        scores = sorted(self._probability(packet) for packet in packets)
        keep = max(1, min(len(scores), math.ceil(len(scores) * self._target_admission_rate)))
        self._threshold = scores[-keep]
        self._fitted = True
        return {
            "schema": "mop-escs-x0-training-receipt/v1",
            "policy_id": self.descriptor.policy_id,
            "visible_packet_count": len(packets),
            "public_feedback_count": len(feedback),
            "positive_public_consequence_count": sum(
                observation.consequence.realized_action_value > 0 for observation in feedback
            ),
            "feedback_delivery_min_tick": min(row.consequence.delivered_tick for row in feedback),
            "feedback_delivery_max_tick": max(row.consequence.delivered_tick for row in feedback),
            "training_operations": operations,
            "retained_state_bytes": self.retained_state_bytes,
            "threshold": self._threshold,
            "state_sha256": canonical_sha256(
                {
                    "weights": self._weights,
                    "threshold": self._threshold,
                    "policy_id": self.descriptor.policy_id,
                }
            ),
            "evaluator_labels_visible": False,
        }

    def score(self, packet: VisiblePacket) -> float:
        _require(self._fitted, "learned event policy used before fit")
        return self._probability(packet)

    @property
    def threshold(self) -> float:
        _require(self._fitted, "learned event policy used before fit")
        return self._threshold


PolicyFactory = Callable[[Mapping[str, Any], int], EventPolicy]

_POLICY_BOOTSTRAP_FIELDS = frozenset(
    {
        "policy_id",
        "epochs",
        "learning_rate",
        "target_admission_rate",
        "decision_threshold",
        "exploration_denominator",
        "training_signal",
        "freeze_before_heldout_and_fresh",
    }
)


def _policy_bootstrap(config: Mapping[str, Any]) -> Mapping[str, Any]:

    learned = cast(Mapping[str, Any], config["learned_policy"])
    _require(set(learned) == _POLICY_BOOTSTRAP_FIELDS, "learned policy bootstrap fields drifted")
    return MappingProxyType({"learned_policy": MappingProxyType(dict(learned))})


def _clock_emit(
    config: Mapping[str, Any],
    family: str,
    sensor: int,
    tick: int,
    seed: int,
    episode: int,
) -> bool:
    spec = config["world"]["clock_specs"][family]
    periods = [int(value) for value in spec["periods"]]
    period = periods[sensor % len(periods)]
    phase = (sensor * int(spec["phase_step"])) % period
    jitter = int(spec["jitter"])
    if jitter:
        phase = (
            phase
            + _stable_int(seed, episode, family, sensor, tick // period, modulus=2 * jitter + 1)
            - jitter
        ) % period
    return tick % period == phase


def _event_tick_set(config: Mapping[str, Any]) -> tuple[int, ...]:
    horizon = int(config["world"]["horizon_ticks"])
    return tuple(
        sorted(
            {
                min(horizon - 2, max(1, round(horizon * float(fraction))))
                for fraction in config["world"]["useful_event_fractions"]
            }
        )
    )


def _interval(config: Mapping[str, Any], prefix: str) -> tuple[int, int]:
    horizon = int(config["world"]["horizon_ticks"])
    start = round(horizon * float(config["world"][f"{prefix}_start_fraction"]))
    end = round(horizon * float(config["world"][f"{prefix}_end_fraction"]))
    return max(1, start), min(horizon - 1, max(start + 1, end))


def _make_signal(
    *,
    channels: int,
    target_action: int | None,
    flipped: bool,
    noise: bool,
    irrelevant: bool,
    seed_parts: tuple[Any, ...],
) -> tuple[int, ...]:
    rng = random.Random(_stable_int(*seed_parts))
    signal = [rng.randint(-80, 80) for _ in range(channels)]
    if noise:
        signal = [rng.randint(-3500, 3500) for _ in range(channels)]
    elif target_action is not None:
        distractor = (target_action + 1) % max(1, channels - 1)
        if flipped:
            signal[target_action] = 2200
            signal[distractor] = 3400
            signal[-1] = -3000
        else:
            signal[target_action] = 3400
            signal[distractor] = 1400
            signal[-1] = 3000
    elif irrelevant:
        direction = -1 if _stable_int(*seed_parts, "direction", modulus=2) else 1
        signal = [direction * (1100 + 170 * index) + rng.randint(-120, 120) for index in range(channels)]
        signal[-1] = rng.randint(-500, 500)
    return tuple(signal)


def generate_episode(
    config: Mapping[str, Any],
    *,
    seed: int,
    split: str,
    episode: int,
) -> EpisodeTrace:

    _require(split in config["splits"], f"unknown X0 split {split!r}")
    split_config = config["splits"][split]
    world = config["world"]
    horizon = int(world["horizon_ticks"])
    sensors = int(world["sensor_count"])
    channels = int(world["signal_channels"])
    clock_family = split_config["clock_families"][
        _stable_int(seed, split, episode, "clock", modulus=len(split_config["clock_families"]))
    ]
    event_types = tuple(str(value) for value in split_config["event_types"])
    useful_ticks = _event_tick_set(config)
    useful_by_tick = {tick: index for index, tick in enumerate(useful_ticks)}
    idle_start, idle_end = _interval(config, "idle")
    noisy_start, noisy_end = _interval(config, "noisy_tv")
    irrelevant_start, irrelevant_end = _interval(config, "irrelevant_burst")
    storm_tick = min(horizon - 2, max(1, round(horizon * float(world["storm_fraction"]))))
    if storm_tick not in useful_by_tick:
        closest = min(useful_ticks, key=lambda value: abs(value - storm_tick))
        storm_tick = closest
    previous = {sensor: tuple(0 for _ in range(channels)) for sensor in range(sensors)}
    packets: list[PacketCase] = []
    capture_ticks_with_packets: set[int] = set()
    sequence = 0
    for tick in range(horizon):
        if idle_start <= tick < idle_end:
            continue
        emitted: list[tuple[int, str]] = []
        for sensor in range(sensors):
            if _clock_emit(config, clock_family, sensor, tick, seed, episode):
                emitted.append((sensor, "clock"))
        if noisy_start <= tick < noisy_end:
            emitted.extend((sensor, "noisy_tv") for sensor in range(sensors))
        if irrelevant_start <= tick < irrelevant_end:
            emitted.extend((sensor, "irrelevant_high_rate") for sensor in range(sensors))
        if tick in useful_by_tick:
            emitted.append((useful_by_tick[tick] % sensors, "useful_forced"))
        if tick == storm_tick:
            for burst in range(int(world["storm_multiplicity"])):
                emitted.extend((sensor, f"storm:{burst}") for sensor in range(sensors))
        for sensor, source_kind in emitted:
            sequence += 1
            capture_ticks_with_packets.add(tick)
            useful_index = useful_by_tick.get(tick)
            useful = source_kind == "useful_forced"
            event_type = (
                event_types[useful_index % len(event_types)] if useful and useful_index is not None else None
            )
            target_action = (
                _stable_int(seed, split, episode, useful_index, "action", modulus=max(1, channels - 1))
                if useful
                else None
            )
            flipped = bool(event_type and ("flip" in event_type or "relation" in event_type))
            irreducible_noise = source_kind == "noisy_tv"
            irrelevant = source_kind == "irrelevant_high_rate" or source_kind.startswith("storm:")
            signal = _make_signal(
                channels=channels,
                target_action=target_action,
                flipped=flipped,
                noise=irreducible_noise,
                irrelevant=irrelevant,
                seed_parts=(seed, split, episode, tick, sensor, source_kind, sequence),
            )
            latency = 1 + _stable_int(seed, split, episode, tick, sensor, sequence, "latency", modulus=3)
            uncertainty = _stable_int(seed, split, episode, tick, sensor, "uncertainty", modulus=501)
            uncertain_identity = _stable_int(
                seed, split, episode, tick, sensor, sequence, "identity", modulus=10_000
            ) < round(10_000 * float(world["uncertain_identity_rate"]))
            identity_confidence = 250 if uncertain_identity else 900
            identity_token = "source:unknown" if uncertain_identity else f"source:sensor-{sensor}"
            delta = sum(abs(left - right) for left, right in zip(signal, previous[sensor], strict=True))
            ranked = sorted(range(channels - 1), key=lambda index: (signal[index], -index), reverse=True)
            dominant = ranked[0]
            arrival = tick + latency
            world_token = canonical_sha256([seed, split, episode])[:16]
            sensor_id = f"sensor:{sensor}"
            clock_reading = tick * (1000 + 17 * sensor) + int(uncertainty)
            payload_bytes = int(world["base_payload_bytes"]) + 8 * channels
            visible_payload = {
                "world_token": world_token,
                "sensor_id": sensor_id,
                "capture_tick": tick,
                "arrival_tick": arrival,
                "clock_reading_milli": clock_reading,
                "clock_uncertainty_milli": int(uncertainty),
                "signal_milli": list(signal),
                "previous_signal_milli": list(previous[sensor]),
                "identity_token": identity_token,
                "identity_confidence_milli": identity_confidence,
                "payload_bytes": payload_bytes,
                "header_delta_milli": delta,
                "header_dominant_channel": dominant,
                "header_identity_confidence_milli": identity_confidence,
                "header_clock_uncertainty_milli": int(uncertainty),
            }
            packet_id = f"packet:{canonical_sha256(visible_payload)}"
            visible = VisiblePacket(
                packet_id=packet_id,
                world_token=world_token,
                sensor_id=sensor_id,
                capture_tick=tick,
                arrival_tick=arrival,
                clock_reading_milli=clock_reading,
                clock_uncertainty_milli=int(uncertainty),
                signal_milli=signal,
                previous_signal_milli=previous[sensor],
                identity_token=identity_token,
                identity_confidence_milli=identity_confidence,
                payload_bytes=payload_bytes,
                header_delta_milli=delta,
                header_dominant_channel=dominant,
                header_identity_confidence_milli=identity_confidence,
                header_clock_uncertainty_milli=int(uncertainty),
            )
            event_id = (
                f"event:{canonical_sha256([seed, split, episode, useful_index, event_type])}"
                if useful
                else None
            )
            consequence_delay = int(world["consequence_delay_ticks"])
            deadline = tick + int(world["event_deadline_ticks"]) if useful else None
            evaluator = EvaluatorTruth(
                useful=useful,
                event_id=event_id,
                event_type=event_type,
                target_action=int(target_action) if target_action is not None else None,
                referent_id=f"referent:{useful_index % 3}" if useful and useful_index is not None else None,
                irreducible_noise=irreducible_noise,
                irrelevant_change=irrelevant,
                storm=tick == storm_tick,
                deadline_tick=deadline,
                consequence_tick=tick + consequence_delay,
            )
            packets.append(PacketCase(visible, evaluator))
            previous[sensor] = signal
    packets.sort(key=lambda row: (row.visible.arrival_tick, row.visible.packet_id))
    cap = int(config["resources"]["max_packets_per_episode"])
    _require(len(packets) <= cap, f"generated X0 packet count {len(packets)} exceeds cap {cap}")
    return EpisodeTrace(
        split=split,
        seed=seed,
        episode=episode,
        clock_family=str(clock_family),
        event_types=event_types,
        horizon_ticks=horizon,
        packets=tuple(packets),
        idle_wall_ticks=horizon - len(capture_ticks_with_packets),
    )


def _decode_action(packet: VisiblePacket, *, header_only: bool) -> int:
    if header_only:
        return int(packet.header_dominant_channel)
    candidates = list(range(len(packet.signal_milli) - 1))
    ranked = sorted(candidates, key=lambda index: (packet.signal_milli[index], -index), reverse=True)
    return int(ranked[1] if packet.signal_milli[-1] < 0 and len(ranked) > 1 else ranked[0])


def _public_action_value(case: PacketCase) -> float:
    if not case.evaluator.useful:
        return 0.0
    action = _decode_action(case.visible, header_only=False)
    return 1.0 if action == case.evaluator.target_action else 0.0


def _train_policy(
    config: Mapping[str, Any],
    *,
    seed: int,
    policy_factory: PolicyFactory | None,
) -> tuple[EventPolicy, dict[str, Any]]:
    factory = policy_factory or (lambda cfg, value: DelayedConsequenceEventPolicy(cfg, value))
    policy = factory(_policy_bootstrap(config), seed)
    _require(isinstance(policy, EventPolicy), "injected event former does not satisfy EventPolicy")
    _require(not policy.descriptor.oracle_access, "learned/injected policy cannot receive oracle access")
    _require(policy.descriptor.evidence_standing == "candidate_unverified", "learned policy evidence drift")
    tune = [
        generate_episode(config, seed=seed, split="tune", episode=episode)
        for episode in range(int(config["splits"]["tune"]["episodes"]))
    ]
    packets = tuple(case.visible for trace in tune for case in trace.packets)
    denominator = int(config["learned_policy"]["exploration_denominator"])
    feedback: list[TrainingObservation] = []
    for trace in tune:
        for case in trace.packets:
            if _stable_int(seed, case.visible.packet_id, "training-explore", modulus=denominator) != 0:
                continue
            consequence = PublicConsequence(
                packet_id=case.visible.packet_id,
                delivered_tick=case.evaluator.consequence_tick,
                realized_action_value=_public_action_value(case),
                charged_work=int(config["work_costs"]["public_feedback"]),
            )
            feedback.append(TrainingObservation(case.visible, consequence))
    if not any(row.consequence.realized_action_value > 0 for row in feedback):
        exploration_order = sorted(
            (case for trace in tune for case in trace.packets),
            key=lambda row: _stable_int(seed, row.visible.packet_id, "fallback-explore"),
        )
        for case in exploration_order:
            consequence = PublicConsequence(
                packet_id=case.visible.packet_id,
                delivered_tick=case.evaluator.consequence_tick,
                realized_action_value=_public_action_value(case),
                charged_work=int(config["work_costs"]["public_feedback"]),
            )
            feedback.append(TrainingObservation(case.visible, consequence))
            if consequence.realized_action_value > 0:
                break
    receipt = dict(policy.fit(packets, tuple(feedback)))
    _require(receipt.get("evaluator_labels_visible") is False, "learned policy claims evaluator visibility")
    return policy, receipt


def leakage_gate(config: Mapping[str, Any], policy: EventPolicy | None = None) -> dict[str, Any]:
    problems: list[str] = []
    if set(config["visible_schema"]) != VISIBLE_PACKET_FIELDS:
        problems.append("visible schema differs from the frozen packet interface")
    if set(config["evaluator_only_schema"]) != EVALUATOR_FIELDS:
        problems.append("evaluator-only schema differs from the frozen scorer interface")
    overlap = set(config["visible_schema"]).intersection(config["evaluator_only_schema"])
    if overlap:
        problems.append(f"visible/evaluator schema overlap: {sorted(overlap)}")
    forbidden = {"event_type", "target_action", "referent_id", "useful", "event_id", "deadline_tick"}
    if forbidden.intersection(config["visible_schema"]):
        problems.append("semantic evaluator fields crossed the raw packet boundary")
    if set(config["public_consequence_schema"]) != PUBLIC_CONSEQUENCE_FIELDS:
        problems.append("public consequence schema contains unregistered fields")
    if EVIDENCE_STANDING["oracle_semantic_nonpromotable"] != "oracle_nonpromotable":
        problems.append("oracle arm lost its nonpromotable taint")
    if policy is not None:
        if policy.descriptor.oracle_access:
            problems.append("learned/injected policy declares oracle access")
        if policy.descriptor.evidence_standing != "candidate_unverified":
            problems.append("learned/injected policy evidence standing drifted")
    return {
        "schema": "mop-escs-x0-leakage-gate/v1",
        "passed": not problems,
        "problems": problems,
        "visible_fields": sorted(VISIBLE_PACKET_FIELDS),
        "evaluator_only_fields": sorted(EVALUATOR_FIELDS),
        "public_consequence_fields": sorted(PUBLIC_CONSEQUENCE_FIELDS),
        "policy_boundary": "VisiblePacket plus delayed PublicConsequence only",
        "oracle_nonpromotable": True,
    }


def _policy_admissions(
    config: Mapping[str, Any],
    trace: EpisodeTrace,
    arm: str,
    policy: EventPolicy,
    learned_indices: set[int],
) -> tuple[set[int], list[float]]:
    packets = trace.packets
    scores: list[float] = []
    selected: set[int] = set()
    if arm == "learned_event_former":
        threshold = float(getattr(policy, "threshold", config["learned_policy"]["decision_threshold"]))
        for index, case in enumerate(packets):
            score = float(policy.score(case.visible))
            _require(math.isfinite(score) and 0 <= score <= 1, "learned policy emitted invalid score")
            scores.append(score)
            if score >= threshold:
                selected.add(index)
    elif arm == "fixed_raw_delta":
        threshold = int(config["controls"]["fixed_delta_threshold_milli"])
        for index, case in enumerate(packets):
            score = min(1.0, _signal_delta(case.visible) / max(1, threshold * 2))
            scores.append(score)
            if _signal_delta(case.visible) >= threshold:
                selected.add(index)
    elif arm == "periodic":
        period = int(config["controls"]["period_ticks"])
        for index, case in enumerate(packets):
            admitted = case.visible.arrival_tick % period == 0
            scores.append(float(admitted))
            if admitted:
                selected.add(index)
    elif arm == "always_on":
        selected = set(range(len(packets)))
        scores = [1.0] * len(packets)
    elif arm == "header_only":
        threshold = int(config["controls"]["header_delta_threshold_milli"])
        for index, case in enumerate(packets):
            score = min(1.0, case.visible.header_delta_milli / max(1, threshold * 2))
            scores.append(score)
            if case.visible.header_delta_milli >= threshold:
                selected.add(index)
    elif arm == "novelty":
        threshold = int(config["controls"]["novelty_threshold_milli"])
        running: dict[str, list[float]] = {}
        counts: dict[str, int] = defaultdict(int)
        for index, case in enumerate(packets):
            key = case.visible.sensor_id
            current = list(case.visible.signal_milli)
            mean = running.setdefault(key, [0.0] * len(current))
            distance = sum(abs(value - mean[channel]) for channel, value in enumerate(current))
            score = min(1.0, distance / max(1, threshold * 2))
            scores.append(score)
            if distance >= threshold:
                selected.add(index)
            counts[key] += 1
            rate = 1.0 / counts[key]
            for channel, value in enumerate(current):
                mean[channel] += rate * (value - mean[channel])
    elif arm == "uncertainty":
        threshold = int(config["controls"]["uncertainty_threshold_milli"])
        for index, case in enumerate(packets):
            uncertainty = max(
                1000 - case.visible.identity_confidence_milli,
                case.visible.clock_uncertainty_milli,
            )
            score = uncertainty / 1000.0
            scores.append(score)
            if uncertainty >= threshold:
                selected.add(index)
    elif arm == "shuffled_rate_matched":
        count = len(learned_indices)
        order = sorted(
            range(len(packets)),
            key=lambda index: _stable_int(
                trace.seed, trace.split, trace.episode, packets[index].visible.packet_id, "shuffle"
            ),
        )
        selected = set(order[:count])
        rate = count / len(packets) if packets else 0.0
        scores = [rate] * len(packets)
    elif arm == "oracle_semantic_nonpromotable":
        for index, case in enumerate(packets):
            scores.append(float(case.evaluator.useful))
            if case.evaluator.useful:
                selected.add(index)
    else:
        raise ValueError(f"unknown X0 arm {arm!r}")
    return selected, scores


def _work_total(components: Mapping[str, int], config: Mapping[str, Any]) -> int:
    weights = config["abstract_work"]["weights"]
    _require(set(components) == set(WORK_COMPONENTS), "incomplete lifecycle work vector")
    return sum(int(components[name]) * int(weights[name]) for name in WORK_COMPONENTS)


def _brier(scores: Sequence[float], labels: Sequence[int]) -> float:
    return (
        statistics.fmean((score - label) ** 2 for score, label in zip(scores, labels, strict=True))
        if scores
        else 0.0
    )


def _calibration_error(scores: Sequence[float], labels: Sequence[int], bins: int = 5) -> float:
    if not scores:
        return 0.0
    total = 0.0
    for bucket in range(bins):
        lower = bucket / bins
        upper = (bucket + 1) / bins
        indices = [
            index
            for index, value in enumerate(scores)
            if lower <= value < upper or (bucket == bins - 1 and value == 1.0)
        ]
        if not indices:
            continue
        confidence = statistics.fmean(scores[index] for index in indices)
        accuracy = statistics.fmean(labels[index] for index in indices)
        total += len(indices) / len(scores) * abs(confidence - accuracy)
    return total


def _simulate_arm(
    config: Mapping[str, Any],
    traces: Sequence[EpisodeTrace],
    *,
    arm: str,
    policy: EventPolicy,
    training_receipt: Mapping[str, Any],
    learned_admissions: Mapping[int, set[int]],
) -> dict[str, Any]:
    work = {name: 0 for name in WORK_COMPONENTS}
    work_costs = config["work_costs"]
    resources = config["resources"]
    all_scores: list[float] = []
    all_labels: list[int] = []
    episode_metrics: list[dict[str, Any]] = []
    header_bytes = 0
    retained_bytes = int(policy.retained_state_bytes) if arm == "learned_event_former" else 0
    _require(
        0 <= retained_bytes <= int(resources["max_policy_retained_state_bytes"]),
        "policy retained state exceeds its registered cap",
    )
    queue_retained_byte_ticks = 0
    peak_queue_bytes = 0
    for trace in traces:
        learned_indices = learned_admissions.get(trace.episode, set())
        selected, scores = _policy_admissions(config, trace, arm, policy, learned_indices)
        cases = trace.packets
        if arm == "shuffled_rate_matched":
            _require(len(selected) == len(learned_indices), "shuffled control lost exact admission rate")
        all_scores.extend(scores)
        all_labels.extend(int(case.evaluator.useful) for case in cases)
        work["raw_transport_and_adapters"] += sum(
            case.visible.payload_bytes + int(work_costs["adapter_per_packet"]) for case in cases
        )
        work["idle_polling"] += trace.idle_wall_ticks * int(work_costs["idle_poll_per_tick"])
        if arm == "always_on":
            work["idle_polling"] += trace.idle_wall_ticks * int(work_costs["always_on_idle_tick"])
        work["event_formation"] += len(cases) * int(work_costs["policy_per_packet"][arm])
        if arm in {"learned_event_former", "header_only"}:
            work["header_construction"] += len(cases) * int(work_costs["header_per_packet"])
            header_bytes += len(cases) * int(work_costs["header_encoded_bytes"])
        work["retained_state_byte_quanta"] += (
            retained_bytes * trace.horizon_ticks // int(config["abstract_work"]["retained_byte_quantum"])
        )
        arrivals: dict[int, list[int]] = defaultdict(list)
        for index in sorted(selected):
            arrivals[cases[index].visible.arrival_tick].append(index)
        queue: deque[int] = deque()
        processed: list[tuple[int, int]] = []
        dropped: list[int] = []
        max_depth = 0
        max_queue_growth = 0
        prior_end_depth = 0
        end_tick = trace.horizon_ticks + int(resources["max_drain_ticks"])
        for tick in range(end_tick + 1):
            for index in arrivals.get(tick, []):
                work["indexing_and_queue"] += int(work_costs["queue_admit"])
                if len(queue) >= int(resources["max_queue_depth"]):
                    dropped.append(index)
                    work["indexing_and_queue"] += int(work_costs["queue_drop"])
                else:
                    queue.append(index)
            max_depth = max(max_depth, len(queue))
            peak_queue_bytes = max(
                peak_queue_bytes,
                len(queue) * int(work_costs["queue_entry_bytes"]),
            )
            max_queue_growth = max(max_queue_growth, len(queue) - prior_end_depth)
            for _ in range(min(len(queue), int(resources["downstream_capacity_per_tick"]))):
                index = queue.popleft()
                processed.append((index, tick))
                work["indexing_and_queue"] += int(work_costs["queue_remove"])
                work["downstream_cognition"] += int(work_costs["downstream_per_activation"])
            max_depth = max(max_depth, len(queue))
            queued_bytes = len(queue) * int(work_costs["queue_entry_bytes"])
            queue_retained_byte_ticks += queued_bytes
            prior_end_depth = len(queue)
        unprocessed = list(queue)
        event_truth: dict[str, EvaluatorTruth] = {}
        for case in cases:
            if case.evaluator.event_id is not None:
                event_truth[case.evaluator.event_id] = case.evaluator
        detections: dict[str, tuple[int, int]] = {}
        useful_processed = 0
        noisy_processed = 0
        irrelevant_processed = 0
        false_actions = 0
        storm_processed = 0
        for index, processed_tick in processed:
            case = cases[index]
            truth = case.evaluator
            if truth.irreducible_noise:
                noisy_processed += 1
            if truth.irrelevant_change:
                irrelevant_processed += 1
            if truth.storm:
                storm_processed += 1
            if not truth.useful:
                false_actions += 1
                continue
            useful_processed += 1
            action = _decode_action(case.visible, header_only=arm == "header_only")
            if action != truth.target_action or truth.event_id is None or truth.deadline_tick is None:
                continue
            if processed_tick > truth.deadline_tick:
                continue
            prior = detections.get(truth.event_id)
            candidate = (processed_tick, case.visible.capture_tick)
            if prior is None or candidate < prior:
                detections[truth.event_id] = candidate
        delays = [processed_tick - capture_tick for processed_tick, capture_tick in detections.values()]
        captured_value = sum(
            max(0.0, 1.0 - float(config["evaluation"]["delay_penalty_per_tick"]) * delay) for delay in delays
        )
        false_penalty = (
            float(config["evaluation"]["false_action_penalty"]) * false_actions / max(1, len(cases))
        )
        utility = captured_value / max(1, len(event_truth)) - false_penalty
        precision = len(detections) / max(1, len(processed))
        recall = len(detections) / max(1, len(event_truth))
        missed_ids = set(event_truth) - set(detections)
        storm_event_ids = {
            case.evaluator.event_id for case in cases if case.evaluator.useful and case.evaluator.storm
        }
        storm_misses = len({value for value in storm_event_ids if value is not None}.intersection(missed_ids))
        episode_metrics.append(
            {
                "episode": trace.episode,
                "clock_family": trace.clock_family,
                "event_types": list(trace.event_types),
                "packet_count": len(cases),
                "useful_event_count": len(event_truth),
                "admitted_count": len(selected),
                "processed_count": len(processed),
                "dropped_count": len(dropped),
                "unprocessed_count": len(unprocessed),
                "event_precision": precision,
                "event_recall": recall,
                "mean_detection_delay": statistics.fmean(delays) if delays else float(trace.horizon_ticks),
                "utility": utility,
                "noisy_tv_activation_rate": noisy_processed
                / max(1, sum(case.evaluator.irreducible_noise for case in cases)),
                "irrelevant_activation_rate": irrelevant_processed
                / max(1, sum(case.evaluator.irrelevant_change for case in cases)),
                "false_action_count": false_actions,
                "storm_processed_count": storm_processed,
                "storm_deadline_misses": storm_misses,
                "deadline_misses": len(missed_ids),
                "max_queue_depth": max_depth,
                "max_queue_growth_per_tick": max_queue_growth,
                "queue_end_depth": len(unprocessed),
                "queue_stable": not dropped and not unprocessed,
                "admission_sha256": canonical_sha256(sorted(selected)),
            }
        )
    if arm == "learned_event_former":
        work["learning_and_feedback"] += int(training_receipt["training_operations"])
        work["learning_and_feedback"] += int(training_receipt["public_feedback_count"]) * int(
            config["work_costs"]["public_feedback"]
        )
    retained_quantum = int(config["abstract_work"]["retained_byte_quantum"])
    work["retained_state_byte_quanta"] += math.ceil(queue_retained_byte_ticks / retained_quantum)
    work["serialization_and_receipts"] += header_bytes
    work["serialization_and_receipts"] += int(config["work_costs"]["receipt_per_episode"]) * len(traces)
    total_work = _work_total(work, config)

    def mean(key: str) -> float:
        return statistics.fmean(float(row[key]) for row in episode_metrics) if episode_metrics else 0.0

    packet_count = sum(int(row["packet_count"]) for row in episode_metrics)
    admitted_count = sum(int(row["admitted_count"]) for row in episode_metrics)
    retrospectively_useful_admitted_count = sum(
        sum(trace.packets[index].evaluator.useful for index in learned_admissions.get(trace.episode, set()))
        if arm == "learned_event_former"
        else 0
        for trace in traces
    )
    if arm != "learned_event_former":
        retrospectively_useful_admitted_count = sum(
            sum(
                trace.packets[index].evaluator.useful
                for index in _policy_admissions(
                    config,
                    trace,
                    arm,
                    policy,
                    learned_admissions.get(trace.episode, set()),
                )[0]
            )
            for trace in traces
        )
    wall_ticks = sum(trace.horizon_ticks for trace in traces)
    dropped_count = sum(int(row["dropped_count"]) for row in episode_metrics)
    unprocessed_count = sum(int(row["unprocessed_count"]) for row in episode_metrics)
    return {
        "arm": arm,
        "evidence_standing": EVIDENCE_STANDING[arm],
        "oracle_access": arm == "oracle_semantic_nonpromotable",
        "scientific_promotion": False,
        "episode_count": len(episode_metrics),
        "packet_count": packet_count,
        "raw_update_rate_per_wall_tick": packet_count / max(1, wall_ticks),
        "candidate_count": packet_count,
        "admitted_count": admitted_count,
        "admission_rate": admitted_count / max(1, packet_count),
        "discarded_candidate_count": packet_count - admitted_count,
        "retrospectively_useful_admitted_count": retrospectively_useful_admitted_count,
        "retrospectively_useful_admission_rate": retrospectively_useful_admitted_count
        / max(1, admitted_count),
        "mean_utility": mean("utility"),
        "event_precision": mean("event_precision"),
        "event_recall": mean("event_recall"),
        "mean_detection_delay": mean("mean_detection_delay"),
        "noisy_tv_false_activation_rate": mean("noisy_tv_activation_rate"),
        "irrelevant_false_activation_rate": mean("irrelevant_activation_rate"),
        "deadline_misses": sum(int(row["deadline_misses"]) for row in episode_metrics),
        "storm_deadline_misses": sum(int(row["storm_deadline_misses"]) for row in episode_metrics),
        "queue_stable": all(bool(row["queue_stable"]) for row in episode_metrics),
        "max_queue_depth": max((int(row["max_queue_depth"]) for row in episode_metrics), default=0),
        "max_queue_growth_per_tick": max(
            (int(row["max_queue_growth_per_tick"]) for row in episode_metrics), default=0
        ),
        "queue_drop_count": dropped_count,
        "queue_unprocessed_count": unprocessed_count,
        "header_encoded_bytes": header_bytes,
        "header_operations": int(work["header_construction"]),
        "header_only_action_value": mean("utility") if arm == "header_only" else None,
        "calibration": {
            "brier": _brier(all_scores, all_labels),
            "expected_calibration_error": _calibration_error(all_scores, all_labels),
            "scored_packet_count": len(all_scores),
        },
        "total_lifecycle_work": total_work,
        "work_components": work,
        "idle_adapter_and_event_former_work": int(work["idle_polling"]),
        "retained_state_bytes": retained_bytes + peak_queue_bytes,
        "policy_retained_state_bytes": retained_bytes,
        "peak_queue_retained_state_bytes": peak_queue_bytes,
        "retained_state_byte_ticks": retained_bytes * wall_ticks + queue_retained_byte_ticks,
        "queue_retained_state_byte_ticks": queue_retained_byte_ticks,
        "per_episode": episode_metrics,
    }


def _trace_properties(traces: Sequence[EpisodeTrace]) -> dict[str, Any]:
    packets = [case for trace in traces for case in trace.packets]
    useful_events = {case.evaluator.event_id for case in packets if case.evaluator.event_id is not None}
    useful_packets = sum(case.evaluator.useful for case in packets)
    irrelevant_packets = sum(case.evaluator.irrelevant_change for case in packets)
    noisy_packets = sum(case.evaluator.irreducible_noise for case in packets)
    storm_packets = sum(case.evaluator.storm for case in packets)
    uncertain_packets = sum(case.visible.identity_token == "source:unknown" for case in packets)
    wall_ticks = sum(trace.horizon_ticks for trace in traces)
    idle_ticks = sum(trace.idle_wall_ticks for trace in traces)
    arrival_ticks = [case.visible.arrival_tick for case in packets]
    capture_ticks = [case.visible.capture_tick for case in packets]
    return {
        "episode_count": len(traces),
        "raw_packet_count": len(packets),
        "useful_event_count": len(useful_events),
        "useful_packet_count": useful_packets,
        "useful_update_rate": useful_packets / max(1, len(packets)),
        "idle_wall_tick_count": idle_ticks,
        "idle_fraction": idle_ticks / max(1, wall_ticks),
        "irrelevant_packet_count": irrelevant_packets,
        "irrelevant_update_fraction": irrelevant_packets / max(1, len(packets)),
        "noisy_tv_packet_count": noisy_packets,
        "storm_packet_count": storm_packets,
        "uncertain_identity_packet_count": uncertain_packets,
        "clock_families": sorted({trace.clock_family for trace in traces}),
        "event_types": sorted(set().union(*(set(trace.event_types) for trace in traces))),
        "asynchronous_arrival_observed": any(
            arrival != capture for arrival, capture in zip(arrival_ticks, capture_ticks, strict=True)
        ),
        "max_packets_same_arrival_tick": max(
            (sum(case.visible.arrival_tick == tick for case in packets) for tick in set(arrival_ticks)),
            default=0,
        ),
    }


def run_seed(
    config: Mapping[str, Any],
    *,
    seed: int,
    split: str,
    policy_factory: PolicyFactory | None = None,
) -> dict[str, Any]:
    _validate_config(config)
    _require(split in {"gate", "heldout", "fresh_verifier"}, "X0 evaluation split is not runnable")
    policy, training = _train_policy(config, seed=seed, policy_factory=policy_factory)
    gate = leakage_gate(config, policy)
    _require(gate["passed"], "X0 leakage gate failed before arm execution")
    traces = [
        generate_episode(config, seed=seed, split=split, episode=episode)
        for episode in range(int(config["splits"][split]["episodes"]))
    ]
    learned_admissions: dict[int, set[int]] = {}
    for trace in traces:
        selected, _ = _policy_admissions(config, trace, "learned_event_former", policy, set())
        learned_admissions[trace.episode] = selected
    arms = {
        arm: _simulate_arm(
            config,
            traces,
            arm=arm,
            policy=policy,
            training_receipt=training,
            learned_admissions=learned_admissions,
        )
        for arm in ARM_NAMES
    }
    _require(
        arms["learned_event_former"]["admitted_count"] == arms["shuffled_rate_matched"]["admitted_count"],
        "shuffled admission count differs from learned arm",
    )
    core = {
        "schema": ROW_SCHEMA,
        "seed": int(seed),
        "split": split,
        "claim_scope": CLAIM_SCOPE,
        "training": training,
        "leakage_gate": gate,
        "trace_properties": _trace_properties(traces),
        "arms": arms,
        "scientific_promotion": False,
    }
    row = dict(core)
    row["row_sha256"] = canonical_sha256(core)
    return row


def difficulty_gate(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    criteria = config["difficulty_gate"]
    complete = len(rows) == len(config["seeds"])
    checks = {
        "all_required_gate_seeds_complete": complete,
        "always_on_action_value": complete
        and all(
            float(row["arms"]["always_on"]["mean_utility"]) >= float(criteria["min_always_on_utility"])
            for row in rows
        ),
        "decision_relevant_change_is_sparse": complete
        and all(
            float(row["trace_properties"]["useful_update_rate"]) <= float(criteria["max_useful_update_rate"])
            for row in rows
        ),
        "long_idle_intervals_present": complete
        and all(
            float(row["trace_properties"]["idle_fraction"]) >= float(criteria["min_idle_fraction"])
            for row in rows
        ),
        "irrelevant_high_rate_change_present": complete
        and all(
            float(row["trace_properties"]["irrelevant_update_fraction"])
            >= float(criteria["min_irrelevant_fraction"])
            for row in rows
        ),
        "noise_storm_identity_and_async_present": complete
        and all(
            int(row["trace_properties"]["noisy_tv_packet_count"]) > 0
            and int(row["trace_properties"]["storm_packet_count"]) > 0
            and int(row["trace_properties"]["uncertain_identity_packet_count"]) > 0
            and bool(row["trace_properties"]["asynchronous_arrival_observed"])
            and int(row["trace_properties"]["max_packets_same_arrival_tick"]) > 1
            for row in rows
        ),
        "leakage_gate_passed": complete and all(bool(row["leakage_gate"]["passed"]) for row in rows),
    }
    return {
        "schema": "mop-escs-x0-difficulty-gate/v1",
        "status": "complete" if complete else "partial",
        "passed": complete and all(checks.values()),
        "checks": checks,
        "failure_interpretation": "invalid_bed_not_mechanism_null",
    }


def mean_ci(values: Sequence[float], t_critical: float) -> dict[str, float]:
    _require(bool(values), "cannot summarize empty paired evidence")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"mean": mean, "lower": mean, "upper": mean, "half_width": 0.0}
    half = t_critical * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "half_width": half}


def aggregate_rows(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    split = str(rows[0]["split"]) if rows else "heldout"
    expected_ids = (
        [int(seed) for seed in config["fresh_verifier_seeds"]]
        if split == "fresh_verifier"
        else [int(seed) for seed in config["seeds"]]
    )
    expected = len(expected_ids)
    if len(rows) != expected:
        return {
            "status": "partial",
            "completed_seed_count": len(rows),
            "required_seed_count": expected,
            "verdict": "pending",
            "scientific_promotion": False,
        }
    _require(all(str(row["split"]) == split for row in rows), "X0 aggregate mixes split identities")
    _require([int(row["seed"]) for row in rows] == expected_ids, "X0 aggregate seed identity drift")
    criteria = config["criteria"]
    t_critical = float(config["evaluation"]["t_critical_95"])
    utility_losses = []
    work_savings = []
    noise_excess = []
    calibration = []
    control_advantages: dict[str, list[float]] = {name: [] for name in PROMOTABLE_COMPARATORS}
    all_seed_directions: dict[str, bool] = {}
    header_diagnostics: list[bool] = []
    for row in rows:
        learned = row["arms"]["learned_event_former"]
        always = row["arms"]["always_on"]
        shuffled = row["arms"]["shuffled_rate_matched"]
        header = row["arms"]["header_only"]
        loss = float(always["mean_utility"]) - float(learned["mean_utility"])
        saving = 1.0 - float(learned["total_lifecycle_work"]) / max(
            1.0, float(always["total_lifecycle_work"])
        )
        utility_losses.append(loss)
        work_savings.append(saving)
        noise_excess.append(
            float(learned["noisy_tv_false_activation_rate"])
            - float(shuffled["noisy_tv_false_activation_rate"])
        )
        calibration.append(float(learned["calibration"]["expected_calibration_error"]))
        seed_ok = loss <= float(criteria["max_utility_loss_vs_always_on"]) and saving >= float(
            criteria["min_work_saving_vs_always_on"]
        )
        for control_name in PROMOTABLE_COMPARATORS:
            control = row["arms"][control_name]
            utility_margin = float(learned["mean_utility"]) - float(control["mean_utility"])
            relative_saving = 1.0 - float(learned["total_lifecycle_work"]) / max(
                1.0, float(control["total_lifecycle_work"])
            )
            contribution = min(
                utility_margin + float(criteria["pareto_utility_tolerance"]),
                relative_saving,
            )
            control_advantages[control_name].append(contribution)
            seed_ok = seed_ok and contribution > 0
        all_seed_directions[str(row["seed"])] = seed_ok
        header_matches = float(header["mean_utility"]) >= float(learned["mean_utility"]) - float(
            criteria["header_utility_tolerance"]
        )
        header_is_no_worse_and_no_more_expensive = header_matches and float(
            header["total_lifecycle_work"]
        ) <= float(learned["total_lifecycle_work"])
        header_diagnostics.append(not header_is_no_worse_and_no_more_expensive)
    intervals: dict[str, Any] = {
        "utility_loss_vs_always_on": mean_ci(utility_losses, t_critical),
        "work_saving_vs_always_on": mean_ci(work_savings, t_critical),
        "noisy_tv_excess_vs_rate_matched_shuffle": mean_ci(noise_excess, t_critical),
        "learned_expected_calibration_error": mean_ci(calibration, t_critical),
        "pareto_contribution": {
            name: mean_ci(values, t_critical) for name, values in control_advantages.items()
        },
    }
    checks = {
        "utility_noninferior": intervals["utility_loss_vs_always_on"]["upper"]
        <= float(criteria["max_utility_loss_vs_always_on"]),
        "lifecycle_work_saving": intervals["work_saving_vs_always_on"]["lower"]
        >= float(criteria["min_work_saving_vs_always_on"]),
        "positive_pareto_contribution_over_registered_controls": all(
            interval["lower"] > 0 for interval in intervals["pareto_contribution"].values()
        ),
        "no_excess_irreducible_noise_activation": intervals["noisy_tv_excess_vs_rate_matched_shuffle"][
            "upper"
        ]
        <= float(criteria["max_noisy_tv_excess_vs_shuffle"]),
        "unseen_family_calibration": intervals["learned_expected_calibration_error"]["upper"]
        <= float(criteria["max_expected_calibration_error"]),
        "queue_and_deadline_integrity": all(
            bool(row["arms"]["learned_event_former"]["queue_stable"])
            and int(row["arms"]["learned_event_former"]["storm_deadline_misses"])
            <= int(criteria["max_storm_deadline_misses"])
            for row in rows
        ),
        "header_does_not_explain_full_result": all(header_diagnostics),
        "no_label_or_oracle_access": all(bool(row["leakage_gate"]["passed"]) for row in rows),
        "required_direction_every_seed": all(all_seed_directions.values()),
    }
    favorable = all(checks.values())
    return {
        "status": "complete",
        "paired_seed_count": len(rows),
        "seed_ids": [int(row["seed"]) for row in rows],
        "split": rows[0]["split"],
        "paired_intervals_95": intervals,
        "all_seed_directions": all_seed_directions,
        "checks": checks,
        "verdict": "gate_a_candidate_pattern_favorable" if favorable else "strong_null_not_rejected",
        "strong_null_rejected": favorable,
        "scientific_promotion": False,
        "interpretation_limit": config["verdict"]["interpretation_limit"],
    }


IMPLEMENTATION_PATHS = (
    Path("configs/experiment/escs_x0_event_formation.json"),
    Path("src/mop/studies/escs_x0_event_formation.py"),
    Path("scripts/run_escs_x0_event_formation.py"),
    Path("tests/test_escs_x0_event_formation.py"),
    Path("docs/audits/escs_x0_event_formation.md"),
)


def build_implementation_authority(
    *,
    config_authority_sha256: str,
    mode: str,
    review_status: str,
    file_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    receipts = (
        [dict(receipt) for receipt in file_receipts]
        if file_receipts is not None
        else [_file_receipt(REPO_ROOT / path) for path in IMPLEMENTATION_PATHS]
    )
    core = {
        "schema": IMPLEMENTATION_AUTHORITY_SCHEMA,
        "study_id": "escs-x0-charged-event-formation-v1",
        "mode": str(mode),
        "config_authority_sha256": str(config_authority_sha256),
        "review_status": str(review_status),
        "files": receipts,
    }
    return {**core, "manifest_sha256": canonical_sha256(core)}


def write_implementation_authority(
    path: Path | str = DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
    *,
    mode: str = "official",
    review_status: str = OFFICIAL_IMPLEMENTATION_REVIEW_STATUS,
) -> dict[str, Any]:
    target = Path(path).resolve()
    _require_distinct_paths(
        {
            "implementation_authority_output": target,
            **{f"scoped:{relative}": REPO_ROOT / relative for relative in IMPLEMENTATION_PATHS},
        }
    )
    document = build_implementation_authority(
        config_authority_sha256=OFFICIAL_CONFIG_AUTHORITY_SHA256,
        mode=mode,
        review_status=review_status,
    )
    _atomic_json(target, document)
    return document


def load_implementation_authority(
    path: Path | str,
    config: Mapping[str, Any],
    *,
    expected_sha256: str | None,
    exploratory: bool,
) -> dict[str, Any]:
    source = Path(path).resolve()
    if exploratory:
        _require(
            source != DEFAULT_IMPLEMENTATION_AUTHORITY_PATH.resolve(),
            "exploratory X0 requires explicit manifest",
        )
    else:
        _require(
            source == DEFAULT_IMPLEMENTATION_AUTHORITY_PATH.resolve(),
            "official X0 requires canonical manifest",
        )
        _require(
            expected_sha256 is not None and len(expected_sha256) == 64, "independent manifest digest required"
        )
    raw = _read_regular_file(source, MAX_ARTIFACT_BYTES, "X0 implementation authority")
    document = json.loads(raw)
    _require(isinstance(document, dict), "X0 implementation authority must be a mapping")
    core = dict(document)
    digest = str(core.pop("manifest_sha256", ""))
    _require(digest == canonical_sha256(core), "X0 implementation authority self-hash mismatch")
    if expected_sha256 is not None:
        _require(digest == expected_sha256, "X0 implementation authority pin mismatch")
    _require(document["schema"] == IMPLEMENTATION_AUTHORITY_SCHEMA, "X0 implementation schema mismatch")
    _require(
        document["config_authority_sha256"] == canonical_sha256(config), "X0 config/implementation mismatch"
    )
    _require(
        document["files"] == [_file_receipt(REPO_ROOT / path) for path in IMPLEMENTATION_PATHS],
        "X0 implementation files drifted",
    )
    if exploratory:
        _require(document["mode"] == "exploratory", "exploratory manifest mode required")
    else:
        _require(document["mode"] == "official", "official manifest mode required")
        _require(document["review_status"] == OFFICIAL_IMPLEMENTATION_REVIEW_STATUS, "X0 review status drift")
    return document


def _checkpoint_core(
    *,
    config_sha256: str,
    implementation_sha256: str,
    gate_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "config_authority_sha256": config_sha256,
        "implementation_authority_sha256": implementation_sha256,
        "gate_rows": [dict(row) for row in gate_rows],
        "heldout_rows": [dict(row) for row in heldout_rows],
        "gate_rows_sha256": canonical_sha256(gate_rows),
        "heldout_rows_sha256": canonical_sha256(heldout_rows),
    }


def _write_checkpoint(
    path: Path,
    *,
    config_sha256: str,
    implementation_sha256: str,
    gate_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    core = _checkpoint_core(
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        gate_rows=gate_rows,
        heldout_rows=heldout_rows,
    )
    checkpoint = {**core, "checkpoint_sha256": canonical_sha256(core)}
    _require(len(canonical_bytes(checkpoint)) + 1 <= MAX_ARTIFACT_BYTES, "X0 checkpoint byte cap exceeded")
    _atomic_json(path, checkpoint)
    return checkpoint


def _load_checkpoint(
    path: Path,
    *,
    config_sha256: str,
    implementation_sha256: str,
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    raw = _read_regular_file(path, MAX_ARTIFACT_BYTES, "X0 checkpoint")
    checkpoint = json.loads(raw)
    core = dict(checkpoint)
    digest = core.pop("checkpoint_sha256", "")
    _require(digest == canonical_sha256(core), "X0 checkpoint self-hash mismatch")
    _require(checkpoint["schema"] == CHECKPOINT_SCHEMA, "X0 checkpoint schema mismatch")
    _require(checkpoint["config_authority_sha256"] == config_sha256, "X0 checkpoint config drift")
    _require(
        checkpoint["implementation_authority_sha256"] == implementation_sha256,
        "X0 checkpoint implementation drift",
    )
    gate_rows = list(checkpoint["gate_rows"])
    heldout_rows = list(checkpoint["heldout_rows"])
    _require(
        checkpoint["gate_rows_sha256"] == canonical_sha256(gate_rows), "X0 gate checkpoint hash mismatch"
    )
    _require(
        checkpoint["heldout_rows_sha256"] == canonical_sha256(heldout_rows),
        "X0 heldout checkpoint hash mismatch",
    )
    _require(
        [int(row["seed"]) for row in gate_rows] == list(seeds[: len(gate_rows)]), "X0 gate seed prefix drift"
    )
    _require(
        [int(row["seed"]) for row in heldout_rows] == list(seeds[: len(heldout_rows)]),
        "X0 heldout seed prefix drift",
    )
    for row in [*gate_rows, *heldout_rows]:
        core_row = dict(row)
        row_digest = core_row.pop("row_sha256", "")
        _require(row_digest == canonical_sha256(core_row), "X0 checkpoint seed-row hash mismatch")
    return gate_rows, heldout_rows


def run_from_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    implementation_authority_path: Path | str = DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
    *,
    implementation_authority_sha256: str | None = None,
    max_new_seeds: int | None = None,
    exploratory: bool = False,
) -> dict[str, Any]:
    _require(max_new_seeds is None or max_new_seeds >= 0, "max_new_seeds must be nonnegative")
    source = Path(config_path).resolve()
    output = Path(output_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    implementation_source = Path(implementation_authority_path).resolve()
    _require_distinct_paths(
        {
            "config": source,
            "output": output,
            "checkpoint": checkpoint,
            "implementation": implementation_source,
        }
    )
    if exploratory:
        _require(output != DEFAULT_OUTPUT_PATH.resolve(), "exploratory X0 cannot write official receipt")
        _require(
            checkpoint != DEFAULT_CHECKPOINT_PATH.resolve(), "exploratory X0 cannot write official checkpoint"
        )
    config = load_config(source, exploratory=exploratory)
    envelope, config_source_receipt = _load_envelope_snapshot(source)
    implementation = load_implementation_authority(
        implementation_source,
        config,
        expected_sha256=implementation_authority_sha256,
        exploratory=exploratory,
    )
    config_sha256 = canonical_sha256(config)
    implementation_sha256 = str(implementation["manifest_sha256"])
    seeds = [int(seed) for seed in config["seeds"]]
    gate_rows, heldout_rows = _load_checkpoint(
        checkpoint,
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        seeds=seeds,
    )
    remaining = max_new_seeds
    for seed in seeds[len(gate_rows) :]:
        if remaining is not None and remaining <= 0:
            break
        gate_rows.append(run_seed(config, seed=seed, split="gate"))
        _write_checkpoint(
            checkpoint,
            config_sha256=config_sha256,
            implementation_sha256=implementation_sha256,
            gate_rows=gate_rows,
            heldout_rows=heldout_rows,
        )
        if remaining is not None:
            remaining -= 1
    gate = difficulty_gate(gate_rows, config)
    if gate["passed"]:
        for seed in seeds[len(heldout_rows) :]:
            if remaining is not None and remaining <= 0:
                break
            heldout_rows.append(run_seed(config, seed=seed, split="heldout"))
            _write_checkpoint(
                checkpoint,
                config_sha256=config_sha256,
                implementation_sha256=implementation_sha256,
                gate_rows=gate_rows,
                heldout_rows=heldout_rows,
            )
            if remaining is not None:
                remaining -= 1
    checkpoint_document = _write_checkpoint(
        checkpoint,
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        gate_rows=gate_rows,
        heldout_rows=heldout_rows,
    )
    gate_complete = len(gate_rows) == len(seeds)
    heldout_complete = len(heldout_rows) == len(seeds)
    invalid_bed = gate_complete and not gate["passed"]
    complete = invalid_bed or heldout_complete
    aggregate = (
        {
            "status": "invalid_bed",
            "verdict": config["verdict"]["invalid_bed_label"],
            "scientific_promotion": False,
        }
        if invalid_bed
        else aggregate_rows(heldout_rows, config)
    )
    core = {
        "schema": RECEIPT_SCHEMA,
        "study_id": "escs-x0-charged-event-formation-v1",
        "claim_scope": CLAIM_SCOPE,
        "strong_null": config["strong_null"],
        "authority": envelope["authority"],
        "authority_sha256": config_sha256,
        "config_source": config_source_receipt,
        "implementation_authority": {
            "source": _file_receipt(implementation_source),
            "manifest_sha256": implementation_sha256,
            "mode": implementation["mode"],
            "review_status": implementation["review_status"],
        },
        "runtime_identity": _runtime_identity(),
        "gate_rows": gate_rows,
        "difficulty_gate": gate,
        "heldout_rows": heldout_rows,
        "aggregate": aggregate,
        "execution_status": "complete" if complete else "partial",
        "resumable": not complete,
        "all_ok": complete,
        "checkpoint": {
            "source": _file_receipt(checkpoint),
            "checkpoint_sha256": checkpoint_document["checkpoint_sha256"],
            "gate_rows_sha256": checkpoint_document["gate_rows_sha256"],
            "heldout_rows_sha256": checkpoint_document["heldout_rows_sha256"],
        },
        "full_lifecycle_work_components": list(WORK_COMPONENTS),
        "fresh_verifier_status": "pending-independent-command",
        "gate_a_integration": "permissive-candidate-or-control-only",
        "scientific_promotion": False,
        "interpretation_limit": config["verdict"]["interpretation_limit"],
        "exploratory": exploratory,
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    _require(
        len(canonical_bytes(receipt)) + 1 <= int(config["resources"]["max_receipt_bytes"]),
        "X0 receipt byte cap exceeded",
    )
    _atomic_json(output, receipt)
    return receipt


def _load_receipt(path: Path) -> dict[str, Any]:
    raw = _read_regular_file(path, MAX_ARTIFACT_BYTES, "X0 producer receipt")
    receipt = json.loads(raw)
    _require(isinstance(receipt, dict), "X0 receipt must be a mapping")
    core = dict(receipt)
    digest = core.pop("receipt_sha256", "")
    _require(digest == canonical_sha256(core), "X0 receipt self-hash mismatch")
    _require(receipt["schema"] == RECEIPT_SCHEMA, "X0 receipt schema mismatch")
    return receipt


def verify_receipt(
    receipt_path: Path | str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    implementation_authority_path: Path | str = DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
    *,
    implementation_authority_sha256: str | None = None,
    exploratory: bool = False,
) -> dict[str, Any]:
    receipt_source = Path(receipt_path).resolve()
    config_source = Path(config_path).resolve()
    implementation_source = Path(implementation_authority_path).resolve()
    _require_distinct_paths(
        {"receipt": receipt_source, "config": config_source, "implementation": implementation_source}
    )
    config = load_config(config_source, exploratory=exploratory)
    implementation = load_implementation_authority(
        implementation_source,
        config,
        expected_sha256=implementation_authority_sha256,
        exploratory=exploratory,
    )
    receipt = _load_receipt(receipt_source)
    _require(not receipt["resumable"], "fresh verifier refuses partial X0 receipt")
    _require(receipt["authority_sha256"] == canonical_sha256(config), "X0 receipt/config mismatch")
    _require(
        receipt["implementation_authority"]["manifest_sha256"] == implementation["manifest_sha256"],
        "X0 receipt/implementation mismatch",
    )
    regenerated_gate = [run_seed(config, seed=int(seed), split="gate") for seed in config["seeds"]]
    _require(regenerated_gate == receipt["gate_rows"], "X0 producer gate regeneration mismatch")
    gate = difficulty_gate(regenerated_gate, config)
    _require(gate == receipt["difficulty_gate"], "X0 difficulty gate regeneration mismatch")
    if gate["passed"]:
        regenerated_heldout = [run_seed(config, seed=int(seed), split="heldout") for seed in config["seeds"]]
        _require(regenerated_heldout == receipt["heldout_rows"], "X0 producer heldout regeneration mismatch")
        primary = aggregate_rows(regenerated_heldout, config)
        _require(primary == receipt["aggregate"], "X0 producer aggregate regeneration mismatch")
        fresh_rows = [
            run_seed(config, seed=int(seed), split="fresh_verifier")
            for seed in config["fresh_verifier_seeds"]
        ]
        fresh_difficulty = difficulty_gate(fresh_rows, config)
        fresh = aggregate_rows(fresh_rows, config)
        candidate_verified = (
            bool(fresh_difficulty["passed"])
            and bool(primary.get("strong_null_rejected"))
            and bool(fresh.get("strong_null_rejected"))
        )
        if not fresh_difficulty["passed"]:
            verdict = config["verdict"]["invalid_bed_label"]
        else:
            verdict = (
                "fresh_verified_gate_a_candidate" if candidate_verified else "strong_null_not_rejected"
            )
    else:
        regenerated_heldout = []
        primary = receipt["aggregate"]
        fresh_rows = []
        fresh_difficulty = {
            "status": "not_run_invalid_primary_bed",
            "passed": False,
            "failure_interpretation": "invalid_bed_not_mechanism_null",
        }
        fresh = {"status": "not_run_invalid_bed", "scientific_promotion": False}
        candidate_verified = False
        verdict = config["verdict"]["invalid_bed_label"]
    core = {
        "schema": VERIFICATION_SCHEMA,
        "study_id": receipt["study_id"],
        "producer_receipt": _file_receipt(receipt_source),
        "producer_receipt_sha256": receipt["receipt_sha256"],
        "implementation_authority_sha256": implementation["manifest_sha256"],
        "producer_regeneration_match": True,
        "regenerated_gate_seed_ids": [int(row["seed"]) for row in regenerated_gate],
        "regenerated_heldout_seed_ids": [int(row["seed"]) for row in regenerated_heldout],
        "fresh_seed_ids": [int(row["seed"]) for row in fresh_rows],
        "fresh_seed_rows": fresh_rows,
        "fresh_difficulty_gate": fresh_difficulty,
        "primary_aggregate": primary,
        "fresh_aggregate": fresh,
        "verdict": verdict,
        "gate_a_candidate_verified": candidate_verified,
        "scientific_promotion": False,
        "interpretation_limit": config["verdict"]["interpretation_limit"],
    }
    return {**core, "verification_sha256": canonical_sha256(core)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--implementation-authority", type=Path, default=DEFAULT_IMPLEMENTATION_AUTHORITY_PATH
    )
    parser.add_argument("--implementation-authority-sha256")
    parser.add_argument("--max-new-seeds", type=int)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--verification-out", type=Path, default=DEFAULT_VERIFICATION_OUTPUT_PATH)
    parser.add_argument("--exploratory", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.verify is not None:
        if arguments.exploratory:
            _require(
                arguments.verification_out.resolve() != DEFAULT_VERIFICATION_OUTPUT_PATH.resolve(),
                "exploratory verifier cannot write official verification path",
            )
        result = verify_receipt(
            arguments.verify,
            arguments.config,
            arguments.implementation_authority,
            implementation_authority_sha256=arguments.implementation_authority_sha256,
            exploratory=arguments.exploratory,
        )
        _require_distinct_paths(
            {
                "verification_output": arguments.verification_out,
                "receipt": arguments.verify,
                "config": arguments.config,
                "implementation": arguments.implementation_authority,
            }
        )
        _atomic_json(arguments.verification_out, result)
        print(
            json.dumps(
                {"verdict": result["verdict"], "scientific_promotion": False}, indent=2, sort_keys=True
            )
        )
        return 0
    result = run_from_config(
        arguments.config,
        arguments.output,
        arguments.checkpoint,
        arguments.implementation_authority,
        implementation_authority_sha256=arguments.implementation_authority_sha256,
        max_new_seeds=arguments.max_new_seeds,
        exploratory=arguments.exploratory,
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 2 if result["resumable"] else 0


__all__ = [
    "ARM_NAMES",
    "CLAIM_SCOPE",
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_IMPLEMENTATION_AUTHORITY_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_VERIFICATION_OUTPUT_PATH",
    "DelayedConsequenceEventPolicy",
    "EpisodeTrace",
    "EvaluatorTruth",
    "EventPolicy",
    "IMPLEMENTATION_PATHS",
    "OFFICIAL_CONFIG_AUTHORITY_SHA256",
    "OFFICIAL_IMPLEMENTATION_REVIEW_STATUS",
    "PacketCase",
    "PolicyDescriptor",
    "PublicConsequence",
    "TrainingObservation",
    "VisiblePacket",
    "aggregate_rows",
    "build_implementation_authority",
    "canonical_sha256",
    "difficulty_gate",
    "generate_episode",
    "leakage_gate",
    "load_config",
    "load_implementation_authority",
    "main",
    "run_from_config",
    "run_seed",
    "verify_receipt",
    "write_implementation_authority",
]
