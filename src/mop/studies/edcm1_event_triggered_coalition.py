"""EDCM-1 v3: falsifiable event-triggered coalition mechanics.

The official study is synthetic, CPU-only, weight-free, and mechanics-only.
Deterministic abstract work is the primary cost axis.  Hardware timing is
deliberately excluded from the receipt and reserved for a separate post-v3
benchmark so timing noise cannot affect resume identity or a scientific verdict.
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
from collections import Counter, defaultdict, deque
from collections.abc import Collection, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from mop.substrate.events import canonical_bytes, canonical_sha256

ENVELOPE_SCHEMA = "mop-edcm1-envelope/v3"
CONFIG_SCHEMA = "mop-edcm1-config/v3"
AUTHORITY_SCHEMA = "mop-edcm1-authority/v1"
CHECKPOINT_SCHEMA = "mop-edcm1-checkpoint/v3"
RECEIPT_SCHEMA = "mop-edcm1-receipt/v3"
PROPOSAL_SCHEMA = "mop-edcm1-proposal/v3"
VERIFICATION_SCHEMA = "mop-edcm1-verification/v3"
VERIFICATION_ARTIFACT_SCHEMA = "mop-edcm1-verification-artifact/v1"
IMPLEMENTATION_AUTHORITY_SCHEMA = "mop-edcm1-implementation-authority/v1"
CLAIM_SCOPE = "event-triggered-coalition-mechanics-only"
OFFICIAL_CONTRACT_ID = "edcm1-v3-2026-07-11"
OFFICIAL_IMPLEMENTATION_REVIEW_STATUS = "approved-for-official-execution"
OFFICIAL_VERIFIER_MODE = "full-deterministic-regeneration/v1"
DIAGNOSTIC_VERIFIER_MODE = "structural-diagnostics-only/v1"
OFFICIAL_AUTHORITY_SHA256 = "8b99bf150f8194f1c0485c536b6b240e611914ccf9ed2df0ad28f78b9a78cfff"

PROPOSER_ORDER = ("reactive_spatial", "episodic_retrieval", "short_horizon_planner")
VERIFIER_ID = "contradiction_verifier"
MAIN_ARMS = (
    "event_triggered",
    "always_on",
    "tuned_best_single",
    "periodic_round_matched",
    "shuffled_round_matched",
    "shuffled_coalition_matched",
    "homogeneous_matched",
    "equal_budget_recurrent",
)
DIRECT_INTERVENTIONS = (
    "clean",
    "no_message",
    "reactive_link_lesion",
    "episodic_link_lesion",
    "planner_link_lesion",
    "planner_channel_delay",
    "wrong_planner_message",
    "verifier_lesion",
)

ARM_SUMMARY_KEYS = frozenset(
    {
        "arm",
        "episode_count",
        "steps",
        "mean_utility",
        "mean_return",
        "success_rate",
        "total_abstract_work",
        "work_component_totals",
        "accounting_sensitivity",
        "abstract_work_per_step",
        "total_message_bytes",
        "message_bytes_per_step",
        "round_activation_counts",
        "activation_record_multiset_sha256_per_episode",
        "coalition_size_histogram",
        "noise_false_expensive_rate",
        "noise_activation_pair_differences",
        "noise_pair_opportunities",
        "change_expensive_rate",
        "hard_dispatch_violations",
        "sample_efficiency",
        "per_episode",
        "direct_effects",
    }
)
RECURRENT_ARM_EXTRA_KEYS = frozenset({"recurrent_hyperparameters", "recurrent_budget"})
GATE_ROW_KEYS = frozenset(
    {
        "schema",
        "seed",
        "tune",
        "gate",
        "oracle_headroom",
        "oracle_values",
        "unique_win_counts",
        "unique_win_rates",
        "niche_advantages",
        "best_gate_kind",
        "best_gate_success_rate",
        "tune_event_reference",
        "recurrent_tuning_budget",
        "recurrent_candidates",
        "verifier",
    }
)
HELDOUT_ROW_KEYS = frozenset(
    {
        "schema",
        "seed",
        "selected_best_single",
        "selected_recurrent",
        "arms",
        "clean_fixed_replay",
        "mechanics",
        "invariants",
        "pareto",
    }
)
CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "authority_sha256",
        "implementation_authority_sha256",
        "implementation_sha256",
        "runtime_identity",
        "gate_rows",
        "heldout_rows",
        "gate_seed_ids",
        "heldout_seed_ids",
        "gate_row_sha256",
        "heldout_row_sha256",
        "checkpoint_sha256",
    }
)
IMPLEMENTATION_AUTHORITY_KEYS = frozenset(
    {
        "schema",
        "study_id",
        "mode",
        "config_authority_sha256",
        "review_status",
        "files",
        "manifest_sha256",
    }
)
IMPLEMENTATION_AUTHORITY_RECEIPT_KEYS = frozenset({"source", "mode", "review_status", "manifest_sha256"})
FILE_RECEIPT_KEYS = frozenset({"path", "bytes", "sha256"})
EMPIRICAL_TIMING_KEYS = frozenset({"included", "reason"})
RESUME_KEYS = frozenset({"granularity", "checkpoint_path", "deterministic"})
CHECKPOINT_BINDING_KEYS = frozenset({"file", "checkpoint_sha256", "gate_row_sha256", "heldout_row_sha256"})
EMPIRICAL_TIMING_CLAIM = {
    "included": False,
    "reason": "reserved for a post-v3 non-verdict benchmark to preserve deterministic identity",
}
RECEIPT_KEYS = frozenset(
    {
        "schema",
        "study_id",
        "claim_scope",
        "strong_null",
        "authority",
        "authority_sha256",
        "config_source",
        "implementation",
        "implementation_authority",
        "implementation_authority_sha256",
        "implementation_sha256",
        "runtime_identity",
        "gate_rows",
        "gate",
        "heldout_rows",
        "aggregate",
        "execution_status",
        "all_ok",
        "problems",
        "resumable",
        "terminal_scientific_stop",
        "terminal_stop_reason",
        "completed_gate_seeds",
        "completed_heldout_seeds",
        "required_gate_seeds",
        "required_heldout_seeds",
        "abstract_work_contract",
        "empirical_timing",
        "resume",
        "checkpoint_binding",
        "prospective_artifact_guard",
        "exploratory",
        "scientific_promotion",
        "verifier_mode",
        "deterministic_core_sha256",
        "receipt_sha256",
    }
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/experiment/edcm1_event_triggered_coalition.yaml"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.checkpoint.json"
DEFAULT_VERIFICATION_OUTPUT_PATH = REPO_ROOT / "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.verification.json"
DEFAULT_IMPLEMENTATION_AUTHORITY_PATH = (
    REPO_ROOT / "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.implementation-authority.json"
)
MAX_IMPLEMENTATION_AUTHORITY_BYTES = 1_048_576
MAX_VERIFICATION_ARTIFACT_BYTES = 1_048_576
MAX_CONFIG_BYTES = 1_048_576
MAX_SCOPED_FILE_RECEIPT_BYTES = 67_108_864


def _stable_int(*parts: Any, modulus: int = 2**63 - 1) -> int:
    digest = hashlib.sha256(canonical_bytes(list(parts))).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_exact_keys(value: Any, allowed: Collection[str], label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    actual = set(value)
    expected = set(allowed)
    _require(
        actual == expected,
        f"{label} keys mismatch: missing={sorted(expected - actual)!r}, "
        f"unknown={sorted(actual - expected)!r}",
    )


def _require_distinct_paths(paths: Mapping[str, Path]) -> None:
    resolved = {label: path.resolve() for label, path in paths.items()}
    reverse: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for label, path in resolved.items():
        logical = unicodedata.normalize("NFC", str(path)).casefold()
        reverse[("logical", logical)].append(label)
        try:
            metadata = path.stat()
        except FileNotFoundError:
            continue
        _require(
            stat.S_ISREG(metadata.st_mode),
            f"artifact path {label!r} must be a regular file",
        )
        reverse[("inode", int(metadata.st_dev), int(metadata.st_ino))].append(label)
    labeled_paths = list(resolved.items())
    for index, (left_label, left_path) in enumerate(labeled_paths):
        for right_label, right_path in labeled_paths[index + 1 :]:
            try:
                aliases = os.path.samefile(left_path, right_path)
            except FileNotFoundError:
                aliases = False
            _require(
                not aliases,
                f"artifact path collision: {left_label!r} and {right_label!r} are the same file",
            )
    collisions = {repr(identity): labels for identity, labels in reverse.items() if len(labels) > 1}
    _require(not collisions, f"artifact path collision: {collisions}")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_regular_file(path: Path, max_bytes: int, label: str) -> bytes:
    _require(max_bytes > 0, f"{label} byte cap must be positive")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        size = int(before.st_size)
        _require(size <= max_bytes, f"{label} byte envelope exceeded before read")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        f"{label} changed during read",
    )
    _require(int(after.st_size) == size, f"{label} size changed during read")
    _require(len(raw) == size, f"{label} size changed during read")
    return raw


def _file_receipt_from_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    try:
        label = str(path.relative_to(REPO_ROOT))
    except ValueError:
        label = str(path)
    return {"path": label, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _read_json_artifact_snapshot(
    path: Path,
    max_bytes: int,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _read_regular_file(path, max_bytes, label)
    size = len(raw)
    value = json.loads(raw)
    _require(isinstance(value, dict), f"{label} must be a mapping")
    canonical = canonical_bytes(value) + b"\n"
    _require(len(canonical) == size, f"{label} canonical encoded size/on-disk size mismatch")
    _require(raw == canonical, f"{label} is not canonical JSON")
    return value, _file_receipt_from_bytes(path, raw)


def _read_json_artifact(path: Path, max_bytes: int, label: str) -> dict[str, Any]:
    value, _ = _read_json_artifact_snapshot(path, max_bytes, label)
    return value


def _file_receipt(
    path: Path,
    max_bytes: int = MAX_SCOPED_FILE_RECEIPT_BYTES,
) -> dict[str, Any]:
    payload = _read_regular_file(path, max_bytes, f"file receipt {path}")
    return _file_receipt_from_bytes(path, payload)


def _read_envelope_snapshot(
    path: Path | str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path).resolve()
    raw = _read_regular_file(source, MAX_CONFIG_BYTES, "configuration")
    envelope = json.loads(raw)
    _require(isinstance(envelope, dict), "configuration envelope must be a mapping")
    _require_exact_keys(envelope, ("schema", "authority", "payload"), "configuration envelope")
    _require(envelope.get("schema") == ENVELOPE_SCHEMA, "unexpected envelope schema")
    _require(isinstance(envelope.get("authority"), dict), "authority envelope missing")
    _require(isinstance(envelope.get("payload"), dict), "configuration payload missing")
    _require_exact_keys(
        envelope["authority"],
        ("schema", "mode", "contract_id", "payload_sha256"),
        "configuration authority",
    )
    return envelope, _file_receipt_from_bytes(source, raw)


def _read_envelope(path: Path | str) -> dict[str, Any]:
    envelope, _ = _read_envelope_snapshot(path)
    return envelope


def _load_config_snapshot(
    path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    exploratory: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = Path(path).resolve()
    envelope, source_receipt = _read_envelope_snapshot(source)
    authority = envelope["authority"]
    payload = envelope["payload"]
    payload_hash = canonical_sha256(payload)
    _require(authority.get("schema") == AUTHORITY_SCHEMA, "authority schema mismatch")
    _require(authority.get("payload_sha256") == payload_hash, "authority payload hash mismatch")
    if not exploratory:
        _require(source == DEFAULT_CONFIG_PATH.resolve(), "official execution requires the repository config")
        _require(authority.get("mode") == "official", "official authority mode required")
        _require(authority.get("contract_id") == OFFICIAL_CONTRACT_ID, "official contract id mismatch")
        _require(
            payload_hash == OFFICIAL_AUTHORITY_SHA256, "configuration is not the frozen official authority"
        )
    _validate_config(payload)
    return copy.deepcopy(payload), copy.deepcopy(envelope), source_receipt


def load_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    exploratory: bool = False,
) -> dict[str, Any]:

    config, _, _ = _load_config_snapshot(path, exploratory=exploratory)
    return config


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema") == CONFIG_SCHEMA, "unexpected config schema")
    _require(config.get("claim_scope") == CLAIM_SCOPE, "claim scope drift")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 5:
        raise ValueError("exactly five seeds required")
    _require(len(set(seeds)) == 5 and all(isinstance(seed, int) for seed in seeds), "invalid seeds")
    _require(
        0 < int(config["splits"]["intervention_episodes"]) <= int(config["splits"]["heldout_episodes"]),
        "intervention episode cap outside heldout split",
    )
    _require(tuple(config["specialists"]["proposer_order"]) == PROPOSER_ORDER, "proposer order drift")
    _require(config["specialists"]["verifier_id"] == VERIFIER_ID, "verifier id drift")
    _require(tuple(config["controls"]["main"]) == MAIN_ARMS, "main controls drift")
    _require(tuple(config["controls"]["direct_interventions"]) == DIRECT_INTERVENTIONS, "intervention drift")
    _require(config["messages"]["proposal_schema"] == PROPOSAL_SCHEMA, "proposal schema drift")
    _require(config["messages"]["verification_schema"] == VERIFICATION_SCHEMA, "verification schema drift")
    _require(config["resume"]["schema"] == CHECKPOINT_SCHEMA, "checkpoint schema drift")
    world = config["world"]
    horizon = int(world["horizon"])
    points = tuple(int(value) for value in world["change_points"])
    radius = int(world["change_window_radius"])
    noise_ticks = tuple(int(value) for value in world["noise_ticks"])
    _require(points == tuple(sorted(set(points))), "change points must be sorted and unique")
    _require(all(0 < point < horizon for point in points), "change point outside horizon")
    change_window = {
        tick for point in points for tick in range(max(0, point - radius), min(horizon, point + radius + 1))
    }
    _require(not change_window.intersection(noise_ticks), "noise ticks overlap change windows")
    _require(all(0 < tick < horizon for tick in noise_ticks), "noise tick outside horizon")
    hidden = set(world["evaluator_hidden_fields"])
    visible = set(world["cognition_visible_fields"])
    _require(not hidden.intersection(visible), "hidden and visible fields overlap")
    gate = config["complementarity_gate"]
    _require(float(gate["min_unique_win_rate"]) >= 0.10, "weak unique-win gate")
    _require(int(gate["min_unique_wins"]) >= 20, "weak unique-win count")
    _require(float(config["evaluation"]["t_critical_95"]) == 2.776, "statistical critical value drift")
    criteria = config["criteria"]
    _require(float(criteria["max_utility_loss_vs_always_on"]) == 0.01, "utility SESOI drift")
    _require(float(criteria["min_work_saving_vs_always_on"]) == 0.25, "work SESOI drift")
    _require(bool(criteria["require_all_seed_directions"]), "all-seed direction requirement disabled")
    _require(int(config["messages"]["max_extra_rounds"]) == 1, "exactly one optional extra round required")
    resources = config["resources"]
    _require(
        resources["cpu_only"] and resources["worker_count"] == 1,
        "official execution is deterministic CPU-only",
    )
    _require(not resources["allow_model_weights"], "model weights forbidden")
    _require(not resources["allow_downloads"], "downloads forbidden")
    _require(not resources["allow_external_data"], "external data forbidden")
    _require(int(resources["max_checkpoint_bytes"]) > 0, "checkpoint byte envelope missing")
    _require(int(resources["prospective_episode_record_bytes"]) > 0, "prospective record size missing")
    _require(
        tuple(float(value) for value in config["abstract_work"]["sensitivity_factors"]) == (0.5, 2.0),
        "accounting sensitivity factors drift",
    )
    _require(
        config["verdict"]["scientific_promotion"] == "blocked", "scientific promotion must remain blocked"
    )


@dataclass
class AbstractWork:

    scalar_ops: int = 0
    comparisons: int = 0
    nonlinearities: int = 0
    table_reads: int = 0
    table_writes: int = 0
    bytes_hashed: int = 0
    bytes_serialized: int = 0

    def add(self, other: AbstractWork) -> AbstractWork:
        for item in dataclasses.fields(self):
            name = item.name
            setattr(self, name, int(getattr(self, name)) + int(getattr(other, name)))
        return self

    def copy(self) -> AbstractWork:
        return AbstractWork(**dataclasses.asdict(self))

    def total(self, weights: Mapping[str, int]) -> int:
        return sum(int(getattr(self, name)) * int(weights[name]) for name in dataclasses.asdict(self))


def accounting_sensitivity(
    components: Mapping[str, int],
    weights: Mapping[str, int],
    factors: Sequence[Any],
) -> dict[str, Any]:

    numeric_factors = tuple(float(value) for value in factors)
    scenarios: dict[str, dict[str, float]] = {}
    for name in sorted(components):
        values: dict[str, float] = {}
        for factor in numeric_factors:
            perturbed = {
                key: float(weight) * (factor if key == name else 1.0) for key, weight in weights.items()
            }
            value = sum(int(components[key]) * perturbed[key] for key in components)
            values[f"factor:{factor:g}"] = value
        scenarios[name] = values
    flattened = [value for values in scenarios.values() for value in values.values()]
    nominal = sum(int(components[name]) * int(weights[name]) for name in components)
    return {
        "nominal": nominal,
        "scenario_min": min(flattened, default=nominal),
        "scenario_max": max(flattened, default=nominal),
        "one_at_a_time": scenarios,
    }


@dataclass(frozen=True)
class VisibleObservation:
    world_id: str
    event_id: str
    tick: int
    local_blocked: tuple[int, int, int, int]
    relative_goal: tuple[int, int]
    previous_action: int
    previous_reward: float
    novelty_channels: tuple[int, ...]

    def structural_vector(self) -> tuple[int, ...]:
        return (
            self.local_blocked
            + self.relative_goal
            + (
                self.previous_action,
                int(round(self.previous_reward * 100)),
            )
        )

    def full_vector(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.structural_vector() + self.novelty_channels)

    def state_key(self) -> tuple[int, ...]:
        dr = max(-4, min(4, self.relative_goal[0]))
        dc = max(-4, min(4, self.relative_goal[1]))
        return (dr, dc) + self.local_blocked


@dataclass(frozen=True)
class PublicFeedback:
    source_event_id: str
    tick: int
    action: int
    reward: float
    blocked: bool
    reached_goal: bool


@dataclass(frozen=True)
class VisibleTransition:
    before: VisibleObservation
    action: int
    feedback: PublicFeedback
    after: VisibleObservation | None
    terminal: bool


@dataclass(frozen=True)
class EvaluatorTransition:
    visible: VisibleTransition
    hidden_change: bool
    action_rotation: int
    physical_action: int
    niche_label: str | None
    noise_label: bool


class PartialChangePointWorld:
    DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(self, config: Mapping[str, Any], seed: int, split: str, episode: int):
        world = config["world"]
        split_offset = int(config["splits"]["split_offsets"][split])
        self.side = int(world["side"])
        self.horizon = int(world["horizon"])
        self.change_points = tuple(int(value) for value in world["change_points"])
        self.noise_ticks = frozenset(int(value) for value in world["noise_ticks"])
        self.novelty_channel_count = int(world["novelty_channels"])
        self.goal_reward = float(world["goal_reward"])
        self.step_cost = float(world["step_cost"])
        self.blocked_cost = float(world["blocked_cost"])
        self.world_seed = _stable_int(seed, split_offset, episode)
        self.stratum = ("spatial", "recurring", "novel")[episode % 3]
        self.rotations = tuple(int(value) for value in world["regime_patterns"][self.stratum])
        _require(len(self.rotations) == len(self.change_points) + 1, "invalid regime pattern")
        self.start, self.goal, self.walls = self._generate_layout(world)
        self.world_id = canonical_sha256(
            {"world_seed": self.world_seed, "split": split, "episode": episode, "stratum": self.stratum}
        )[:24]
        self.position = self.start
        self.tick = 0
        self.previous_action = 0
        self.previous_reward = 0.0
        self.goal_ever_reached = False
        self.terminal = False

    def _generate_layout(
        self,
        world: Mapping[str, Any],
    ) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
        for attempt in range(10_000):
            rng = random.Random(_stable_int(self.world_seed, "layout", attempt))
            start = (rng.randrange(self.side), rng.randrange(self.side))
            goal = (rng.randrange(self.side), rng.randrange(self.side))
            if goal == start:
                continue
            walls: set[tuple[int, int]] = set()
            while len(walls) < int(world["wall_count"]):
                cell = (rng.randrange(self.side), rng.randrange(self.side))
                if cell not in (start, goal):
                    walls.add(cell)
            distance = self._shortest_path(start, goal, walls)
            if distance is not None and distance <= int(world["max_shortest_path"]):
                return start, goal, walls
        raise RuntimeError("failed to generate a connected preregistered world")

    def _shortest_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        walls: set[tuple[int, int]],
    ) -> int | None:
        frontier = deque([(start, 0)])
        visited = {start}
        while frontier:
            position, distance = frontier.popleft()
            if position == goal:
                return distance
            for dr, dc in self.DELTAS:
                candidate = (position[0] + dr, position[1] + dc)
                if (
                    0 <= candidate[0] < self.side
                    and 0 <= candidate[1] < self.side
                    and candidate not in walls
                    and candidate not in visited
                ):
                    visited.add(candidate)
                    frontier.append((candidate, distance + 1))
        return None

    def _rotation(self, tick: int) -> int:
        return self.rotations[sum(tick >= point for point in self.change_points)]

    def _blocked(self, position: tuple[int, int], physical_action: int) -> bool:
        dr, dc = self.DELTAS[physical_action]
        candidate = (position[0] + dr, position[1] + dc)
        return (
            candidate[0] < 0
            or candidate[1] < 0
            or candidate[0] >= self.side
            or candidate[1] >= self.side
            or candidate in self.walls
        )

    def _niche(self, tick: int) -> str | None:
        if self.stratum == "spatial" and tick <= 4:
            return "reactive_spatial"
        if self.stratum == "novel" and self.change_points[0] <= tick < self.change_points[1]:
            return "short_horizon_planner"
        if self.stratum == "recurring" and tick >= self.change_points[1]:
            return "episodic_retrieval"
        return None

    def observe(self) -> VisibleObservation:
        rotation = self._rotation(self.tick)
        local = tuple(int(self._blocked(self.position, (action + rotation) % 4)) for action in range(4))
        relative = (self.goal[0] - self.position[0], self.goal[1] - self.position[1])
        novelty = [0] * self.novelty_channel_count
        if self.tick in self.noise_ticks:
            forced = _stable_int(
                self.world_seed, "forced-noise", self.tick, modulus=self.novelty_channel_count
            )
            novelty[forced] = 1
            for channel in range(self.novelty_channel_count):
                novelty[channel] |= _stable_int(self.world_seed, "noise", self.tick, channel, modulus=2)
        event_id = canonical_sha256({"world_id": self.world_id, "tick": self.tick})
        return VisibleObservation(
            world_id=self.world_id,
            event_id=event_id,
            tick=self.tick,
            local_blocked=local,  # type: ignore[arg-type]
            relative_goal=relative,
            previous_action=self.previous_action,
            previous_reward=self.previous_reward,
            novelty_channels=tuple(novelty),
        )

    def step(self, action: int) -> EvaluatorTransition:
        _require(not self.terminal, "cannot step a terminal world")
        _require(0 <= action < 4, "action outside [0, 3]")
        before = self.observe()
        rotation = self._rotation(self.tick)
        physical_action = (action + rotation) % 4
        blocked = self._blocked(self.position, physical_action)
        if not blocked:
            dr, dc = self.DELTAS[physical_action]
            self.position = (self.position[0] + dr, self.position[1] + dc)
        reached = self.position == self.goal and not self.goal_ever_reached
        if reached:
            self.goal_ever_reached = True
        reward = self.goal_reward if reached else self.step_cost + (self.blocked_cost if blocked else 0.0)
        feedback = PublicFeedback(before.event_id, self.tick, action, reward, blocked, reached)
        hidden_change = self.tick in self.change_points
        niche = self._niche(self.tick)
        noise = self.tick in self.noise_ticks
        self.previous_action = action
        self.previous_reward = reward
        self.tick += 1
        self.terminal = self.tick >= self.horizon
        after = self.observe()
        visible = VisibleTransition(before, action, feedback, after, self.terminal)
        return EvaluatorTransition(visible, hidden_change, rotation, physical_action, niche, noise)

    def state_payload(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "world_seed": self.world_seed,
            "stratum": self.stratum,
            "start": self.start,
            "goal": self.goal,
            "walls": sorted(self.walls),
            "rotations": self.rotations,
            "position": self.position,
            "tick": self.tick,
            "previous_action": self.previous_action,
            "previous_reward": self.previous_reward,
            "goal_ever_reached": self.goal_ever_reached,
            "terminal": self.terminal,
        }

    def clone(self) -> PartialChangePointWorld:
        return copy.deepcopy(self)


def evaluator_step_value(transition: EvaluatorTransition, progress_weight: float) -> float:
    before = transition.visible.before.relative_goal
    after = transition.visible.after.relative_goal if transition.visible.after is not None else before
    progress = (abs(before[0]) + abs(before[1])) - (abs(after[0]) + abs(after[1]))
    return transition.visible.feedback.reward + progress_weight * progress


@dataclass(frozen=True)
class Provenance:
    producer_id: str
    producer_kind: str
    world_id: str
    source_event_id: str
    state_digest: str


@dataclass(frozen=True)
class ProposalMessage:
    schema: str
    message_id: str
    referent_id: str
    source_event_id: str
    created_tick: int
    max_age: int
    specialist_id: str
    specialist_kind: str
    proposed_action: int
    confidence: float
    expected_progress: float
    evidence: tuple[str, ...]
    producer_work_units: int
    provenance: Provenance

    @classmethod
    def create(
        cls,
        *,
        observation: VisibleObservation,
        specialist_id: str,
        specialist_kind: str,
        action: int,
        confidence: float,
        expected_progress: float,
        evidence: Sequence[str],
        state_payload: Any,
        weights: Mapping[str, int],
        max_age: int,
        work: AbstractWork,
    ) -> ProposalMessage:
        state_bytes = canonical_bytes(state_payload)
        state_digest = hashlib.sha256(state_bytes).hexdigest()
        work.bytes_serialized += len(state_bytes)
        work.bytes_hashed += len(state_bytes)
        producer_work_units = work.total(weights)
        provenance = Provenance(
            producer_id=specialist_id,
            producer_kind=specialist_kind,
            world_id=observation.world_id,
            source_event_id=observation.event_id,
            state_digest=state_digest,
        )
        body = {
            "schema": PROPOSAL_SCHEMA,
            "referent_id": f"goal@{observation.world_id}",
            "source_event_id": observation.event_id,
            "created_tick": observation.tick,
            "max_age": int(max_age),
            "specialist_id": specialist_id,
            "specialist_kind": specialist_kind,
            "proposed_action": int(action),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "expected_progress": float(expected_progress),
            "evidence": list(evidence),
            "producer_work_units": int(producer_work_units),
            "provenance": dataclasses.asdict(provenance),
        }
        body_bytes = canonical_bytes(body)
        work.bytes_serialized += len(body_bytes)
        work.bytes_hashed += len(body_bytes)
        message_id = hashlib.sha256(body_bytes).hexdigest()
        body["evidence"] = tuple(evidence)
        body["provenance"] = provenance
        return cls(message_id=message_id, **body)  # type: ignore[arg-type]

    def payload(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["evidence"] = list(self.evidence)
        return value

    @property
    def encoded_bytes(self) -> int:
        return len(canonical_bytes(self.payload()))

    def integrity_valid(self) -> bool:
        body = self.payload()
        message_id = body.pop("message_id")
        return self.schema == PROPOSAL_SCHEMA and message_id == canonical_sha256(body)

    def referent_valid(self, observation: VisibleObservation) -> bool:
        return (
            self.referent_id == f"goal@{observation.world_id}"
            and self.provenance.world_id == observation.world_id
        )

    def age(self, observation: VisibleObservation) -> int:
        return observation.tick - self.created_tick

    def usable(self, observation: VisibleObservation, allowed_age: int) -> bool:
        return (
            self.integrity_valid()
            and self.referent_valid(observation)
            and 0 <= self.age(observation) <= min(self.max_age, allowed_age)
        )


@dataclass(frozen=True)
class VerificationMessage:
    schema: str
    message_id: str
    source_event_id: str
    created_tick: int
    endorsed_message_id: str | None
    contradicted_message_ids: tuple[str, ...]
    confidence: float
    reason_codes: tuple[str, ...]
    abstained: bool
    state_digest: str

    @classmethod
    def create(
        cls,
        *,
        observation: VisibleObservation,
        endorsed_message_id: str | None,
        contradicted_message_ids: Sequence[str],
        confidence: float,
        reason_codes: Sequence[str],
        abstained: bool,
        state_payload: Any,
        work: AbstractWork,
    ) -> VerificationMessage:
        state_bytes = canonical_bytes(state_payload)
        state_digest = hashlib.sha256(state_bytes).hexdigest()
        work.bytes_serialized += len(state_bytes)
        work.bytes_hashed += len(state_bytes)
        body = {
            "schema": VERIFICATION_SCHEMA,
            "source_event_id": observation.event_id,
            "created_tick": observation.tick,
            "endorsed_message_id": endorsed_message_id,
            "contradicted_message_ids": list(contradicted_message_ids),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "reason_codes": list(reason_codes),
            "abstained": bool(abstained),
            "state_digest": state_digest,
        }
        body_bytes = canonical_bytes(body)
        work.bytes_serialized += len(body_bytes)
        work.bytes_hashed += len(body_bytes)
        message_id = hashlib.sha256(body_bytes).hexdigest()
        body["contradicted_message_ids"] = tuple(contradicted_message_ids)
        body["reason_codes"] = tuple(reason_codes)
        return cls(message_id=message_id, **body)  # type: ignore[arg-type]

    @property
    def encoded_bytes(self) -> int:
        return len(canonical_bytes(dataclasses.asdict(self)))


@dataclass
class SpecialistTelemetry:
    propose_calls: int = 0
    update_calls: int = 0


class Proposer:
    kind = "abstract"

    def __init__(self, specialist_id: str):
        self.specialist_id = specialist_id
        self.telemetry = SpecialistTelemetry()

    def state_payload(self) -> Any:
        raise NotImplementedError

    def propose(
        self,
        observation: VisibleObservation,
        weights: Mapping[str, int],
        max_age: int,
    ) -> tuple[ProposalMessage, AbstractWork]:
        raise NotImplementedError

    def update(self, transition: VisibleTransition) -> AbstractWork:
        raise NotImplementedError


def _goal_action(observation: VisibleObservation) -> int:
    dr, dc = observation.relative_goal
    candidates: list[int] = []
    if abs(dr) >= abs(dc) and dr:
        candidates.append(2 if dr > 0 else 0)
    if dc:
        candidates.append(1 if dc > 0 else 3)
    if dr:
        candidates.append(2 if dr > 0 else 0)
    candidates.extend((0, 1, 2, 3))
    for action in candidates:
        if not observation.local_blocked[action]:
            return action
    return candidates[0]


def _safe_fallback(observation: VisibleObservation) -> int:
    if not observation.local_blocked[observation.previous_action]:
        return observation.previous_action
    return next((action for action in range(4) if not observation.local_blocked[action]), 0)


class ReactiveSpatialProposer(Proposer):
    kind = "reactive_spatial"

    def __init__(self, specialist_id: str = kind):
        super().__init__(specialist_id)
        self.last_unblocked_action = 0

    def state_payload(self) -> Any:
        return {"last_unblocked_action": self.last_unblocked_action}

    def propose(
        self, observation: VisibleObservation, weights: Mapping[str, int], max_age: int
    ) -> tuple[ProposalMessage, AbstractWork]:
        self.telemetry.propose_calls += 1
        work = AbstractWork(scalar_ops=10, comparisons=9, table_reads=8)
        action = _goal_action(observation)
        if observation.previous_reward > 0 and not observation.local_blocked[self.last_unblocked_action]:
            action = self.last_unblocked_action
            work.comparisons += 2
        message = ProposalMessage.create(
            observation=observation,
            specialist_id=self.specialist_id,
            specialist_kind=self.kind,
            action=action,
            confidence=0.52,
            expected_progress=0.3,
            evidence=("visible-local-geometry", "relative-goal"),
            state_payload=self.state_payload(),
            weights=weights,
            max_age=max_age,
            work=work,
        )
        return message, work

    def update(self, transition: VisibleTransition) -> AbstractWork:
        self.telemetry.update_calls += 1
        work = AbstractWork(comparisons=1, table_writes=1)
        if not transition.feedback.blocked:
            self.last_unblocked_action = transition.action
        return work


@dataclass(frozen=True)
class EpisodeRecord:
    world_id: str
    source_event_id: str
    before_key: tuple[int, ...]
    action: int
    reward: float
    progress: int
    blocked: bool
    tick: int


class EpisodicRetrievalProposer(Proposer):
    kind = "episodic_retrieval"

    def __init__(self, specialist_id: str = kind, capacity: int = 96, freshness_horizon: int = 12):
        super().__init__(specialist_id)
        self.capacity = capacity
        self.freshness_horizon = freshness_horizon
        self.records: deque[EpisodeRecord] = deque(maxlen=capacity)

    def state_payload(self) -> Any:
        return {"capacity": self.capacity, "records": [dataclasses.asdict(record) for record in self.records]}

    def propose(
        self, observation: VisibleObservation, weights: Mapping[str, int], max_age: int
    ) -> tuple[ProposalMessage, AbstractWork]:
        self.telemetry.propose_calls += 1
        work = AbstractWork(table_reads=1)
        query = observation.state_key()
        best: EpisodeRecord | None = None
        best_score = -(10**9)
        for record in self.records:
            work.table_reads += 1
            work.comparisons += 1
            if record.world_id != observation.world_id:
                continue
            distance = sum(int(left != right) for left, right in zip(query, record.before_key, strict=True))
            age = max(0, observation.tick - record.tick)
            score = 6 * record.progress + int(round(20 * record.reward)) - 3 * distance - min(age, 20)
            work.comparisons += len(query) + 4
            work.scalar_ops += len(query) + 6
            if score > best_score:
                best, best_score = record, score
        evidence: tuple[str, ...]
        if best is None:
            action = _goal_action(observation)
            confidence = 0.16
            progress = 0.0
            evidence = ("episodic-empty",)
        else:
            age = max(0, observation.tick - best.tick)
            freshness = max(0.0, 1.0 - age / max(1, self.freshness_horizon))
            action = best.action
            confidence = 0.25 + 0.6 * freshness * int(not best.blocked)
            progress = float(best.progress)
            evidence = ("episodic-nearest-transition", best.source_event_id, f"age:{age}")
            work.scalar_ops += 5
        message = ProposalMessage.create(
            observation=observation,
            specialist_id=self.specialist_id,
            specialist_kind=self.kind,
            action=action,
            confidence=confidence,
            expected_progress=progress,
            evidence=evidence,
            state_payload=self.state_payload(),
            weights=weights,
            max_age=max_age,
            work=work,
        )
        return message, work

    def update(self, transition: VisibleTransition) -> AbstractWork:
        self.telemetry.update_calls += 1
        after_goal = (
            transition.after.relative_goal
            if transition.after is not None
            else transition.before.relative_goal
        )
        before_goal = transition.before.relative_goal
        progress = (abs(before_goal[0]) + abs(before_goal[1])) - (abs(after_goal[0]) + abs(after_goal[1]))
        self.records.append(
            EpisodeRecord(
                world_id=transition.before.world_id,
                source_event_id=transition.before.event_id,
                before_key=transition.before.state_key(),
                action=transition.action,
                reward=transition.feedback.reward,
                progress=progress,
                blocked=transition.feedback.blocked,
                tick=transition.before.tick,
            )
        )
        return AbstractWork(scalar_ops=6, table_reads=2, table_writes=1)


@dataclass(frozen=True)
class ModelTransition:
    before_key: tuple[int, ...]
    action: int
    next_key: tuple[int, ...]
    reward: float


class ShortHorizonPlannerProposer(Proposer):

    kind = "short_horizon_planner"

    def __init__(
        self,
        specialist_id: str = kind,
        history_capacity: int = 96,
        horizon: int = 3,
        discount: float = 0.8,
    ):
        super().__init__(specialist_id)
        self.history: deque[ModelTransition] = deque(maxlen=history_capacity)
        self.horizon = horizon
        self.discount = discount

    def state_payload(self) -> Any:
        return {
            "horizon": self.horizon,
            "discount": self.discount,
            "history": [dataclasses.asdict(item) for item in self.history],
        }

    @staticmethod
    def _prior_successor(state: tuple[int, ...], action: int) -> tuple[int, ...]:
        dr, dc = state[:2]
        blocked = state[2:]
        if blocked[action]:
            return state
        delta = PartialChangePointWorld.DELTAS[action]
        return (max(-4, min(4, dr - delta[0])), max(-4, min(4, dc - delta[1]))) + blocked

    def _build_model(
        self,
        work: AbstractWork,
    ) -> dict[tuple[tuple[int, ...], int], list[tuple[tuple[int, ...], float]]]:
        model: dict[tuple[tuple[int, ...], int], list[tuple[tuple[int, ...], float]]] = defaultdict(list)
        for transition in self.history:
            model[(transition.before_key, transition.action)].append((transition.next_key, transition.reward))
            work.table_reads += 1
            work.table_writes += 1
        return model

    def propose(
        self, observation: VisibleObservation, weights: Mapping[str, int], max_age: int
    ) -> tuple[ProposalMessage, AbstractWork]:
        self.telemetry.propose_calls += 1
        work = AbstractWork(table_reads=1)
        model = self._build_model(work)
        memo: dict[tuple[tuple[int, ...], int], float] = {}

        def best_value(state: tuple[int, ...], depth: int) -> float:
            cache_key = (state, depth)
            work.table_reads += 1
            if cache_key in memo:
                return memo[cache_key]
            if depth <= 0:
                return 0.0
            values = [action_value(state, action, depth) for action in range(4)]
            value = max(values)
            memo[cache_key] = value
            work.comparisons += 4
            work.table_writes += 1
            return value

        def action_value(state: tuple[int, ...], action: int, depth: int) -> float:
            outcomes = model.get((state, action))
            work.table_reads += 1
            if not outcomes:
                successor = self._prior_successor(state, action)
                immediate = -0.05 if state[2 + action] else -0.01
                work.table_reads += 1
                work.scalar_ops += 5
                return immediate + self.discount * best_value(successor, depth - 1)
            total = 0.0
            for successor, reward in outcomes:
                total += reward + self.discount * best_value(successor, depth - 1)
                work.scalar_ops += 3
                work.table_reads += 2
            work.scalar_ops += 1
            return total / len(outcomes)

        state = observation.state_key()
        values = [action_value(state, action, self.horizon) for action in range(4)]
        action = max(range(4), key=lambda candidate: (values[candidate], -candidate))
        ordered = sorted(values, reverse=True)
        gap = ordered[0] - ordered[1]
        evidence_count = len(model.get((state, action), ()))
        confidence = 0.25 + min(0.6, 0.08 * evidence_count + max(0.0, gap) / 2)
        work.comparisons += 10
        work.scalar_ops += 5
        message = ProposalMessage.create(
            observation=observation,
            specialist_id=self.specialist_id,
            specialist_kind=self.kind,
            action=action,
            confidence=confidence,
            expected_progress=values[action],
            evidence=(
                "learned-visible-transition-model",
                f"rollout-horizon:{self.horizon}",
                f"samples:{evidence_count}",
            ),
            state_payload=self.state_payload(),
            weights=weights,
            max_age=max_age,
            work=work,
        )
        return message, work

    def update(self, transition: VisibleTransition) -> AbstractWork:
        self.telemetry.update_calls += 1
        next_key = (
            transition.after.state_key() if transition.after is not None else transition.before.state_key()
        )
        self.history.append(
            ModelTransition(
                before_key=transition.before.state_key(),
                action=transition.action,
                next_key=next_key,
                reward=transition.feedback.reward,
            )
        )
        return AbstractWork(table_reads=2, table_writes=1)


class ContradictionVerifier:

    def __init__(self, momentum: float = 0.8):
        self.momentum = momentum
        self.reliability = {kind: 0.5 for kind in PROPOSER_ORDER}
        self.verify_calls = 0
        self.update_calls = 0

    def state_payload(self) -> Any:
        return {"momentum": self.momentum, "reliability": dict(sorted(self.reliability.items()))}

    def verify(
        self,
        observation: VisibleObservation,
        proposals: Sequence[ProposalMessage],
    ) -> tuple[VerificationMessage, AbstractWork]:
        self.verify_calls += 1
        work = AbstractWork(table_reads=2 * len(proposals), comparisons=2 * len(proposals))
        valid = []
        for proposal in proposals:
            _charge_proposal_validation(proposal, work)
            if proposal.usable(observation, allowed_age=1):
                valid.append(proposal)
        actions = {proposal.proposed_action for proposal in valid}
        if len(valid) < 2 or len(actions) < 2:
            message = VerificationMessage.create(
                observation=observation,
                endorsed_message_id=None,
                contradicted_message_ids=(),
                confidence=0.0,
                reason_codes=("insufficient-current-disagreement",),
                abstained=True,
                state_payload=self.state_payload(),
                work=work,
            )
            return message, work
        ranked = sorted(
            valid,
            key=lambda proposal: (
                proposal.confidence * self.reliability[proposal.specialist_kind],
                proposal.expected_progress,
                proposal.specialist_id,
            ),
            reverse=True,
        )
        endorsed = ranked[0]
        contradicted = tuple(
            proposal.message_id for proposal in valid if proposal.proposed_action != endorsed.proposed_action
        )
        work.scalar_ops += 2 * len(valid)
        work.comparisons += len(valid) * max(1, int(math.ceil(math.log2(len(valid)))))
        message = VerificationMessage.create(
            observation=observation,
            endorsed_message_id=endorsed.message_id,
            contradicted_message_ids=contradicted,
            confidence=min(0.95, 0.4 + 0.1 * len(contradicted)),
            reason_codes=("action-disagreement", "learned-proposer-reliability"),
            abstained=False,
            state_payload=self.state_payload(),
            work=work,
        )
        return message, work

    def update(self, transition: VisibleTransition, proposals: Sequence[ProposalMessage]) -> AbstractWork:
        self.update_calls += 1
        work = AbstractWork(table_reads=1, table_writes=0)
        before = transition.before.relative_goal
        after = transition.after.relative_goal if transition.after is not None else before
        improvement = (abs(before[0]) + abs(before[1])) - (abs(after[0]) + abs(after[1]))
        good = float(improvement > 0 or transition.feedback.reached_goal)
        for proposal in proposals:
            work.comparisons += 1
            if proposal.proposed_action != transition.action:
                continue
            prior = self.reliability[proposal.specialist_kind]
            self.reliability[proposal.specialist_kind] = self.momentum * prior + (1 - self.momentum) * good
            work.scalar_ops += 4
            work.table_reads += 1
            work.table_writes += 1
        return work


def make_proposer(config: Mapping[str, Any], kind: str, specialist_id: str | None = None) -> Proposer:
    settings = config["specialists"]
    identifier = specialist_id or kind
    if kind == "reactive_spatial":
        return ReactiveSpatialProposer(identifier)
    if kind == "episodic_retrieval":
        return EpisodicRetrievalProposer(
            identifier,
            int(settings["episodic_capacity"]),
            int(settings["episodic_freshness_horizon"]),
        )
    if kind == "short_horizon_planner":
        return ShortHorizonPlannerProposer(
            identifier,
            int(settings["planner_history_capacity"]),
            int(settings["planner_horizon"]),
            float(settings["planner_discount"]),
        )
    raise ValueError(f"unknown proposer kind: {kind}")


@dataclass(frozen=True)
class SentinelFrame:
    observation: VisibleObservation
    previous_feedback: PublicFeedback | None
    published_memory_age: int
    steps_remaining: int


@dataclass(frozen=True)
class ActivationRecord:
    initial: tuple[str, ...]
    extra_round: tuple[str, ...] = ()
    scheduler_tag: str = "event"
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require(
            all(kind in PROPOSER_ORDER for kind in self.initial), "initial round contains a non-proposer"
        )
        _require(self.extra_round in ((), (VERIFIER_ID,)), "only the verifier may occupy the extra round")


class EventSentinel:

    def __init__(self, config: Mapping[str, Any]):
        self.settings = config["sentinel"]
        self.previous_structural: tuple[int, ...] | None = None

    def reset_episode(self) -> None:
        self.previous_structural = None

    def state_payload(self) -> Any:
        return {"previous_structural": self.previous_structural}

    def select(self, frame: SentinelFrame) -> tuple[ActivationRecord, AbstractWork]:
        structural = frame.observation.structural_vector()
        work = AbstractWork(
            scalar_ops=len(structural) + len(frame.observation.novelty_channels),
            comparisons=len(structural) + 5,
            table_reads=len(structural) + len(frame.observation.novelty_channels) + 4,
            table_writes=1,
        )
        raw_novelty = sum(frame.observation.novelty_channels)
        delta = (
            0
            if self.previous_structural is None
            else sum(
                int(left != right) for left, right in zip(structural, self.previous_structural, strict=True)
            )
        )
        self.previous_structural = structural
        residual = bool(
            frame.previous_feedback is not None
            and (
                frame.previous_feedback.blocked
                or frame.previous_feedback.reward < float(self.settings["low_reward_threshold"])
            )
        )
        active = ["reactive_spatial"]
        reasons = ["baseline-reflex"]
        if delta >= int(self.settings["structural_delta_threshold"]) or frame.published_memory_age >= int(
            self.settings["memory_age_threshold"]
        ):
            active.append("episodic_retrieval")
            reasons.append("structural-change-or-memory-age")
        if residual or frame.steps_remaining <= int(self.settings["deadline_steps"]):
            active.append("short_horizon_planner")
            reasons.append("public-residual-or-deadline")
        _ = raw_novelty
        ordered = tuple(kind for kind in PROPOSER_ORDER if kind in active)
        return ActivationRecord(ordered, (), "event", tuple(reasons)), work


def round_activation_counts(records: Sequence[ActivationRecord]) -> dict[str, int]:
    counts = {f"initial:{kind}": 0 for kind in PROPOSER_ORDER}
    counts[f"extra:{VERIFIER_ID}"] = 0
    for record in records:
        for kind in record.initial:
            counts[f"initial:{kind}"] += 1
        counts[f"extra:{VERIFIER_ID}"] += int(record.extra_round == (VERIFIER_ID,))
    return counts


def _exact_positions(size: int, count: int) -> set[int]:
    if count <= 0:
        return set()
    if count >= size:
        return set(range(size))
    positions = {min(size - 1, ((2 * index + 1) * size) // (2 * count)) for index in range(count)}
    cursor = 0
    while len(positions) < count:
        positions.add(cursor)
        cursor += 1
    return positions


def periodic_round_matched_schedule(records: Sequence[ActivationRecord]) -> list[ActivationRecord]:
    size = len(records)
    counts = round_activation_counts(records)
    initial_positions = {kind: _exact_positions(size, counts[f"initial:{kind}"]) for kind in PROPOSER_ORDER}
    extra_positions = _exact_positions(size, counts[f"extra:{VERIFIER_ID}"])
    return [
        ActivationRecord(
            tuple(kind for kind in PROPOSER_ORDER if tick in initial_positions[kind]),
            (VERIFIER_ID,) if tick in extra_positions else (),
            "periodic",
            ("periodic-round-matched",),
        )
        for tick in range(size)
    ]


def shuffled_round_matched_schedule(records: Sequence[ActivationRecord], seed: int) -> list[ActivationRecord]:
    size = len(records)
    counts = round_activation_counts(records)
    rng = random.Random(seed)
    initial_positions = {
        kind: set(rng.sample(range(size), counts[f"initial:{kind}"])) for kind in PROPOSER_ORDER
    }
    extra_positions = set(rng.sample(range(size), counts[f"extra:{VERIFIER_ID}"]))
    return [
        ActivationRecord(
            tuple(kind for kind in PROPOSER_ORDER if tick in initial_positions[kind]),
            (VERIFIER_ID,) if tick in extra_positions else (),
            "shuffled",
            ("shuffled-round-matched",),
        )
        for tick in range(size)
    ]


def shuffled_coalition_matched_schedule(
    records: Sequence[ActivationRecord],
    seed: int,
) -> list[ActivationRecord]:

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled


@dataclass(frozen=True)
class PreparedDecision:
    observation: VisibleObservation
    activation: ActivationRecord
    proposals: tuple[ProposalMessage, ...]
    active_ids: tuple[str, ...]
    work: AbstractWork
    pre_bus_state_sha256: str


@dataclass(frozen=True)
class Resolution:
    action: int
    chosen_message_id: str | None
    delivered: tuple[ProposalMessage, ...]
    verification: VerificationMessage | None
    work: AbstractWork
    message_bytes: int
    delayed_planner_for_next_tick: ProposalMessage | None
    verifier_executed: bool


def _charge_proposal_validation(message: ProposalMessage, work: AbstractWork) -> None:
    body = message.payload()
    body.pop("message_id")
    encoded = canonical_bytes(body)
    work.bytes_serialized += len(encoded)
    work.bytes_hashed += len(encoded)


def _scheduler_work(tag: str, active_count: int) -> AbstractWork:
    if tag == "event":
        return AbstractWork()
    if tag == "always":
        return AbstractWork(comparisons=1, table_reads=1)
    if tag == "single":
        return AbstractWork(comparisons=1)
    if tag == "periodic":
        return AbstractWork(scalar_ops=2, comparisons=4, table_reads=active_count + 1)
    if tag == "shuffled":
        return AbstractWork(scalar_ops=3, comparisons=4, table_reads=active_count + 1)
    if tag == "coalition_shuffle":
        return AbstractWork(scalar_ops=3, comparisons=2, table_reads=active_count + 1)
    if tag == "replay":
        return AbstractWork(table_reads=active_count + 1)
    if tag == "homogeneous":
        return AbstractWork(scalar_ops=2, table_reads=active_count + 1)
    return AbstractWork(comparisons=1)


def _replace_proposal(
    message: ProposalMessage,
    *,
    action: int,
    evidence_suffix: str,
    work: AbstractWork,
) -> ProposalMessage:
    body = message.payload()
    body.pop("message_id")
    body["proposed_action"] = int(action)
    body["evidence"] = list(message.evidence + (evidence_suffix,))
    body_bytes = canonical_bytes(body)
    work.bytes_serialized += len(body_bytes)
    work.bytes_hashed += len(body_bytes)
    body["evidence"] = tuple(body["evidence"])
    body["provenance"] = message.provenance
    return ProposalMessage(message_id=hashlib.sha256(body_bytes).hexdigest(), **body)  # type: ignore[arg-type]


def apply_message_condition(
    config: Mapping[str, Any],
    prepared: PreparedDecision,
    condition: str,
    delayed_planner: ProposalMessage | None,
) -> tuple[tuple[ProposalMessage, ...], ProposalMessage | None, AbstractWork]:
    _require(
        condition in DIRECT_INTERVENTIONS or condition == "restored_planner_link", "unknown intervention"
    )
    observation = prepared.observation
    current_planner = next(
        (proposal for proposal in prepared.proposals if proposal.specialist_kind == "short_horizon_planner"),
        None,
    )
    work = AbstractWork(comparisons=len(prepared.proposals), table_reads=len(prepared.proposals))
    if condition == "no_message":
        delivered: list[ProposalMessage] = []
    elif condition in ("reactive_link_lesion", "episodic_link_lesion", "planner_link_lesion"):
        lesion_target = {
            "reactive_link_lesion": "reactive_spatial",
            "episodic_link_lesion": "episodic_retrieval",
            "planner_link_lesion": "short_horizon_planner",
        }[condition]
        delivered = [proposal for proposal in prepared.proposals if proposal.specialist_kind != lesion_target]
    elif condition == "planner_channel_delay":
        delivered = [
            proposal for proposal in prepared.proposals if proposal.specialist_kind != "short_horizon_planner"
        ]
        if delayed_planner is not None:
            _charge_proposal_validation(delayed_planner, work)
            if (
                delayed_planner.usable(
                    observation,
                    int(config["messages"]["delayed_max_age"]),
                )
                and delayed_planner.age(observation) == 1
            ):
                delivered.append(delayed_planner)
    elif condition == "wrong_planner_message":
        delivered = []
        for proposal in prepared.proposals:
            if proposal.specialist_kind == "short_horizon_planner":
                delivered.append(
                    _replace_proposal(
                        proposal,
                        action=(proposal.proposed_action + 2) % 4,
                        evidence_suffix="adversarial-wrong-message",
                        work=work,
                    )
                )
            else:
                delivered.append(proposal)
    else:
        delivered = list(prepared.proposals)
    transmitted = sum(proposal.encoded_bytes for proposal in delivered)
    work.bytes_serialized += transmitted
    return tuple(delivered), current_planner, work


def _resolve_prepared(
    config: Mapping[str, Any],
    verifier: ContradictionVerifier,
    prepared: PreparedDecision,
    condition: str,
    delayed_planner: ProposalMessage | None,
) -> Resolution:
    delivered, next_delayed, work = apply_message_condition(config, prepared, condition, delayed_planner)
    observation = prepared.observation
    verification: VerificationMessage | None = None
    if prepared.activation.extra_round == (VERIFIER_ID,) and condition != "verifier_lesion":
        verification, verifier_work = verifier.verify(observation, delivered)
        work.add(verifier_work)
    allowed_age = (
        int(config["messages"]["delayed_max_age"])
        if condition == "planner_channel_delay"
        else int(config["messages"]["clean_max_age"])
    )
    valid = []
    for proposal in delivered:
        _charge_proposal_validation(proposal, work)
        if proposal.usable(observation, allowed_age):
            valid.append(proposal)
    work.comparisons += 3 * len(delivered)
    if not valid:
        action = _safe_fallback(observation)
        chosen: ProposalMessage | None = None
        work.comparisons += 5
    else:
        age_discount = float(config["messages"]["age_discount"])

        def score(proposal: ProposalMessage) -> tuple[float, str]:
            age_factor = age_discount ** proposal.age(observation)
            value = proposal.confidence * age_factor + 0.25 * proposal.expected_progress
            value -= min(0.1, proposal.producer_work_units / 1_000_000)
            if verification is not None and not verification.abstained:
                if verification.endorsed_message_id == proposal.message_id:
                    value += 0.15 * verification.confidence
                if proposal.message_id in verification.contradicted_message_ids:
                    value -= 0.15 * verification.confidence
            return value, proposal.specialist_id

        chosen = max(valid, key=score)
        action = chosen.proposed_action
        work.scalar_ops += 8 * len(valid)
        work.comparisons += 4 * len(valid)
        work.table_reads += 3 * len(valid)
    message_bytes = sum(proposal.encoded_bytes for proposal in delivered)
    if verification is not None:
        message_bytes += verification.encoded_bytes
        work.bytes_serialized += verification.encoded_bytes
    return Resolution(
        action=action,
        chosen_message_id=chosen.message_id if chosen is not None else None,
        delivered=valid and tuple(valid) or (),
        verification=verification,
        work=work,
        message_bytes=message_bytes,
        delayed_planner_for_next_tick=next_delayed,
        verifier_executed=verification is not None,
    )


class CoalitionController:

    def __init__(self, config: Mapping[str, Any], mode: str, single_kind: str | None = None):
        self.config = config
        self.mode = mode
        self.single_kind = single_kind
        self.weights = config["abstract_work"]["weights"]
        self.proposers = {kind: make_proposer(config, kind) for kind in PROPOSER_ORDER}
        self.verifier = ContradictionVerifier(float(config["specialists"]["verifier_momentum"]))
        self.sentinel = EventSentinel(config)
        self.previous_feedback: PublicFeedback | None = None
        self.published_memory_age = 10**6
        self.last_prepared: PreparedDecision | None = None
        self.last_resolution: Resolution | None = None
        self.hard_dispatch_violations = 0

    def reset_episode(self) -> None:
        self.sentinel.reset_episode()
        self.previous_feedback = None
        self.published_memory_age = 10**6
        self.last_prepared = None
        self.last_resolution = None

    def state_payload(self) -> Any:
        return {
            "mode": self.mode,
            "single_kind": self.single_kind,
            "proposers": {kind: proposer.state_payload() for kind, proposer in self.proposers.items()},
            "verifier": self.verifier.state_payload(),
            "sentinel": self.sentinel.state_payload(),
            "previous_feedback": dataclasses.asdict(self.previous_feedback)
            if self.previous_feedback
            else None,
            "published_memory_age": self.published_memory_age,
        }

    def clone(self) -> CoalitionController:
        return copy.deepcopy(self)

    def _activation(
        self,
        observation: VisibleObservation,
        fixed: ActivationRecord | None,
    ) -> tuple[ActivationRecord, AbstractWork]:
        if fixed is not None:
            scheduler_tag = "coalition_shuffle" if self.mode == "coalition_fixed" else fixed.scheduler_tag
            return fixed, _scheduler_work(scheduler_tag, len(fixed.initial) + len(fixed.extra_round))
        if self.mode == "event_triggered":
            return self.sentinel.select(
                SentinelFrame(
                    observation,
                    self.previous_feedback,
                    self.published_memory_age,
                    int(self.config["world"]["horizon"]) - observation.tick,
                )
            )
        if self.mode == "always_on":
            record = ActivationRecord(PROPOSER_ORDER, (VERIFIER_ID,), "always", ("always-on",))
            return record, _scheduler_work("always", 4)
        if self.mode == "tuned_best_single":
            _require(self.single_kind in PROPOSER_ORDER, "single proposer kind required")
            record = ActivationRecord((str(self.single_kind),), (), "single", ("tuned-single",))
            return record, _scheduler_work("single", 1)
        raise ValueError(f"mode {self.mode!r} requires a fixed activation")

    def prepare(
        self, observation: VisibleObservation, fixed: ActivationRecord | None = None
    ) -> PreparedDecision:
        record, work = self._activation(observation, fixed)
        before = {kind: proposer.telemetry.propose_calls for kind, proposer in self.proposers.items()}
        proposals: list[ProposalMessage] = []
        max_age = int(self.config["messages"]["delayed_max_age"])
        for kind in record.initial:
            proposal, proposal_work = self.proposers[kind].propose(observation, self.weights, max_age)
            proposals.append(proposal)
            work.add(proposal_work)
        if fixed is None and self.mode == "event_triggered":
            distinct = {proposal.proposed_action for proposal in proposals}
            if len(proposals) >= 2 and len(distinct) >= 2:
                record = dataclasses.replace(
                    record,
                    extra_round=(VERIFIER_ID,),
                    reasons=record.reasons + ("current-message-disagreement",),
                )
        for kind, proposer in self.proposers.items():
            expected = int(kind in record.initial)
            if proposer.telemetry.propose_calls - before[kind] != expected:
                self.hard_dispatch_violations += 1
        state_payload = self.state_payload()
        state_bytes = canonical_bytes(state_payload)
        prepared = PreparedDecision(
            observation=observation,
            activation=record,
            proposals=tuple(proposals),
            active_ids=record.initial,
            work=work,
            pre_bus_state_sha256=hashlib.sha256(state_bytes).hexdigest(),
        )
        return prepared

    def resolve(
        self,
        prepared: PreparedDecision,
        condition: str = "clean",
        delayed_planner: ProposalMessage | None = None,
    ) -> Resolution:
        before_verify = self.verifier.verify_calls
        resolution = _resolve_prepared(self.config, self.verifier, prepared, condition, delayed_planner)
        expected_verify = int(
            prepared.activation.extra_round == (VERIFIER_ID,) and condition != "verifier_lesion"
        )
        if self.verifier.verify_calls - before_verify != expected_verify:
            self.hard_dispatch_violations += 1
        self.last_prepared = prepared
        self.last_resolution = resolution
        return resolution

    def update(self, transition: VisibleTransition) -> AbstractWork:
        prepared = self.last_prepared
        resolution = self.last_resolution
        if prepared is None or resolution is None:
            raise ValueError("prepare/resolve must precede update")
        before = {kind: proposer.telemetry.update_calls for kind, proposer in self.proposers.items()}
        work = AbstractWork()
        for kind in prepared.active_ids:
            work.add(self.proposers[kind].update(transition))
        for kind, proposer in self.proposers.items():
            expected = int(kind in prepared.active_ids)
            if proposer.telemetry.update_calls - before[kind] != expected:
                self.hard_dispatch_violations += 1
        before_verifier_update = self.verifier.update_calls
        if resolution.verifier_executed:
            work.add(self.verifier.update(transition, resolution.delivered))
        if self.verifier.update_calls - before_verifier_update != int(resolution.verifier_executed):
            self.hard_dispatch_violations += 1
        self.previous_feedback = transition.feedback
        self.published_memory_age += 1
        if any(proposal.specialist_kind == "episodic_retrieval" for proposal in resolution.delivered):
            self.published_memory_age = 0
        return work


class HomogeneousController:

    def __init__(self, config: Mapping[str, Any], kind: str):
        _require(kind in PROPOSER_ORDER, "homogeneous kind must be a proposer")
        self.config = config
        self.kind = kind
        self.weights = config["abstract_work"]["weights"]
        self.copies = {
            f"homogeneous_{index}": make_proposer(config, kind, f"homogeneous_{index}") for index in range(4)
        }
        self.verifier = ContradictionVerifier(float(config["specialists"]["verifier_momentum"]))
        self.last_active_ids: tuple[str, ...] = ()
        self.last_prepared: PreparedDecision | None = None
        self.last_resolution: Resolution | None = None
        self.hard_dispatch_violations = 0

    def reset_episode(self) -> None:
        self.last_active_ids = ()
        self.last_prepared = None
        self.last_resolution = None

    def state_payload(self) -> Any:
        return {
            "copies": {identifier: proposer.state_payload() for identifier, proposer in self.copies.items()},
            "verifier": self.verifier.state_payload(),
        }

    def prepare(self, observation: VisibleObservation, reference: ActivationRecord) -> PreparedDecision:
        count = len(reference.initial)
        _require(0 <= count <= 4, "homogeneous reference count outside [0, 4]")
        identifiers = tuple(self.copies)
        start = observation.tick % len(identifiers)
        active_ids = tuple(identifiers[(start + offset) % len(identifiers)] for offset in range(count))
        before = {
            identifier: proposer.telemetry.propose_calls for identifier, proposer in self.copies.items()
        }
        work = _scheduler_work("homogeneous", count)
        proposals: list[ProposalMessage] = []
        for identifier in active_ids:
            proposal, proposal_work = self.copies[identifier].propose(
                observation,
                self.weights,
                int(self.config["messages"]["delayed_max_age"]),
            )
            proposals.append(proposal)
            work.add(proposal_work)
        for identifier, proposer in self.copies.items():
            if proposer.telemetry.propose_calls - before[identifier] != int(identifier in active_ids):
                self.hard_dispatch_violations += 1
        state_bytes = canonical_bytes(self.state_payload())
        prepared = PreparedDecision(
            observation=observation,
            activation=ActivationRecord(
                tuple(self.kind for _ in active_ids),
                reference.extra_round,
                "homogeneous",
                ("homogeneous-proposer-copies-with-relational-verifier",),
            ),
            proposals=tuple(proposals),
            active_ids=active_ids,
            work=work,
            pre_bus_state_sha256=hashlib.sha256(state_bytes).hexdigest(),
        )
        self.last_active_ids = active_ids
        self.last_prepared = prepared
        return prepared

    def resolve(self, prepared: PreparedDecision) -> Resolution:
        before_verify = self.verifier.verify_calls
        resolution = _resolve_prepared(self.config, self.verifier, prepared, "clean", None)
        expected_verify = int(prepared.activation.extra_round == (VERIFIER_ID,))
        if self.verifier.verify_calls - before_verify != expected_verify:
            self.hard_dispatch_violations += 1
        self.last_resolution = resolution
        return resolution

    def update(self, transition: VisibleTransition) -> AbstractWork:
        before = {identifier: proposer.telemetry.update_calls for identifier, proposer in self.copies.items()}
        work = AbstractWork()
        for identifier in self.last_active_ids:
            work.add(self.copies[identifier].update(transition))
        for identifier, proposer in self.copies.items():
            if proposer.telemetry.update_calls - before[identifier] != int(
                identifier in self.last_active_ids
            ):
                self.hard_dispatch_violations += 1
        resolution = self.last_resolution
        if resolution is None:
            raise ValueError("resolve must precede update")
        before_verifier_update = self.verifier.update_calls
        if resolution.verifier_executed:
            work.add(self.verifier.update(transition, resolution.delivered))
        if self.verifier.update_calls - before_verifier_update != int(resolution.verifier_executed):
            self.hard_dispatch_violations += 1
        return work


@dataclass
class BudgetLedger:
    total_budget: int
    total_steps: int
    credit: int = 0
    spent: int = 0
    issued_steps: int = 0

    def issue(self) -> int:
        base, remainder = divmod(self.total_budget, self.total_steps)
        amount = base + int(self.issued_steps < remainder)
        self.issued_steps += 1
        self.credit += amount
        return amount

    def spend(self, amount: int) -> None:
        _require(amount >= 0 and amount <= self.credit, "recurrent controller exceeded available budget")
        self.credit -= amount
        self.spent += amount


@dataclass(frozen=True)
class RecurrentTrace:
    action: int
    work: AbstractWork
    sweeps: int


class EqualBudgetRecurrentController:

    def __init__(
        self,
        config: Mapping[str, Any],
        seed: int,
        hidden_size: int,
        learning_rate: float,
        reservoir_scale: float,
        ledger: BudgetLedger,
    ):
        self.config = config
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.reservoir_scale = reservoir_scale
        self.discount = float(config["recurrent_control"]["discount"])
        self.weights = config["abstract_work"]["weights"]
        input_size = 12
        self.w_input = [
            [self._weight(seed, "input", row, column) for column in range(input_size)]
            for row in range(hidden_size)
        ]
        self.w_hidden = [
            [
                reservoir_scale * self._weight(seed, "hidden", row, column) / max(1, hidden_size)
                for column in range(hidden_size)
            ]
            for row in range(hidden_size)
        ]
        self.q_head = [[0.0 for _ in range(hidden_size)] for _ in range(4)]
        self.action_counts = [0, 0, 0, 0]
        self.hidden = [0.0 for _ in range(hidden_size)]
        self.ledger = ledger
        self.last_action = 0
        self.last_hidden = list(self.hidden)

    @staticmethod
    def _weight(seed: int, role: str, row: int, column: int) -> float:
        integer = _stable_int(seed, role, row, column, modulus=2_000_001)
        return (integer - 1_000_000) / 1_000_000

    def reset_episode(self) -> None:
        self.hidden = [0.0 for _ in range(self.hidden_size)]
        self.last_hidden = list(self.hidden)

    def state_payload(self) -> Any:
        return {
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "reservoir_scale": self.reservoir_scale,
            "hidden": self.hidden,
            "q_head": self.q_head,
            "action_counts": self.action_counts,
        }

    def _sweep_work(self, input_size: int) -> AbstractWork:
        multiplies_and_adds = 2 * self.hidden_size * (input_size + self.hidden_size)
        return AbstractWork(
            scalar_ops=multiplies_and_adds,
            nonlinearities=self.hidden_size,
            table_reads=2 * self.hidden_size * (input_size + self.hidden_size),
            table_writes=self.hidden_size,
        )

    def _head_work(self) -> AbstractWork:
        return AbstractWork(
            scalar_ops=2 * 4 * self.hidden_size + 12,
            comparisons=9,
            nonlinearities=4,
            table_reads=8 * self.hidden_size + 5,
            table_writes=1,
        )

    def _update_work(self) -> AbstractWork:
        return AbstractWork(
            scalar_ops=5 + 13 * self.hidden_size,
            comparisons=6,
            table_reads=12 * self.hidden_size,
            table_writes=self.hidden_size,
        )

    def _sweep(self, features: Sequence[float]) -> AbstractWork:
        new_hidden: list[float] = []
        for row in range(self.hidden_size):
            value = 0.0
            for column, feature in enumerate(features):
                value += self.w_input[row][column] * feature
            for column, hidden_value in enumerate(self.hidden):
                value += self.w_hidden[row][column] * hidden_value
            new_hidden.append(math.tanh(value))
        self.hidden = new_hidden
        return self._sweep_work(len(features))

    def act(self, observation: VisibleObservation) -> RecurrentTrace:
        self.ledger.issue()
        features = observation.full_vector()
        head_work = self._head_work()
        update_work = self._update_work()
        sweep_template = self._sweep_work(len(features))
        head_cost = head_work.total(self.weights)
        update_cost = update_work.total(self.weights)
        sweep_cost = sweep_template.total(self.weights)
        successor_sweep_cost = (
            sweep_cost if observation.tick < int(self.config["world"]["horizon"]) - 1 else 0
        )
        reserved_cost = head_cost + update_cost + successor_sweep_cost
        _require(
            self.ledger.credit >= reserved_cost,
            "event budget cannot fund recurrent action and successor TD update",
        )
        work = AbstractWork()
        sweeps = 0
        while self.ledger.credit >= reserved_cost + sweep_cost:
            sweep_work = self._sweep(features)
            sweep_actual = sweep_work.total(self.weights)
            self.ledger.spend(sweep_actual)
            work.add(sweep_work)
            sweeps += 1
        goal_prior = _goal_action(observation)
        scores = []
        for action in range(4):
            learned = sum(
                weight * hidden for weight, hidden in zip(self.q_head[action], self.hidden, strict=True)
            )
            exploration = 0.2 / math.sqrt(self.action_counts[action] + 1)
            scores.append(learned + exploration + 0.08 * int(action == goal_prior))
        action = max(range(4), key=lambda candidate: (scores[candidate], -candidate))
        self.ledger.spend(head_cost)
        work.add(head_work)
        self.last_action = action
        self.action_counts[action] += 1
        self.last_hidden = list(self.hidden)
        return RecurrentTrace(action, work, sweeps)

    def update(self, transition: VisibleTransition) -> AbstractWork:
        work = self._update_work()
        cost = work.total(self.weights)
        prediction = sum(
            weight * hidden
            for weight, hidden in zip(self.q_head[self.last_action], self.last_hidden, strict=True)
        )
        before = transition.before.relative_goal
        after = transition.after.relative_goal if transition.after is not None else before
        progress = (abs(before[0]) + abs(before[1])) - (abs(after[0]) + abs(after[1]))
        if transition.terminal:
            bootstrap = 0.0
        else:
            successor = transition.after
            if successor is None:
                raise ValueError("nonterminal recurrent update requires successor observation")
            successor_work = self._sweep(successor.full_vector())
            self.ledger.spend(successor_work.total(self.weights))
            work.add(successor_work)
            next_values = [
                sum(weight * hidden for weight, hidden in zip(self.q_head[action], self.hidden, strict=True))
                for action in range(4)
            ]
            bootstrap = max(next_values)
        self.ledger.spend(cost)
        target = (
            transition.feedback.reward
            + float(self.config["world"]["progress_weight"]) * progress
            + self.discount * bootstrap
        )
        error = target - prediction
        for index, hidden in enumerate(self.last_hidden):
            self.q_head[self.last_action][index] += self.learning_rate * error * hidden
        return work


@dataclass(frozen=True)
class EpisodeMetric:
    utility: float
    total_return: float
    success: int
    work_units: int
    work_components: Mapping[str, int]
    message_bytes: int
    niche_values: Mapping[str, float]
    actions: tuple[int, ...]
    actions_sha256: str


def compact_evidence(values: Sequence[float]) -> dict[str, Any]:
    materialized = [float(value) for value in values]
    return {
        "count": len(materialized),
        "sum": sum(materialized),
        "mean": statistics.fmean(materialized) if materialized else 0.0,
        "min": min(materialized) if materialized else 0.0,
        "max": max(materialized) if materialized else 0.0,
        "values_sha256": canonical_sha256(materialized),
    }


def validate_compact_evidence(series: Mapping[str, Any]) -> None:
    count_value = int(series["count"])
    _require(count_value >= 0, "negative compact evidence count")
    expected_mean = float(series["sum"]) / count_value if count_value else 0.0
    _require(
        math.isclose(float(series["mean"]), expected_mean, rel_tol=0.0, abs_tol=1e-12),
        "compact evidence mean mismatch",
    )
    _require(
        isinstance(series["values_sha256"], str) and len(series["values_sha256"]) == 64,
        "compact evidence digest malformed",
    )
    if count_value == 0:
        _require(
            float(series["min"]) == 0.0 and float(series["max"]) == 0.0,
            "empty compact evidence bounds mismatch",
        )
    else:
        _require(
            float(series["min"]) <= float(series["mean"]) <= float(series["max"]),
            "compact evidence bounds mismatch",
        )


@dataclass
class DirectEffects:
    samples: MutableMapping[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    restoration_gains: list[float] = field(default_factory=list)
    fork_identity_violations: int = 0
    channel_delay_origin_violations: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "metrics": {name: compact_evidence(values) for name, values in sorted(self.samples.items())},
            "restoration": compact_evidence(self.restoration_gains),
            "fork_identity_violations": self.fork_identity_violations,
            "channel_delay_origin_violations": self.channel_delay_origin_violations,
        }


@dataclass
class ArmAccumulator:
    name: str
    work_weights: Mapping[str, int]
    sensitivity_factors: Sequence[Any]
    episodes: list[EpisodeMetric] = field(default_factory=list)
    schedules: list[list[ActivationRecord]] = field(default_factory=list)
    round_counts: Counter[str] = field(default_factory=Counter)
    coalition_sizes: Counter[int] = field(default_factory=Counter)
    noise_opportunities: int = 0
    noise_expensive_activations: int = 0
    noise_activation_pair_differences: int = 0
    noise_pair_opportunities: int = 0
    change_opportunities: int = 0
    change_expensive_activations: int = 0
    hard_dispatch_violations: int = 0
    direct_effects: DirectEffects | None = None

    def summary(self, horizon: int) -> dict[str, Any]:
        def mean(values: Sequence[float]) -> float:
            return statistics.fmean(values) if values else 0.0

        steps = len(self.episodes) * horizon
        total_work = sum(episode.work_units for episode in self.episodes)
        component_totals = {
            name: sum(int(episode.work_components[name]) for episode in self.episodes)
            for name in dataclasses.asdict(AbstractWork())
        }
        sensitivity = accounting_sensitivity(
            component_totals,
            self.work_weights,
            self.sensitivity_factors,
        )
        _require(sensitivity["nominal"] == total_work, "arm nominal/component work mismatch")
        total_bytes = sum(episode.message_bytes for episode in self.episodes)
        quartile = max(1, len(self.episodes) // 4) if self.episodes else 0
        first_quartile = self.episodes[:quartile]
        last_quartile = self.episodes[-quartile:] if quartile else []
        activation_multiset_hashes = [
            canonical_sha256(
                sorted(
                    (
                        list(record.initial),
                        list(record.extra_round),
                    )
                    for record in schedule
                )
            )
            for schedule in self.schedules
        ]
        return {
            "arm": self.name,
            "episode_count": len(self.episodes),
            "steps": steps,
            "mean_utility": mean([episode.utility for episode in self.episodes]),
            "mean_return": mean([episode.total_return for episode in self.episodes]),
            "success_rate": mean([episode.success for episode in self.episodes]),
            "total_abstract_work": total_work,
            "work_component_totals": component_totals,
            "accounting_sensitivity": sensitivity,
            "abstract_work_per_step": total_work / steps if steps else 0.0,
            "total_message_bytes": total_bytes,
            "message_bytes_per_step": total_bytes / steps if steps else 0.0,
            "round_activation_counts": dict(sorted(self.round_counts.items())),
            "activation_record_multiset_sha256_per_episode": activation_multiset_hashes,
            "coalition_size_histogram": {
                str(size): count for size, count in sorted(self.coalition_sizes.items())
            },
            "noise_false_expensive_rate": (
                self.noise_expensive_activations / self.noise_opportunities
                if self.noise_opportunities
                else 0.0
            ),
            "noise_activation_pair_differences": self.noise_activation_pair_differences,
            "noise_pair_opportunities": self.noise_pair_opportunities,
            "change_expensive_rate": (
                self.change_expensive_activations / self.change_opportunities
                if self.change_opportunities
                else 0.0
            ),
            "hard_dispatch_violations": self.hard_dispatch_violations,
            "sample_efficiency": {
                "first_quartile_mean_utility": mean([episode.utility for episode in first_quartile]),
                "last_quartile_mean_utility": mean([episode.utility for episode in last_quartile]),
                "within_run_learning_gain": mean([episode.utility for episode in last_quartile])
                - mean([episode.utility for episode in first_quartile]),
            },
            "per_episode": [dataclasses.asdict(episode) for episode in self.episodes],
            "direct_effects": self.direct_effects.summary() if self.direct_effects is not None else None,
        }


def _episode_utility(success: int, total_return: float, config: Mapping[str, Any]) -> float:
    world = config["world"]
    scale = abs(float(world["goal_reward"])) + int(world["horizon"]) * (
        abs(float(world["step_cost"])) + abs(float(world["blocked_cost"]))
    )
    return float(success) + 0.01 * max(-1.0, min(1.0, total_return / max(scale, 1e-9)))


def _record_round_counts(counter: Counter[str], record: ActivationRecord) -> None:
    for kind in record.initial:
        counter[f"initial:{kind}"] += 1
    if record.extra_round:
        counter[f"extra:{VERIFIER_ID}"] += 1


def _state_fork_sha256(world: PartialChangePointWorld, controller: CoalitionController) -> str:
    return canonical_sha256({"world": world.state_payload(), "controller": controller.state_payload()})


def _counterfactual_value(
    world: PartialChangePointWorld,
    action: int,
    progress_weight: float,
) -> float:
    clone = world.clone()
    return evaluator_step_value(clone.step(action), progress_weight)


def _two_tick_channel_delay_assay(
    config: Mapping[str, Any],
    world: PartialChangePointWorld,
    controller: CoalitionController,
    prepared: PreparedDecision,
) -> tuple[float, float, int] | None:

    origin_planner = next(
        (proposal for proposal in prepared.proposals if proposal.specialist_kind == "short_horizon_planner"),
        None,
    )
    if origin_planner is None or prepared.observation.tick >= int(config["world"]["horizon"]) - 1:
        return None
    progress_weight = float(config["world"]["progress_weight"])
    branch_worlds = [world.clone() for _ in range(3)]
    branch_controllers = [controller.clone() for _ in range(3)]
    origin_hashes = [
        _state_fork_sha256(branch_world, branch_controller)
        for branch_world, branch_controller in zip(branch_worlds, branch_controllers, strict=True)
    ]
    violations = int(len(set(origin_hashes)) != 1)
    clean_world, delayed_world, lesion_world = branch_worlds
    clean_controller, delayed_controller, lesion_controller = branch_controllers

    first_resolutions = (
        clean_controller.resolve(prepared, "clean"),
        delayed_controller.resolve(prepared, "planner_link_lesion"),
        lesion_controller.resolve(prepared, "planner_link_lesion"),
    )
    totals: list[float] = []
    for branch_world, branch_controller, resolution in zip(
        branch_worlds,
        branch_controllers,
        first_resolutions,
        strict=True,
    ):
        transition = branch_world.step(resolution.action)
        branch_controller.update(transition.visible)
        totals.append(evaluator_step_value(transition, progress_weight))

    clean_prepared = clean_controller.prepare(clean_world.observe())
    replay_record = dataclasses.replace(clean_prepared.activation, scheduler_tag="replay")
    delayed_prepared = delayed_controller.prepare(delayed_world.observe(), replay_record)
    lesion_prepared = lesion_controller.prepare(lesion_world.observe(), replay_record)
    second_resolutions = (
        clean_controller.resolve(clean_prepared, "clean"),
        delayed_controller.resolve(delayed_prepared, "planner_channel_delay", origin_planner),
        lesion_controller.resolve(lesion_prepared, "planner_link_lesion"),
    )
    delayed_delivery = second_resolutions[1].delivered
    origin_delivered = [
        proposal
        for proposal in delayed_delivery
        if proposal.message_id == origin_planner.message_id
        and proposal.source_event_id == prepared.observation.event_id
        and proposal.age(delayed_prepared.observation) == 1
    ]
    violations += int(len(origin_delivered) != 1)
    for index, (branch_world, branch_controller, resolution) in enumerate(
        zip(branch_worlds, branch_controllers, second_resolutions, strict=True)
    ):
        transition = branch_world.step(resolution.action)
        branch_controller.update(transition.visible)
        totals[index] += evaluator_step_value(transition, progress_weight)
    clean_value, delayed_value, lesion_value = totals
    return clean_value - delayed_value, delayed_value - lesion_value, violations


def _collect_direct_effects(
    config: Mapping[str, Any],
    effects: DirectEffects,
    world: PartialChangePointWorld,
    controller: CoalitionController,
    prepared: PreparedDecision,
    clean_resolution: Resolution,
) -> None:
    progress_weight = float(config["world"]["progress_weight"])
    clean_value = _counterfactual_value(world, clean_resolution.action, progress_weight)
    no_message = _resolve_prepared(
        config,
        copy.deepcopy(controller.verifier),
        prepared,
        "no_message",
        None,
    )
    effects.samples["clean_minus_no_message"].append(
        clean_value - _counterfactual_value(world, no_message.action, progress_weight)
    )
    lesion_conditions = {
        "reactive_spatial": ("reactive_link_lesion", "clean_minus_reactive_lesion"),
        "episodic_retrieval": ("episodic_link_lesion", "clean_minus_episodic_lesion"),
        "short_horizon_planner": ("planner_link_lesion", "clean_minus_planner_lesion"),
    }
    lesion_values: dict[str, float] = {}
    niche = world._niche(prepared.observation.tick)
    for kind, (condition, metric_name) in lesion_conditions.items():
        if not any(proposal.specialist_kind == kind for proposal in prepared.proposals):
            continue
        lesion = _resolve_prepared(
            config,
            copy.deepcopy(controller.verifier),
            prepared,
            condition,
            None,
        )
        lesion_value = _counterfactual_value(world, lesion.action, progress_weight)
        lesion_values[kind] = lesion_value
        penalty = clean_value - lesion_value
        effects.samples[metric_name].append(penalty)
        suffix = "matching_niche" if niche == kind else "outside_niche"
        effects.samples[f"{metric_name}_{suffix}"].append(penalty)
    planner_present = "short_horizon_planner" in lesion_values
    if planner_present:
        wrong = _resolve_prepared(
            config,
            copy.deepcopy(controller.verifier),
            prepared,
            "wrong_planner_message",
            None,
        )
        effects.samples["clean_minus_wrong_planner"].append(
            clean_value - _counterfactual_value(world, wrong.action, progress_weight)
        )
    delay_assay = _two_tick_channel_delay_assay(config, world, controller, prepared)
    if delay_assay is not None:
        clean_minus_delay, delay_minus_lesion, violations = delay_assay
        effects.samples["clean_minus_two_tick_delay"].append(clean_minus_delay)
        effects.samples["two_tick_delay_minus_two_tick_lesion"].append(delay_minus_lesion)
        effects.channel_delay_origin_violations += violations
    if prepared.activation.extra_round == (VERIFIER_ID,):
        without_verifier = _resolve_prepared(
            config,
            copy.deepcopy(controller.verifier),
            prepared,
            "verifier_lesion",
            None,
        )
        effects.samples["clean_minus_verifier_lesion"].append(
            clean_value - _counterfactual_value(world, without_verifier.action, progress_weight)
        )


def _run_fixed_branch_step(
    controller: CoalitionController,
    world: PartialChangePointWorld,
    record: ActivationRecord,
    condition: str,
) -> float:
    replay = dataclasses.replace(record, scheduler_tag="replay")
    prepared = controller.prepare(world.observe(), replay)
    resolution = controller.resolve(prepared, condition)
    transition = world.step(resolution.action)
    controller.update(transition.visible)
    return evaluator_step_value(transition, float(controller.config["world"]["progress_weight"]))


def _restoration_counterfactual(
    config: Mapping[str, Any],
    snapshot_world: PartialChangePointWorld,
    snapshot_controller: CoalitionController,
    schedule: Sequence[ActivationRecord],
) -> tuple[float, int]:
    lesion_start = int(config["interventions"]["lesion_start_tick"])
    restoration_tick = int(config["interventions"]["restoration_tick"])
    lesion_world = snapshot_world.clone()
    lesion_controller = snapshot_controller.clone()
    for tick in range(lesion_start, restoration_tick):
        _run_fixed_branch_step(
            lesion_controller,
            lesion_world,
            schedule[tick],
            "planner_link_lesion",
        )
    continued_world = lesion_world.clone()
    continued_controller = lesion_controller.clone()
    restored_world = lesion_world.clone()
    restored_controller = lesion_controller.clone()
    before_hash = _state_fork_sha256(continued_world, continued_controller)
    restored_hash = _state_fork_sha256(restored_world, restored_controller)
    violation = int(before_hash != restored_hash)
    continued_value = 0.0
    restored_value = 0.0
    for tick in range(restoration_tick, int(config["world"]["horizon"])):
        continued_value += _run_fixed_branch_step(
            continued_controller,
            continued_world,
            schedule[tick],
            "planner_link_lesion",
        )
        restored_value += _run_fixed_branch_step(
            restored_controller,
            restored_world,
            schedule[tick],
            "restored_planner_link",
        )
    denominator = max(1, int(config["world"]["horizon"]) - restoration_tick)
    return (restored_value - continued_value) / denominator, violation


def run_coalition_arm(
    config: Mapping[str, Any],
    *,
    seed: int,
    split: str,
    episodes: int,
    name: str,
    mode: str,
    single_kind: str | None = None,
    fixed_schedules: Sequence[Sequence[ActivationRecord]] | None = None,
    capture_interventions: bool = False,
) -> ArmAccumulator:
    controller = CoalitionController(config, mode, single_kind)
    accumulator = ArmAccumulator(
        name,
        config["abstract_work"]["weights"],
        config["abstract_work"]["sensitivity_factors"],
    )
    if capture_interventions:
        accumulator.direct_effects = DirectEffects()
    weights = config["abstract_work"]["weights"]
    horizon = int(config["world"]["horizon"])
    change_points = tuple(int(value) for value in config["world"]["change_points"])
    radius = int(config["world"]["change_window_radius"])
    intervention_limit = int(config["splits"]["intervention_episodes"])
    for episode in range(episodes):
        world = PartialChangePointWorld(config, seed, split, episode)
        controller.reset_episode()
        schedule: list[ActivationRecord] = []
        actions: list[int] = []
        total_return = 0.0
        total_work = 0
        episode_work = AbstractWork()
        total_bytes = 0
        success = 0
        niche_values: MutableMapping[str, list[float]] = defaultdict(list)
        branch_world: PartialChangePointWorld | None = None
        branch_controller: CoalitionController | None = None
        while not world.terminal:
            observation = world.observe()
            fixed = fixed_schedules[episode][observation.tick] if fixed_schedules is not None else None
            quiet_controller: CoalitionController | None = None
            capture_this_episode = capture_interventions and episode < intervention_limit
            if capture_this_episode and observation.tick in set(config["world"]["noise_ticks"]):
                quiet_controller = controller.clone()
            if capture_this_episode and observation.tick == int(config["interventions"]["lesion_start_tick"]):
                branch_world = world.clone()
                branch_controller = controller.clone()
            prepared = controller.prepare(observation, fixed)
            if quiet_controller is not None:
                quiet_observation = dataclasses.replace(
                    observation,
                    novelty_channels=tuple(0 for _ in observation.novelty_channels),
                )
                quiet_prepared = quiet_controller.prepare(quiet_observation, fixed)
                accumulator.noise_activation_pair_differences += int(
                    quiet_prepared.activation != prepared.activation
                )
                accumulator.noise_pair_opportunities += 1
            resolution = controller.resolve(prepared, "clean")
            if capture_this_episode and accumulator.direct_effects is not None:
                _collect_direct_effects(
                    config,
                    accumulator.direct_effects,
                    world,
                    controller,
                    prepared,
                    resolution,
                )
            transition = world.step(resolution.action)
            update_work = controller.update(transition.visible)
            step_work = prepared.work.copy().add(resolution.work).add(update_work)
            episode_work.add(step_work)
            total_work += step_work.total(weights)
            total_bytes += resolution.message_bytes
            total_return += transition.visible.feedback.reward
            success = max(success, int(transition.visible.feedback.reached_goal))
            actions.append(resolution.action)
            schedule.append(prepared.activation)
            _record_round_counts(accumulator.round_counts, prepared.activation)
            size = len(prepared.activation.initial) + len(prepared.activation.extra_round)
            accumulator.coalition_sizes[size] += 1
            expensive = size > 1 or any(kind != "reactive_spatial" for kind in prepared.activation.initial)
            if transition.noise_label:
                accumulator.noise_opportunities += 1
                accumulator.noise_expensive_activations += int(expensive)
            in_change_window = any(abs(observation.tick - point) <= radius for point in change_points)
            if in_change_window:
                accumulator.change_opportunities += 1
                accumulator.change_expensive_activations += int(expensive)
            value = evaluator_step_value(transition, float(config["world"]["progress_weight"]))
            if transition.niche_label is not None:
                niche_values[transition.niche_label].append(value)
        if (
            capture_interventions
            and accumulator.direct_effects is not None
            and branch_world is not None
            and branch_controller is not None
        ):
            gain, violation = _restoration_counterfactual(
                config,
                branch_world,
                branch_controller,
                schedule,
            )
            accumulator.direct_effects.restoration_gains.append(gain)
            accumulator.direct_effects.fork_identity_violations += violation
        accumulator.schedules.append(schedule)
        accumulator.episodes.append(
            EpisodeMetric(
                utility=_episode_utility(success, total_return, config),
                total_return=total_return,
                success=success,
                work_units=total_work,
                work_components=dataclasses.asdict(episode_work),
                message_bytes=total_bytes,
                niche_values={
                    kind: statistics.fmean(values) if values else 0.0 for kind, values in niche_values.items()
                },
                actions=tuple(actions),
                actions_sha256=canonical_sha256(actions),
            )
        )
        accumulator.hard_dispatch_violations = controller.hard_dispatch_violations
    _require(
        all(len(schedule) == horizon for schedule in accumulator.schedules), "fixed-horizon invariant failed"
    )
    return accumulator


def run_homogeneous_arm(
    config: Mapping[str, Any],
    *,
    seed: int,
    split: str,
    episodes: int,
    kind: str,
    reference_schedules: Sequence[Sequence[ActivationRecord]],
) -> ArmAccumulator:
    controller = HomogeneousController(config, kind)
    accumulator = ArmAccumulator(
        "homogeneous_matched",
        config["abstract_work"]["weights"],
        config["abstract_work"]["sensitivity_factors"],
    )
    weights = config["abstract_work"]["weights"]
    for episode in range(episodes):
        world = PartialChangePointWorld(config, seed, split, episode)
        controller.reset_episode()
        actions: list[int] = []
        total_return = 0.0
        total_work = 0
        episode_work = AbstractWork()
        total_bytes = 0
        success = 0
        niche_values: MutableMapping[str, list[float]] = defaultdict(list)
        while not world.terminal:
            observation = world.observe()
            reference = reference_schedules[episode][observation.tick]
            prepared = controller.prepare(observation, reference)
            resolution = controller.resolve(prepared)
            transition = world.step(resolution.action)
            update_work = controller.update(transition.visible)
            step_work = prepared.work.copy().add(resolution.work).add(update_work)
            episode_work.add(step_work)
            total_work += step_work.total(weights)
            total_bytes += resolution.message_bytes
            total_return += transition.visible.feedback.reward
            success = max(success, int(transition.visible.feedback.reached_goal))
            actions.append(resolution.action)
            if transition.niche_label is not None:
                niche_values[transition.niche_label].append(
                    evaluator_step_value(transition, float(config["world"]["progress_weight"]))
                )
            _record_round_counts(accumulator.round_counts, prepared.activation)
            accumulator.coalition_sizes[len(prepared.active_ids) + len(prepared.activation.extra_round)] += 1
        accumulator.episodes.append(
            EpisodeMetric(
                _episode_utility(success, total_return, config),
                total_return,
                success,
                total_work,
                dataclasses.asdict(episode_work),
                total_bytes,
                {kind: statistics.fmean(values) for kind, values in niche_values.items()},
                tuple(actions),
                canonical_sha256(actions),
            )
        )
        accumulator.hard_dispatch_violations = controller.hard_dispatch_violations
    return accumulator


def run_recurrent_arm(
    config: Mapping[str, Any],
    *,
    seed: int,
    split: str,
    episodes: int,
    total_budget: int,
    hyperparameters: Mapping[str, Any],
    name: str = "equal_budget_recurrent",
) -> ArmAccumulator:
    horizon = int(config["world"]["horizon"])
    steps = episodes * horizon
    ledger = BudgetLedger(total_budget, steps)
    controller = EqualBudgetRecurrentController(
        config,
        seed,
        int(hyperparameters["hidden_size"]),
        float(hyperparameters["learning_rate"]),
        float(hyperparameters["reservoir_scale"]),
        ledger,
    )
    accumulator = ArmAccumulator(
        name,
        config["abstract_work"]["weights"],
        config["abstract_work"]["sensitivity_factors"],
    )
    weights = config["abstract_work"]["weights"]
    for episode in range(episodes):
        world = PartialChangePointWorld(config, seed, split, episode)
        controller.reset_episode()
        actions: list[int] = []
        total_return = 0.0
        total_work = 0
        episode_work = AbstractWork()
        success = 0
        niche_values: MutableMapping[str, list[float]] = defaultdict(list)
        while not world.terminal:
            trace = controller.act(world.observe())
            transition = world.step(trace.action)
            update_work = controller.update(transition.visible)
            step_work = trace.work.copy().add(update_work)
            episode_work.add(step_work)
            total_work += step_work.total(weights)
            total_return += transition.visible.feedback.reward
            success = max(success, int(transition.visible.feedback.reached_goal))
            actions.append(trace.action)
            if transition.niche_label is not None:
                niche_values[transition.niche_label].append(
                    evaluator_step_value(transition, float(config["world"]["progress_weight"]))
                )
        accumulator.episodes.append(
            EpisodeMetric(
                _episode_utility(success, total_return, config),
                total_return,
                success,
                total_work,
                dataclasses.asdict(episode_work),
                0,
                {kind: statistics.fmean(values) for kind, values in niche_values.items()},
                tuple(actions),
                canonical_sha256(actions),
            )
        )
    summary = accumulator.summary(horizon)
    _require(summary["total_abstract_work"] == ledger.spent, "recurrent ledger/work trace mismatch")
    accumulator.round_counts["budget:residual"] = ledger.credit
    return accumulator


def mean_ci(values: Sequence[float], t_critical: float) -> dict[str, float]:
    _require(bool(values), "cannot summarize an empty vector")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"mean": mean, "lower": mean, "upper": mean, "half_width": 0.0}
    half = t_critical * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "half_width": half}


def _utilities(summary: Mapping[str, Any]) -> list[float]:
    return [float(episode["utility"]) for episode in summary["per_episode"]]


def _niche_values(summary: Mapping[str, Any], niche: str) -> list[float | None]:
    return [
        float(episode["niche_values"][niche]) if niche in episode["niche_values"] else None
        for episode in summary["per_episode"]
    ]


def _required_niche_value(value: float | None) -> float:
    if value is None:
        raise ValueError("required niche value is missing")
    return value


def _candidate_key(hidden_size: int, learning_rate: float, reservoir_scale: float) -> str:
    return f"h{hidden_size}:lr{learning_rate:.6f}:rs{reservoir_scale:.6f}"


def _candidate_payload(key: str) -> dict[str, Any]:
    hidden, learning, reservoir = key.split(":")
    return {
        "hidden_size": int(hidden[1:]),
        "learning_rate": float(learning[2:]),
        "reservoir_scale": float(reservoir[2:]),
    }


def _canonical_candidate_keys(candidates: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(key) for key in candidates))


def _expected_candidate_keys(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            _candidate_key(int(hidden_size), float(learning_rate), float(reservoir_scale))
            for hidden_size in config["recurrent_control"]["hidden_sizes"]
            for learning_rate in config["recurrent_control"]["learning_rates"]
            for reservoir_scale in config["recurrent_control"]["reservoir_scales"]
        )
    )


def replay_episode_actions(
    config: Mapping[str, Any],
    seed: int,
    split: str,
    episode_index: int,
    actions: Sequence[int],
) -> dict[str, Any]:
    world = PartialChangePointWorld(config, seed, split, episode_index)
    total_return = 0.0
    success = 0
    niche_values: MutableMapping[str, list[float]] = defaultdict(list)
    for action in actions:
        transition = world.step(int(action))
        total_return += transition.visible.feedback.reward
        success = max(success, int(transition.visible.feedback.reached_goal))
        if transition.niche_label is not None:
            niche_values[transition.niche_label].append(
                evaluator_step_value(transition, float(config["world"]["progress_weight"]))
            )
    _require(world.terminal, "stored action sequence does not reach the deterministic horizon")
    return {
        "total_return": total_return,
        "success": success,
        "utility": _episode_utility(success, total_return, config),
        "niche_values": {kind: statistics.fmean(values) for kind, values in niche_values.items()},
    }


def validate_arm_summary(
    summary: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    seed: int | None = None,
    split: str | None = None,
    expected_episode_count: int | None = None,
    expected_arm: str | None = None,
) -> None:
    arm_name = str(summary.get("arm", ""))
    allowed_keys = set(ARM_SUMMARY_KEYS)
    if arm_name == "equal_budget_recurrent" or arm_name.startswith("tune_recurrent_"):
        allowed_keys.update(RECURRENT_ARM_EXTRA_KEYS)
    _require_exact_keys(summary, allowed_keys, f"arm summary {arm_name!r}")
    episodes = summary["per_episode"]
    count = len(episodes)
    steps = int(summary["steps"])
    _require(int(summary["episode_count"]) == count, "arm episode-count mismatch")
    if expected_episode_count is not None:
        _require(count == expected_episode_count, "arm preregistered episode-count mismatch")
    if expected_arm is not None:
        _require(summary["arm"] == expected_arm, "arm identity mismatch")
    _require(count == 0 or steps % count == 0, "arm step-count mismatch")
    _require(sum(len(episode["actions"]) for episode in episodes) == steps, "arm action-count mismatch")
    _require(
        all(episode["actions_sha256"] == canonical_sha256(episode["actions"]) for episode in episodes),
        "arm action hash mismatch",
    )
    total_work = sum(int(episode["work_units"]) for episode in episodes)
    component_totals = {
        name: sum(int(episode["work_components"][name]) for episode in episodes)
        for name in dataclasses.asdict(AbstractWork())
    }
    total_bytes = sum(int(episode["message_bytes"]) for episode in episodes)

    def mean(key: str) -> float:
        return statistics.fmean(float(episode[key]) for episode in episodes) if episodes else 0.0

    _require(float(summary["mean_utility"]) == mean("utility"), "arm utility aggregate mismatch")
    _require(float(summary["mean_return"]) == mean("total_return"), "arm return aggregate mismatch")
    _require(float(summary["success_rate"]) == mean("success"), "arm success aggregate mismatch")
    _require(int(summary["total_abstract_work"]) == total_work, "arm work aggregate mismatch")
    _require(summary["work_component_totals"] == component_totals, "arm work-component aggregate mismatch")
    _require(
        float(summary["abstract_work_per_step"]) == (total_work / steps if steps else 0.0),
        "arm work-rate mismatch",
    )
    _require(int(summary["total_message_bytes"]) == total_bytes, "arm byte aggregate mismatch")
    _require(
        float(summary["message_bytes_per_step"]) == (total_bytes / steps if steps else 0.0),
        "arm byte-rate mismatch",
    )
    quartile = max(1, count // 4) if count else 0
    first = episodes[:quartile]
    last = episodes[-quartile:] if quartile else []
    first_mean = statistics.fmean(float(episode["utility"]) for episode in first) if first else 0.0
    last_mean = statistics.fmean(float(episode["utility"]) for episode in last) if last else 0.0
    _require(
        summary["sample_efficiency"]
        == {
            "first_quartile_mean_utility": first_mean,
            "last_quartile_mean_utility": last_mean,
            "within_run_learning_gain": last_mean - first_mean,
        },
        "arm sample-efficiency aggregate mismatch",
    )
    direct = summary.get("direct_effects")
    if direct is not None:
        for series in list(direct["metrics"].values()) + [direct["restoration"]]:
            validate_compact_evidence(series)
    if config is not None:
        if seed is None or split is None:
            raise ValueError("semantic replay context incomplete")
        for episode in episodes:
            _require(
                int(episode["work_units"])
                == sum(
                    int(episode["work_components"][name]) * int(config["abstract_work"]["weights"][name])
                    for name in dataclasses.asdict(AbstractWork())
                ),
                "episode work/component mismatch",
            )
        _require(
            summary["accounting_sensitivity"]
            == accounting_sensitivity(
                component_totals,
                config["abstract_work"]["weights"],
                config["abstract_work"]["sensitivity_factors"],
            ),
            "arm accounting-sensitivity mismatch",
        )
        for episode_index, episode in enumerate(episodes):
            replayed = replay_episode_actions(config, seed, split, episode_index, episode["actions"])
            for key in ("total_return", "success", "utility", "niche_values"):
                _require(episode[key] == replayed[key], f"semantic replay {key} mismatch")


def validate_gate_row(row: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _require_exact_keys(row, GATE_ROW_KEYS, "gate row")
    _require(row.get("schema") == "mop-edcm1-gate-row/v3", "gate row schema mismatch")
    seed = int(row["seed"])
    expected_counts = {
        "tune": int(config["splits"]["tune_episodes"]),
        "gate": int(config["splits"]["gate_episodes"]),
    }
    for split_name in ("tune", "gate"):
        _require(set(row[split_name]) == set(PROPOSER_ORDER), f"{split_name} proposer grid mismatch")
        for kind in PROPOSER_ORDER:
            validate_arm_summary(
                row[split_name][kind],
                config,
                seed,
                split_name,
                expected_counts[split_name],
                f"{split_name}_{kind}",
            )
    validate_arm_summary(
        row["tune_event_reference"],
        config,
        seed,
        "tune",
        expected_counts["tune"],
        "tune_event_budget_reference",
    )
    _require(
        int(row["recurrent_tuning_budget"]) == int(row["tune_event_reference"]["total_abstract_work"]),
        "recurrent tuning budget is not the measured tune event budget",
    )
    gate_episodes = int(config["splits"]["gate_episodes"])
    utilities = {kind: _utilities(row["gate"][kind]) for kind in PROPOSER_ORDER}
    best_kind = max(PROPOSER_ORDER, key=lambda kind: (row["gate"][kind]["mean_utility"], kind))
    oracle = [max(utilities[kind][index] for kind in PROPOSER_ORDER) for index in range(gate_episodes)]
    _require(row["oracle_values"] == oracle, "gate oracle-vector mismatch")
    _require(
        float(row["oracle_headroom"])
        == statistics.fmean(oracle) - float(row["gate"][best_kind]["mean_utility"]),
        "gate oracle-headroom mismatch",
    )
    wins = {kind: 0 for kind in PROPOSER_ORDER}
    for index in range(gate_episodes):
        ranked = sorted(((utilities[kind][index], kind) for kind in PROPOSER_ORDER), reverse=True)
        if ranked[0][0] > ranked[1][0]:
            wins[ranked[0][1]] += 1
    _require(row["unique_win_counts"] == wins, "gate unique-win mismatch")
    _require(
        row["unique_win_rates"] == {kind: wins[kind] / gate_episodes for kind in PROPOSER_ORDER},
        "gate unique-win-rate mismatch",
    )
    expected_niches: dict[str, list[float]] = {}
    for kind in PROPOSER_ORDER:
        own = _niche_values(row["gate"][kind], kind)
        others = {other: _niche_values(row["gate"][other], kind) for other in PROPOSER_ORDER if other != kind}
        expected_niches[kind] = [
            _required_niche_value(own[index])
            - max(_required_niche_value(others[other][index]) for other in others)
            for index in range(gate_episodes)
            if own[index] is not None and all(others[other][index] is not None for other in others)
        ]
    _require(row["niche_advantages"] == expected_niches, "gate niche-vector mismatch")
    _require(row["best_gate_kind"] == best_kind, "gate best-kind mismatch")
    _require(
        float(row["best_gate_success_rate"]) == float(row["gate"][best_kind]["success_rate"]),
        "gate ceiling metric mismatch",
    )
    verifier = row["verifier"]
    validate_compact_evidence(verifier["disagreement_evidence"])
    validate_compact_evidence(verifier["agreement_evidence"])
    _require(
        float(verifier["disagreement_gain"]) == float(verifier["disagreement_evidence"]["mean"]),
        "verifier disagreement mean mismatch",
    )
    _require(
        float(verifier["agreement_absolute_effect"]) == float(verifier["agreement_evidence"]["mean"]),
        "verifier agreement mean mismatch",
    )
    candidate_keys = _canonical_candidate_keys(row["recurrent_candidates"])
    _require(candidate_keys == _expected_candidate_keys(config), "recurrent candidate grid/config mismatch")
    for key in candidate_keys:
        candidate = row["recurrent_candidates"][key]
        validate_arm_summary(
            candidate,
            config,
            seed,
            "tune",
            expected_counts["tune"],
            f"tune_recurrent_{key}",
        )
        _require(
            candidate["recurrent_hyperparameters"] == _candidate_payload(key),
            "recurrent candidate hyperparameter/key mismatch",
        )
        _require(
            int(candidate["recurrent_budget"]) == int(row["recurrent_tuning_budget"]),
            "recurrent candidate budget mismatch",
        )
        residual = int(candidate["round_activation_counts"].get("budget:residual", 0))
        _require(
            0 <= residual <= int(row["recurrent_tuning_budget"]),
            "recurrent candidate residual outside budget",
        )
        _require(
            int(candidate["total_abstract_work"]) + residual == int(row["recurrent_tuning_budget"]),
            "recurrent candidate ledger/budget mismatch",
        )


def run_verifier_gate(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    episodes = int(config["splits"]["gate_episodes"])
    controller = CoalitionController(config, "always_on")
    disagreement: list[float] = []
    agreement_abs: list[float] = []
    progress_weight = float(config["world"]["progress_weight"])
    for episode in range(episodes):
        world = PartialChangePointWorld(config, seed, "gate", episode)
        controller.reset_episode()
        while not world.terminal:
            prepared = controller.prepare(world.observe())
            clean = controller.resolve(prepared, "clean")
            lesion = _resolve_prepared(
                config,
                copy.deepcopy(controller.verifier),
                prepared,
                "verifier_lesion",
                None,
            )
            difference = _counterfactual_value(world, clean.action, progress_weight) - _counterfactual_value(
                world,
                lesion.action,
                progress_weight,
            )
            if len({proposal.proposed_action for proposal in prepared.proposals}) >= 2:
                disagreement.append(difference)
            else:
                agreement_abs.append(abs(difference))
            transition = world.step(clean.action)
            controller.update(transition.visible)
    return {
        "disagreement_gain": statistics.fmean(disagreement) if disagreement else 0.0,
        "agreement_absolute_effect": statistics.fmean(agreement_abs) if agreement_abs else 0.0,
        "disagreement_evidence": compact_evidence(disagreement),
        "agreement_evidence": compact_evidence(agreement_abs),
        "hard_dispatch_violations": controller.hard_dispatch_violations,
    }


def validate_gate_verifier_evidence(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    regenerated = run_verifier_gate(config, int(row["seed"]))
    _require(
        regenerated == row["verifier"],
        "regenerated gate verifier evidence mismatch",
    )


def run_gate_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    horizon = int(config["world"]["horizon"])
    tune_episodes = int(config["splits"]["tune_episodes"])
    gate_episodes = int(config["splits"]["gate_episodes"])
    tune: dict[str, dict[str, Any]] = {}
    gate: dict[str, dict[str, Any]] = {}
    for kind in PROPOSER_ORDER:
        tune[kind] = run_coalition_arm(
            config,
            seed=seed,
            split="tune",
            episodes=tune_episodes,
            name=f"tune_{kind}",
            mode="tuned_best_single",
            single_kind=kind,
        ).summary(horizon)
        gate[kind] = run_coalition_arm(
            config,
            seed=seed,
            split="gate",
            episodes=gate_episodes,
            name=f"gate_{kind}",
            mode="tuned_best_single",
            single_kind=kind,
        ).summary(horizon)
    utility_vectors = {kind: _utilities(gate[kind]) for kind in PROPOSER_ORDER}
    best_gate_kind = max(PROPOSER_ORDER, key=lambda kind: (gate[kind]["mean_utility"], kind))
    oracle = [max(utility_vectors[kind][index] for kind in PROPOSER_ORDER) for index in range(gate_episodes)]
    win_counts = {kind: 0 for kind in PROPOSER_ORDER}
    for index in range(gate_episodes):
        ranked = sorted(
            ((utility_vectors[kind][index], kind) for kind in PROPOSER_ORDER),
            reverse=True,
        )
        if ranked[0][0] > ranked[1][0]:
            win_counts[ranked[0][1]] += 1
    niche_advantages: dict[str, list[float]] = {}
    for kind in PROPOSER_ORDER:
        own = _niche_values(gate[kind], kind)
        others = {other: _niche_values(gate[other], kind) for other in PROPOSER_ORDER if other != kind}
        niche_advantages[kind] = [
            _required_niche_value(own[index])
            - max(_required_niche_value(others[other][index]) for other in others)
            for index in range(gate_episodes)
            if own[index] is not None and all(others[other][index] is not None for other in others)
        ]
    tune_event_reference = run_coalition_arm(
        config,
        seed=seed,
        split="tune",
        episodes=tune_episodes,
        name="tune_event_budget_reference",
        mode="event_triggered",
    ).summary(horizon)
    recurrent_candidates: dict[str, dict[str, Any]] = {}
    tuning_budget = int(tune_event_reference["total_abstract_work"])
    for hidden_size in config["recurrent_control"]["hidden_sizes"]:
        for learning_rate in config["recurrent_control"]["learning_rates"]:
            for reservoir_scale in config["recurrent_control"]["reservoir_scales"]:
                key = _candidate_key(int(hidden_size), float(learning_rate), float(reservoir_scale))
                result = run_recurrent_arm(
                    config,
                    seed=seed,
                    split="tune",
                    episodes=tune_episodes,
                    total_budget=tuning_budget,
                    hyperparameters=_candidate_payload(key),
                    name=f"tune_recurrent_{key}",
                ).summary(horizon)
                result["recurrent_hyperparameters"] = _candidate_payload(key)
                result["recurrent_budget"] = tuning_budget
                recurrent_candidates[key] = result
    verifier = run_verifier_gate(config, seed)
    return {
        "schema": "mop-edcm1-gate-row/v3",
        "seed": seed,
        "tune": tune,
        "gate": gate,
        "oracle_headroom": statistics.fmean(oracle) - float(gate[best_gate_kind]["mean_utility"]),
        "oracle_values": oracle,
        "unique_win_counts": win_counts,
        "unique_win_rates": {kind: win_counts[kind] / gate_episodes for kind in PROPOSER_ORDER},
        "niche_advantages": niche_advantages,
        "best_gate_kind": best_gate_kind,
        "best_gate_success_rate": float(gate[best_gate_kind]["success_rate"]),
        "tune_event_reference": tune_event_reference,
        "recurrent_tuning_budget": tuning_budget,
        "recurrent_candidates": recurrent_candidates,
        "verifier": verifier,
    }


def aggregate_gate(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    seeds = list(config["seeds"])
    for row in rows:
        validate_gate_row(row, config)
    if [int(row["seed"]) for row in rows] != seeds:
        return {"status": "incomplete", "passed": False}
    critical = float(config["evaluation"]["t_critical_95"])
    settings = config["complementarity_gate"]
    oracle_ci = mean_ci([float(row["oracle_headroom"]) for row in rows], critical)
    niche_ci = {
        kind: mean_ci(
            [statistics.fmean(row["niche_advantages"][kind]) for row in rows],
            critical,
        )
        for kind in PROPOSER_ORDER
    }
    verifier_disagreement_ci = mean_ci(
        [float(row["verifier"]["disagreement_gain"]) for row in rows],
        critical,
    )
    verifier_agreement_ci = mean_ci(
        [float(row["verifier"]["agreement_absolute_effect"]) for row in rows],
        critical,
    )
    tune_means = {
        kind: statistics.fmean(float(row["tune"][kind]["mean_utility"]) for row in rows)
        for kind in PROPOSER_ORDER
    }
    selected_kind = max(PROPOSER_ORDER, key=lambda kind: (tune_means[kind], kind))
    candidate_keys = _canonical_candidate_keys(rows[0]["recurrent_candidates"])
    _require(
        all(_canonical_candidate_keys(row["recurrent_candidates"]) == candidate_keys for row in rows),
        "recurrent tuning candidate grid mismatch",
    )
    recurrent_means = {
        key: statistics.fmean(float(row["recurrent_candidates"][key]["mean_utility"]) for row in rows)
        for key in candidate_keys
    }
    selected_recurrent = max(candidate_keys, key=lambda key: (recurrent_means[key], key))
    checks = {
        "oracle_headroom": oracle_ci["lower"] >= float(settings["min_oracle_headroom"]),
        "every_proposer_has_enough_unique_wins": all(
            int(row["unique_win_counts"][kind]) >= int(settings["min_unique_wins"])
            and float(row["unique_win_rates"][kind]) >= float(settings["min_unique_win_rate"])
            for row in rows
            for kind in PROPOSER_ORDER
        ),
        "every_proposer_has_positive_niche_advantage": all(
            niche_ci[kind]["lower"] > float(settings["min_niche_advantage"]) for kind in PROPOSER_ORDER
        ),
        "off_ceiling": all(
            float(row["best_gate_success_rate"]) <= float(settings["max_best_single_success_rate"])
            for row in rows
        ),
        "verifier_helps_disagreement": verifier_disagreement_ci["lower"]
        > float(settings["min_verifier_disagreement_gain"]),
        "verifier_abstains_without_material_agreement_effect": verifier_agreement_ci["upper"]
        <= float(settings["max_verifier_agreement_effect"]),
        "hard_dispatch": all(
            int(row["verifier"]["hard_dispatch_violations"]) == 0
            and all(
                int(row[split_name][kind]["hard_dispatch_violations"]) == 0
                for split_name in ("tune", "gate")
                for kind in PROPOSER_ORDER
            )
            for row in rows
        ),
    }
    return {
        "status": "complete",
        "passed": all(checks.values()),
        "checks": checks,
        "oracle_headroom_95": oracle_ci,
        "niche_advantage_95": niche_ci,
        "verifier_disagreement_gain_95": verifier_disagreement_ci,
        "verifier_agreement_effect_95": verifier_agreement_ci,
        "selected_best_single": selected_kind,
        "selected_recurrent": _candidate_payload(selected_recurrent),
        "tune_means": tune_means,
        "recurrent_tune_means": recurrent_means,
    }


def _pareto_front(arms: Mapping[str, Mapping[str, Any]], always_work: float) -> list[str]:
    front: list[str] = []
    for name, candidate in arms.items():
        utility = float(candidate["mean_utility"])
        cost = float(candidate["abstract_work_per_step"]) / always_work
        dominated = False
        for other_name, other in arms.items():
            if other_name == name:
                continue
            other_utility = float(other["mean_utility"])
            other_cost = float(other["abstract_work_per_step"]) / always_work
            if (
                other_utility >= utility
                and other_cost <= cost
                and (other_utility > utility or other_cost < cost)
            ):
                dominated = True
                break
        if not dominated:
            front.append(name)
    return sorted(front)


def pareto_report(arms: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    always_work = float(arms["always_on"]["abstract_work_per_step"])
    reference_cost = float(config["evaluation"]["pareto_reference_relative_work"])
    reference_utility = float(config["evaluation"]["pareto_reference_utility"])

    def hypervolume(selected: Mapping[str, Mapping[str, Any]]) -> float:
        points = sorted(
            (
                float(selected[name]["abstract_work_per_step"]) / always_work,
                float(selected[name]["mean_utility"]),
            )
            for name in _pareto_front(selected, always_work)
        )
        area = 0.0
        best = reference_utility
        for index, (cost, utility) in enumerate(points):
            best = max(best, utility)
            next_cost = points[index + 1][0] if index + 1 < len(points) else reference_cost
            area += max(0.0, min(reference_cost, next_cost) - cost) * max(0.0, best - reference_utility)
        return area

    total = hypervolume(arms)
    without_event = {name: arm for name, arm in arms.items() if name != "event_triggered"}
    return {
        "axes": {"maximize": "mean_utility", "minimize": "relative_abstract_work"},
        "fixed_reference": {"relative_work": reference_cost, "utility": reference_utility},
        "front": _pareto_front(arms, always_work),
        "hypervolume": total,
        "event_hypervolume_contribution": total - hypervolume(without_event),
        "event_nondominated": "event_triggered" in _pareto_front(arms, always_work),
    }


def run_heldout_seed(
    config: Mapping[str, Any],
    seed: int,
    selected_kind: str,
    recurrent_hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    episodes = int(config["splits"]["heldout_episodes"])
    horizon = int(config["world"]["horizon"])
    event = run_coalition_arm(
        config,
        seed=seed,
        split="heldout",
        episodes=episodes,
        name="event_triggered",
        mode="event_triggered",
        capture_interventions=True,
    )
    event_summary = event.summary(horizon)
    periodic = [periodic_round_matched_schedule(schedule) for schedule in event.schedules]
    shuffled = [
        shuffled_round_matched_schedule(schedule, _stable_int(seed, "round-shuffle", episode))
        for episode, schedule in enumerate(event.schedules)
    ]
    coalition_shuffled = [
        shuffled_coalition_matched_schedule(
            schedule,
            _stable_int(seed, "coalition-shuffle", episode),
        )
        for episode, schedule in enumerate(event.schedules)
    ]
    replay = [
        [dataclasses.replace(record, scheduler_tag="replay") for record in schedule]
        for schedule in event.schedules
    ]
    arms: dict[str, dict[str, Any]] = {"event_triggered": event_summary}
    arms["always_on"] = run_coalition_arm(
        config,
        seed=seed,
        split="heldout",
        episodes=episodes,
        name="always_on",
        mode="always_on",
    ).summary(horizon)
    arms["tuned_best_single"] = run_coalition_arm(
        config,
        seed=seed,
        split="heldout",
        episodes=episodes,
        name="tuned_best_single",
        mode="tuned_best_single",
        single_kind=selected_kind,
    ).summary(horizon)
    arms["periodic_round_matched"] = run_coalition_arm(
        config,
        seed=seed,
        split="heldout",
        episodes=episodes,
        name="periodic_round_matched",
        mode="fixed",
        fixed_schedules=periodic,
    ).summary(horizon)
    arms["shuffled_round_matched"] = run_coalition_arm(
        config,
        seed=seed,
        split="heldout",
        episodes=episodes,
        name="shuffled_round_matched",
        mode="fixed",
        fixed_schedules=shuffled,
    ).summary(horizon)
    arms["shuffled_coalition_matched"] = run_coalition_arm(
        config,
        seed=seed,
        split="heldout",
        episodes=episodes,
        name="shuffled_coalition_matched",
        mode="coalition_fixed",
        fixed_schedules=coalition_shuffled,
    ).summary(horizon)
    arms["homogeneous_matched"] = run_homogeneous_arm(
        config,
        seed=seed,
        split="heldout",
        episodes=episodes,
        kind=selected_kind,
        reference_schedules=event.schedules,
    ).summary(horizon)
    recurrent_accumulator = run_recurrent_arm(
        config,
        seed=seed,
        split="heldout",
        episodes=episodes,
        total_budget=int(event_summary["total_abstract_work"]),
        hyperparameters=recurrent_hyperparameters,
    )
    arms["equal_budget_recurrent"] = recurrent_accumulator.summary(horizon)
    arms["equal_budget_recurrent"]["recurrent_hyperparameters"] = dict(recurrent_hyperparameters)
    arms["equal_budget_recurrent"]["recurrent_budget"] = int(event_summary["total_abstract_work"])
    replay_summary = run_coalition_arm(
        config,
        seed=seed,
        split="heldout",
        episodes=episodes,
        name="clean_fixed_replay",
        mode="fixed",
        fixed_schedules=replay,
    ).summary(horizon)
    event_actions = [episode["actions_sha256"] for episode in event_summary["per_episode"]]
    replay_actions = [episode["actions_sha256"] for episode in replay_summary["per_episode"]]
    event_round_counts = round_activation_counts(
        [record for schedule in event.schedules for record in schedule]
    )
    periodic_counts = round_activation_counts([record for schedule in periodic for record in schedule])
    shuffled_counts = round_activation_counts([record for schedule in shuffled for record in schedule])
    homogeneous_initial_calls = sum(
        count
        for key, count in arms["homogeneous_matched"]["round_activation_counts"].items()
        if key.startswith("initial:")
    )
    homogeneous_verifier_calls = int(
        arms["homogeneous_matched"]["round_activation_counts"].get(f"extra:{VERIFIER_ID}", 0)
    )
    event_initial_calls = sum(
        count for key, count in event_round_counts.items() if key.startswith("initial:")
    )
    event_verifier_calls = int(event_round_counts[f"extra:{VERIFIER_ID}"])
    coalition_records_exact = all(
        Counter((record.initial, record.extra_round) for record in source)
        == Counter((record.initial, record.extra_round) for record in shuffled_schedule)
        for source, shuffled_schedule in zip(event.schedules, coalition_shuffled, strict=True)
    )
    residual = int(arms["equal_budget_recurrent"]["round_activation_counts"].get("budget:residual", 0))
    event_work = float(event_summary["abstract_work_per_step"])
    direct = event_summary["direct_effects"]

    def direct_mean(name: str) -> float:
        return float(direct["metrics"].get(name, {"mean": 0.0})["mean"])

    mechanics = {
        "utility_loss_vs_always": float(arms["always_on"]["mean_utility"])
        - float(event_summary["mean_utility"]),
        "work_saving_vs_always": 1.0 - event_work / float(arms["always_on"]["abstract_work_per_step"]),
        "conservative_work_saving_vs_always": 1.0
        - float(event_summary["accounting_sensitivity"]["scenario_max"])
        / max(1.0, float(arms["always_on"]["accounting_sensitivity"]["scenario_min"])),
        "utility_margin_vs_single": float(event_summary["mean_utility"])
        - float(arms["tuned_best_single"]["mean_utility"]),
        "utility_margin_vs_recurrent": float(event_summary["mean_utility"])
        - float(arms["equal_budget_recurrent"]["mean_utility"]),
        "utility_margin_vs_periodic": float(event_summary["mean_utility"])
        - float(arms["periodic_round_matched"]["mean_utility"]),
        "utility_margin_vs_shuffled": float(event_summary["mean_utility"])
        - float(arms["shuffled_round_matched"]["mean_utility"]),
        "utility_margin_vs_coalition_shuffled": float(event_summary["mean_utility"])
        - float(arms["shuffled_coalition_matched"]["mean_utility"]),
        "utility_margin_vs_homogeneous": float(event_summary["mean_utility"])
        - float(arms["homogeneous_matched"]["mean_utility"]),
        "clean_minus_no_message": direct_mean("clean_minus_no_message"),
        "clean_minus_reactive_lesion": direct_mean("clean_minus_reactive_lesion"),
        "clean_minus_episodic_lesion": direct_mean("clean_minus_episodic_lesion"),
        "clean_minus_planner_lesion": direct_mean("clean_minus_planner_lesion"),
        "clean_minus_two_tick_delay": direct_mean("clean_minus_two_tick_delay"),
        "two_tick_delay_minus_two_tick_lesion": direct_mean("two_tick_delay_minus_two_tick_lesion"),
        "clean_minus_wrong_planner": direct_mean("clean_minus_wrong_planner"),
        "clean_minus_verifier_lesion": direct_mean("clean_minus_verifier_lesion"),
        "restoration_gain": float(direct["restoration"]["mean"]),
        "reactive_lesion_niche_selectivity": float(direct_mean("clean_minus_reactive_lesion_matching_niche"))
        - direct_mean("clean_minus_reactive_lesion_outside_niche"),
        "episodic_lesion_niche_selectivity": float(direct_mean("clean_minus_episodic_lesion_matching_niche"))
        - direct_mean("clean_minus_episodic_lesion_outside_niche"),
        "planner_lesion_niche_selectivity": float(direct_mean("clean_minus_planner_lesion_matching_niche"))
        - direct_mean("clean_minus_planner_lesion_outside_niche"),
        "change_vs_noise_activation_gap": float(event_summary["change_expensive_rate"])
        - float(event_summary["noise_false_expensive_rate"]),
        "recurrent_budget_residual_ratio": residual / max(1.0, float(event_summary["total_abstract_work"])),
    }
    invariants = {
        "periodic_round_counts_exact": periodic_counts == event_round_counts,
        "shuffled_round_counts_exact": shuffled_counts == event_round_counts,
        "shuffled_coalition_records_exact": coalition_records_exact,
        "homogeneous_initial_calls_exact": homogeneous_initial_calls == event_initial_calls,
        "homogeneous_verifier_calls_exact": homogeneous_verifier_calls == event_verifier_calls,
        "clean_fixed_replay_identical": event_actions == replay_actions
        and float(event_summary["mean_utility"]) == float(replay_summary["mean_utility"]),
        "hard_dispatch_zero": all(int(arm["hard_dispatch_violations"]) == 0 for arm in arms.values()),
        "noise_pair_invariance": int(event_summary["noise_activation_pair_differences"]) == 0,
        "common_state_forks_identical": int(direct["fork_identity_violations"]) == 0,
        "channel_delay_common_origin_valid": int(direct["channel_delay_origin_violations"]) == 0,
        "intervention_episode_cap_respected": int(event_summary["noise_pair_opportunities"])
        == int(config["splits"]["intervention_episodes"]) * len(config["world"]["noise_ticks"]),
        "all_direct_effect_counts_capped": _direct_effect_counts_within_cap(direct, config),
        "recurrent_has_no_padding_and_within_budget": mechanics["recurrent_budget_residual_ratio"]
        <= float(config["recurrent_control"]["max_budget_residual_ratio"]),
    }
    return {
        "schema": "mop-edcm1-heldout-row/v3",
        "seed": seed,
        "selected_best_single": selected_kind,
        "selected_recurrent": dict(recurrent_hyperparameters),
        "arms": arms,
        "clean_fixed_replay": replay_summary,
        "mechanics": mechanics,
        "invariants": invariants,
        "pareto": pareto_report(arms, config),
    }


def _recompute_row_mechanics(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    arms = row["arms"]
    event = arms["event_triggered"]
    direct = event["direct_effects"]
    residual = int(arms["equal_budget_recurrent"]["round_activation_counts"].get("budget:residual", 0))
    event_work = float(event["abstract_work_per_step"])

    def direct_mean(name: str) -> float:
        return float(direct["metrics"].get(name, {"mean": 0.0})["mean"])

    return {
        "utility_loss_vs_always": float(arms["always_on"]["mean_utility"]) - float(event["mean_utility"]),
        "work_saving_vs_always": 1.0 - event_work / float(arms["always_on"]["abstract_work_per_step"]),
        "conservative_work_saving_vs_always": 1.0
        - float(event["accounting_sensitivity"]["scenario_max"])
        / max(1.0, float(arms["always_on"]["accounting_sensitivity"]["scenario_min"])),
        "utility_margin_vs_single": float(event["mean_utility"])
        - float(arms["tuned_best_single"]["mean_utility"]),
        "utility_margin_vs_recurrent": float(event["mean_utility"])
        - float(arms["equal_budget_recurrent"]["mean_utility"]),
        "utility_margin_vs_periodic": float(event["mean_utility"])
        - float(arms["periodic_round_matched"]["mean_utility"]),
        "utility_margin_vs_shuffled": float(event["mean_utility"])
        - float(arms["shuffled_round_matched"]["mean_utility"]),
        "utility_margin_vs_coalition_shuffled": float(event["mean_utility"])
        - float(arms["shuffled_coalition_matched"]["mean_utility"]),
        "utility_margin_vs_homogeneous": float(event["mean_utility"])
        - float(arms["homogeneous_matched"]["mean_utility"]),
        "clean_minus_no_message": direct_mean("clean_minus_no_message"),
        "clean_minus_reactive_lesion": direct_mean("clean_minus_reactive_lesion"),
        "clean_minus_episodic_lesion": direct_mean("clean_minus_episodic_lesion"),
        "clean_minus_planner_lesion": direct_mean("clean_minus_planner_lesion"),
        "clean_minus_two_tick_delay": direct_mean("clean_minus_two_tick_delay"),
        "two_tick_delay_minus_two_tick_lesion": direct_mean("two_tick_delay_minus_two_tick_lesion"),
        "clean_minus_wrong_planner": direct_mean("clean_minus_wrong_planner"),
        "clean_minus_verifier_lesion": direct_mean("clean_minus_verifier_lesion"),
        "restoration_gain": float(direct["restoration"]["mean"]),
        "reactive_lesion_niche_selectivity": float(direct_mean("clean_minus_reactive_lesion_matching_niche"))
        - direct_mean("clean_minus_reactive_lesion_outside_niche"),
        "episodic_lesion_niche_selectivity": float(direct_mean("clean_minus_episodic_lesion_matching_niche"))
        - direct_mean("clean_minus_episodic_lesion_outside_niche"),
        "planner_lesion_niche_selectivity": float(direct_mean("clean_minus_planner_lesion_matching_niche"))
        - direct_mean("clean_minus_planner_lesion_outside_niche"),
        "change_vs_noise_activation_gap": float(event["change_expensive_rate"])
        - float(event["noise_false_expensive_rate"]),
        "recurrent_budget_residual_ratio": residual / max(1.0, float(event["total_abstract_work"])),
    }


def _direct_effect_counts_within_cap(
    direct: Mapping[str, Any],
    config: Mapping[str, Any],
) -> bool:
    episode_cap = int(config["splits"]["intervention_episodes"])
    decision_cap = episode_cap * int(config["world"]["horizon"])
    return (
        int(direct["restoration"]["count"]) == episode_cap
        and 0 <= int(direct["fork_identity_violations"]) <= episode_cap
        and 0 <= int(direct["channel_delay_origin_violations"]) <= 2 * decision_cap
        and all(0 <= int(series["count"]) <= decision_cap for series in direct["metrics"].values())
    )


def _recompute_row_invariants(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, bool]:
    arms = row["arms"]
    event = arms["event_triggered"]
    event_counts = event["round_activation_counts"]
    event_initial_calls = sum(int(value) for key, value in event_counts.items() if key.startswith("initial:"))
    event_verifier_calls = int(event_counts.get(f"extra:{VERIFIER_ID}", 0))
    homogeneous_initial_calls = sum(
        int(value)
        for key, value in arms["homogeneous_matched"]["round_activation_counts"].items()
        if key.startswith("initial:")
    )
    homogeneous_verifier_calls = int(
        arms["homogeneous_matched"]["round_activation_counts"].get(f"extra:{VERIFIER_ID}", 0)
    )
    event_actions = [episode["actions_sha256"] for episode in event["per_episode"]]
    replay_actions = [episode["actions_sha256"] for episode in row["clean_fixed_replay"]["per_episode"]]
    mechanics = _recompute_row_mechanics(row, config)
    direct = event["direct_effects"]
    return {
        "periodic_round_counts_exact": arms["periodic_round_matched"]["round_activation_counts"]
        == event_counts,
        "shuffled_round_counts_exact": arms["shuffled_round_matched"]["round_activation_counts"]
        == event_counts,
        "shuffled_coalition_records_exact": (
            arms["shuffled_coalition_matched"]["activation_record_multiset_sha256_per_episode"]
            == event["activation_record_multiset_sha256_per_episode"]
        ),
        "homogeneous_initial_calls_exact": homogeneous_initial_calls == event_initial_calls,
        "homogeneous_verifier_calls_exact": homogeneous_verifier_calls == event_verifier_calls,
        "clean_fixed_replay_identical": event_actions == replay_actions
        and float(event["mean_utility"]) == float(row["clean_fixed_replay"]["mean_utility"]),
        "hard_dispatch_zero": all(int(arm["hard_dispatch_violations"]) == 0 for arm in arms.values()),
        "noise_pair_invariance": int(event["noise_activation_pair_differences"]) == 0,
        "common_state_forks_identical": int(direct["fork_identity_violations"]) == 0,
        "channel_delay_common_origin_valid": int(direct["channel_delay_origin_violations"]) == 0,
        "intervention_episode_cap_respected": int(event["noise_pair_opportunities"])
        == int(config["splits"]["intervention_episodes"]) * len(config["world"]["noise_ticks"]),
        "all_direct_effect_counts_capped": _direct_effect_counts_within_cap(direct, config),
        "recurrent_has_no_padding_and_within_budget": mechanics["recurrent_budget_residual_ratio"]
        <= float(config["recurrent_control"]["max_budget_residual_ratio"]),
    }


def validate_heldout_row(row: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _require_exact_keys(row, HELDOUT_ROW_KEYS, "heldout row")
    _require(row.get("schema") == "mop-edcm1-heldout-row/v3", "heldout row schema mismatch")
    seed = int(row["seed"])
    expected_count = int(config["splits"]["heldout_episodes"])
    _require(set(row["arms"]) == set(MAIN_ARMS), "heldout arm grid mismatch")
    for name, arm in row["arms"].items():
        validate_arm_summary(
            arm,
            config,
            seed,
            "heldout",
            expected_count,
            name,
        )
    validate_arm_summary(
        row["clean_fixed_replay"],
        config,
        seed,
        "heldout",
        expected_count,
        "clean_fixed_replay",
    )
    recurrent = row["arms"]["equal_budget_recurrent"]
    _require(
        recurrent["recurrent_hyperparameters"] == row["selected_recurrent"],
        "heldout recurrent hyperparameter mismatch",
    )
    _require(
        int(recurrent["recurrent_budget"]) == int(row["arms"]["event_triggered"]["total_abstract_work"]),
        "heldout recurrent/event budget mismatch",
    )
    recurrent_residual = int(recurrent["round_activation_counts"].get("budget:residual", 0))
    _require(
        0 <= recurrent_residual <= int(recurrent["recurrent_budget"]),
        "heldout recurrent residual outside budget",
    )
    _require(
        int(recurrent["total_abstract_work"]) + recurrent_residual == int(recurrent["recurrent_budget"]),
        "heldout recurrent ledger/budget mismatch",
    )
    _require(row["mechanics"] == _recompute_row_mechanics(row, config), "heldout mechanics mismatch")
    _require(row["invariants"] == _recompute_row_invariants(row, config), "heldout invariant mismatch")
    _require(row["pareto"] == pareto_report(row["arms"], config), "heldout Pareto mismatch")


def validate_intervention_evidence(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:

    episode_cap = int(config["splits"]["intervention_episodes"])
    regenerated = run_coalition_arm(
        config,
        seed=int(row["seed"]),
        split="heldout",
        episodes=episode_cap,
        name="event_triggered_intervention_verification",
        mode="event_triggered",
        capture_interventions=True,
    ).summary(int(config["world"]["horizon"]))
    stored = row["arms"]["event_triggered"]
    _require(
        regenerated["direct_effects"] == stored["direct_effects"],
        "regenerated intervention evidence mismatch",
    )
    _require(
        [episode["actions_sha256"] for episode in regenerated["per_episode"]]
        == [episode["actions_sha256"] for episode in stored["per_episode"][:episode_cap]],
        "intervention regeneration action-prefix mismatch",
    )


def aggregate_heldout(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    gate_result: Mapping[str, Any],
) -> dict[str, Any]:
    seeds = list(config["seeds"])
    if [int(row["seed"]) for row in rows] != seeds:
        return {"status": "incomplete", "verdict": "not_evaluated", "scientific_promotion": False}
    for row in rows:
        validate_heldout_row(row, config)
    critical = float(config["evaluation"]["t_critical_95"])
    names = tuple(rows[0]["mechanics"])
    intervals = {name: mean_ci([float(row["mechanics"][name]) for row in rows], critical) for name in names}
    hypervolume_ci = mean_ci(
        [float(row["pareto"]["event_hypervolume_contribution"]) for row in rows],
        critical,
    )
    criteria = config["criteria"]
    direct_counts_nonzero = all(
        all(
            int(row["arms"]["event_triggered"]["direct_effects"]["metrics"].get(name, {"count": 0})["count"])
            > 0
            for name in (
                "clean_minus_no_message",
                "clean_minus_reactive_lesion",
                "clean_minus_episodic_lesion",
                "clean_minus_planner_lesion",
                "clean_minus_two_tick_delay",
                "two_tick_delay_minus_two_tick_lesion",
                "clean_minus_wrong_planner",
                "clean_minus_verifier_lesion",
                "clean_minus_reactive_lesion_matching_niche",
                "clean_minus_reactive_lesion_outside_niche",
                "clean_minus_episodic_lesion_matching_niche",
                "clean_minus_episodic_lesion_outside_niche",
                "clean_minus_planner_lesion_matching_niche",
                "clean_minus_planner_lesion_outside_niche",
            )
        )
        and int(row["arms"]["event_triggered"]["direct_effects"]["restoration"]["count"]) > 0
        for row in rows
    )
    positive_fields = (
        "work_saving_vs_always",
        "conservative_work_saving_vs_always",
        "utility_margin_vs_single",
        "utility_margin_vs_recurrent",
        "utility_margin_vs_periodic",
        "utility_margin_vs_shuffled",
        "utility_margin_vs_coalition_shuffled",
        "utility_margin_vs_homogeneous",
        "clean_minus_no_message",
        "clean_minus_reactive_lesion",
        "clean_minus_episodic_lesion",
        "clean_minus_planner_lesion",
        "clean_minus_two_tick_delay",
        "two_tick_delay_minus_two_tick_lesion",
        "clean_minus_wrong_planner",
        "clean_minus_verifier_lesion",
        "restoration_gain",
        "reactive_lesion_niche_selectivity",
        "episodic_lesion_niche_selectivity",
        "planner_lesion_niche_selectivity",
        "change_vs_noise_activation_gap",
    )
    all_directions = {
        name: all(float(row["mechanics"][name]) > 0.0 for row in rows) for name in positive_fields
    }
    checks = {
        "gate_passed_before_heldout": bool(gate_result.get("passed")),
        "utility_loss_within_one_point": intervals["utility_loss_vs_always"]["upper"]
        <= float(criteria["max_utility_loss_vs_always_on"]),
        "at_least_twenty_five_percent_less_work": intervals["work_saving_vs_always"]["lower"]
        >= float(criteria["min_work_saving_vs_always_on"]),
        "work_saving_survives_accounting_perturbations": (
            intervals["conservative_work_saving_vs_always"]["lower"]
            >= float(criteria["min_conservative_work_saving"])
        ),
        "margin_vs_single": intervals["utility_margin_vs_single"]["lower"]
        >= float(criteria["min_utility_margin_vs_single"]),
        "margin_vs_recurrent": intervals["utility_margin_vs_recurrent"]["lower"]
        >= float(criteria["min_utility_margin_vs_recurrent"]),
        "periodic_worse": intervals["utility_margin_vs_periodic"]["lower"]
        > float(criteria["min_matched_control_margin"]),
        "shuffled_worse": intervals["utility_margin_vs_shuffled"]["lower"]
        > float(criteria["min_matched_control_margin"]),
        "coalition_shuffled_worse": intervals["utility_margin_vs_coalition_shuffled"]["lower"]
        > float(criteria["min_matched_control_margin"]),
        "homogeneous_worse": intervals["utility_margin_vs_homogeneous"]["lower"]
        > float(criteria["min_matched_control_margin"]),
        "no_message_degrades": intervals["clean_minus_no_message"]["lower"]
        >= float(criteria["min_direct_message_penalty"]),
        "specialist_link_lesions_degrade": all(
            intervals[name]["lower"] >= float(criteria["min_direct_message_penalty"])
            for name in (
                "clean_minus_reactive_lesion",
                "clean_minus_episodic_lesion",
                "clean_minus_planner_lesion",
            )
        ),
        "specialist_link_lesions_are_niche_selective": all(
            intervals[name]["lower"] > float(criteria["min_niche_lesion_selectivity"])
            for name in (
                "reactive_lesion_niche_selectivity",
                "episodic_lesion_niche_selectivity",
                "planner_lesion_niche_selectivity",
            )
        ),
        "two_tick_channel_delay_degrades_but_is_not_a_lesion": (
            intervals["clean_minus_two_tick_delay"]["lower"] >= float(criteria["min_direct_message_penalty"])
            and intervals["two_tick_delay_minus_two_tick_lesion"]["lower"]
            > float(criteria["min_delay_advantage_over_lesion"])
        ),
        "wrong_planner_message_degrades": intervals["clean_minus_wrong_planner"]["lower"]
        >= float(criteria["min_direct_message_penalty"]),
        "verifier_is_causally_useful": intervals["clean_minus_verifier_lesion"]["lower"]
        >= float(criteria["min_direct_message_penalty"]),
        "restoration_beats_continued_lesion_from_common_state": intervals["restoration_gain"]["lower"]
        >= float(criteria["min_restoration_gain"]),
        "noise_invariance": all(
            int(row["arms"]["event_triggered"]["noise_activation_pair_differences"]) == 0 for row in rows
        ),
        "change_selectivity": intervals["change_vs_noise_activation_gap"]["lower"]
        >= float(criteria["min_change_vs_noise_activation_gap"]),
        "event_pareto_nondominated": all(bool(row["pareto"]["event_nondominated"]) for row in rows),
        "positive_hypervolume_contribution": hypervolume_ci["lower"] > 0.0,
        "all_mechanical_invariants": all(all(row["invariants"].values()) for row in rows),
        "all_direct_interventions_sampled": direct_counts_nonzero,
        "loss_and_efficiency_hold_in_every_seed": all(
            float(row["mechanics"]["utility_loss_vs_always"])
            <= float(criteria["max_utility_loss_vs_always_on"])
            and float(row["mechanics"]["work_saving_vs_always"])
            >= float(criteria["min_work_saving_vs_always_on"])
            and float(row["mechanics"]["conservative_work_saving_vs_always"])
            >= float(criteria["min_conservative_work_saving"])
            for row in rows
        ),
        "all_seed_directions": all(all_directions.values()),
    }
    favorable = all(checks.values())
    return {
        "status": "complete",
        "paired_seed_intervals_95": intervals,
        "event_hypervolume_contribution_95": hypervolume_ci,
        "all_seed_directions": all_directions,
        "checks": checks,
        "verdict": "mechanics_pattern_favorable" if favorable else "strong_null_not_rejected",
        "strong_null_rejected": favorable,
        "scientific_promotion": False,
        "interpretation_limit": config["verdict"]["interpretation_limit"],
    }


IMPLEMENTATION_PATHS = (
    Path("configs/experiment/edcm1_event_triggered_coalition.yaml"),
    Path("src/mop/studies/edcm1_event_triggered_coalition.py"),
    Path("scripts/run_edcm1_event_triggered_coalition.py"),
    Path("tests/test_edcm1_event_triggered_coalition.py"),
    Path("docs/audits/edcm1_event_triggered_coalition.md"),
)


def build_implementation_authority(
    *,
    config_authority_sha256: str,
    mode: str,
    review_status: str,
    study_id: str = "edcm1-event-triggered-heterogeneous-coalition-crossover-v3",
    file_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    receipts = (
        [dict(receipt) for receipt in file_receipts]
        if file_receipts is not None
        else [_file_receipt(REPO_ROOT / relative) for relative in IMPLEMENTATION_PATHS]
    )
    core = {
        "schema": IMPLEMENTATION_AUTHORITY_SCHEMA,
        "study_id": str(study_id),
        "mode": str(mode),
        "config_authority_sha256": str(config_authority_sha256),
        "review_status": str(review_status),
        "files": receipts,
    }
    document = dict(core)
    document["manifest_sha256"] = canonical_sha256(core)
    return document


def write_implementation_authority(
    path: Path | str = DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
    *,
    config_authority_sha256: str = OFFICIAL_AUTHORITY_SHA256,
    mode: str = "official",
    review_status: str = OFFICIAL_IMPLEMENTATION_REVIEW_STATUS,
    study_id: str = "edcm1-event-triggered-heterogeneous-coalition-crossover-v3",
) -> dict[str, Any]:
    target = Path(path).resolve()
    _require_distinct_paths(
        {
            "implementation_authority_output": target,
            **{f"scoped_file:{relative}": (REPO_ROOT / relative) for relative in IMPLEMENTATION_PATHS},
        }
    )
    document = build_implementation_authority(
        config_authority_sha256=config_authority_sha256,
        mode=mode,
        review_status=review_status,
        study_id=study_id,
    )
    _atomic_json(target, document)
    return document


def _load_implementation_authority_snapshot(
    path: Path | str,
    config: Mapping[str, Any],
    *,
    expected_sha256: str | None,
    exploratory: bool,
    config_source_receipt: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path).resolve()
    if exploratory:
        _require(
            source != DEFAULT_IMPLEMENTATION_AUTHORITY_PATH.resolve(),
            "exploratory execution requires an explicit nonofficial implementation authority",
        )
    else:
        _require(
            source == DEFAULT_IMPLEMENTATION_AUTHORITY_PATH.resolve(),
            "official execution requires the default frozen implementation authority",
        )
        _require(
            isinstance(expected_sha256, str)
            and len(expected_sha256) == 64
            and all(character in "0123456789abcdef" for character in expected_sha256),
            "official execution requires an independently supplied implementation authority SHA-256",
        )
    document, source_receipt = _read_json_artifact_snapshot(
        source,
        MAX_IMPLEMENTATION_AUTHORITY_BYTES,
        "implementation authority",
    )
    _require_exact_keys(document, IMPLEMENTATION_AUTHORITY_KEYS, "implementation authority")
    digest = str(document["manifest_sha256"])
    core = dict(document)
    core.pop("manifest_sha256")
    _require(digest == canonical_sha256(core), "implementation authority self-hash mismatch")
    if exploratory:
        if expected_sha256 is not None:
            _require(digest == expected_sha256, "nonofficial implementation authority pin mismatch")
    else:
        _require(digest == expected_sha256, "official implementation authority pin mismatch")
    _require(
        document["schema"] == IMPLEMENTATION_AUTHORITY_SCHEMA,
        "implementation authority schema mismatch",
    )
    _require(document["study_id"] == config["study_id"], "implementation authority study mismatch")
    _require(
        document["config_authority_sha256"] == canonical_sha256(config),
        "implementation/config authority mismatch",
    )
    _require(isinstance(document["files"], list), "implementation authority files must be a list")
    for index, file_receipt in enumerate(document["files"]):
        _require_exact_keys(
            file_receipt,
            FILE_RECEIPT_KEYS,
            f"implementation authority file receipt {index}",
        )
    if config_source_receipt is not None:
        _require_exact_keys(
            config_source_receipt,
            FILE_RECEIPT_KEYS,
            "configuration source snapshot receipt",
        )
    current_files = []
    for relative in IMPLEMENTATION_PATHS:
        scoped_path = (REPO_ROOT / relative).resolve()
        scoped_label = str(scoped_path.relative_to(REPO_ROOT))
        if (
            config_source_receipt is not None
            and scoped_path == DEFAULT_CONFIG_PATH.resolve()
            and config_source_receipt["path"] == scoped_label
        ):
            current_files.append(dict(config_source_receipt))
        else:
            current_files.append(_file_receipt(scoped_path))
    _require(document["files"] == current_files, "implementation authority file receipts mismatch")
    if exploratory:
        _require(document["mode"] == "exploratory", "nonofficial implementation authority mode required")
        _require(bool(document["review_status"]), "nonofficial review status missing")
    else:
        _require(document["mode"] == "official", "official implementation authority mode required")
        _require(
            document["review_status"] == OFFICIAL_IMPLEMENTATION_REVIEW_STATUS,
            "official implementation review status mismatch",
        )
    return document, source_receipt


def load_implementation_authority(
    path: Path | str,
    config: Mapping[str, Any],
    *,
    expected_sha256: str | None,
    exploratory: bool,
) -> dict[str, Any]:
    document, _ = _load_implementation_authority_snapshot(
        path,
        config,
        expected_sha256=expected_sha256,
        exploratory=exploratory,
        config_source_receipt=None,
    )
    return document


def _runtime_identity() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


def _prospective_artifact_guard(config: Mapping[str, Any]) -> dict[str, int]:
    recurrent = config["recurrent_control"]
    candidate_count = (
        len(recurrent["hidden_sizes"]) * len(recurrent["learning_rates"]) * len(recurrent["reservoir_scales"])
    )
    tune = int(config["splits"]["tune_episodes"])
    gate = int(config["splits"]["gate_episodes"])
    heldout = int(config["splits"]["heldout_episodes"])
    per_seed_records = 3 * tune + 3 * gate + candidate_count * tune + tune
    per_seed_records += (len(MAIN_ARMS) + 1) * heldout
    total_records = len(config["seeds"]) * per_seed_records
    resources = config["resources"]
    estimated_bytes = total_records * int(resources["prospective_episode_record_bytes"]) + int(
        resources["prospective_fixed_overhead_bytes"]
    )
    limit = min(
        int(resources["max_receipt_bytes"]),
        int(resources["max_checkpoint_bytes"]),
    )
    guard_limit = int(0.90 * limit)
    _require(estimated_bytes <= guard_limit, "prospective artifact byte envelope exceeded")
    return {
        "candidate_count": candidate_count,
        "episode_records": total_records,
        "estimated_bytes": estimated_bytes,
        "guard_limit_bytes": guard_limit,
    }


def _checkpoint_core(
    authority_hash: str,
    implementation_authority_hash: str,
    implementation_hash: str,
    gate_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "authority_sha256": authority_hash,
        "implementation_authority_sha256": implementation_authority_hash,
        "implementation_sha256": implementation_hash,
        "runtime_identity": _runtime_identity(),
        "gate_rows": list(gate_rows),
        "heldout_rows": list(heldout_rows),
        "gate_seed_ids": [int(row["seed"]) for row in gate_rows],
        "heldout_seed_ids": [int(row["seed"]) for row in heldout_rows],
        "gate_row_sha256": [canonical_sha256(row) for row in gate_rows],
        "heldout_row_sha256": [canonical_sha256(row) for row in heldout_rows],
    }


def _write_checkpoint(
    path: Path,
    authority_hash: str,
    implementation_authority_hash: str,
    implementation_hash: str,
    gate_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    max_bytes: int,
) -> dict[str, Any]:
    payload = _checkpoint_core(
        authority_hash,
        implementation_authority_hash,
        implementation_hash,
        gate_rows,
        heldout_rows,
    )
    payload["checkpoint_sha256"] = canonical_sha256(payload)
    _require(len(canonical_bytes(payload)) + 1 <= max_bytes, "checkpoint byte envelope exceeded")
    _atomic_json(path, payload)
    return payload


def _read_written_checkpoint_snapshot(
    path: Path,
    expected_payload: Mapping[str, Any],
    max_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, source_receipt = _read_json_artifact_snapshot(path, max_bytes, "checkpoint")
    _require(
        canonical_bytes(document) == canonical_bytes(expected_payload),
        "written checkpoint snapshot mismatch",
    )
    return document, source_receipt


def _validate_checkpoint_document(
    document: Mapping[str, Any],
    authority_hash: str,
    implementation_authority_hash: str,
    implementation_hash: str,
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _require_exact_keys(document, CHECKPOINT_KEYS, "checkpoint")
    payload = dict(document)
    digest = payload.pop("checkpoint_sha256", None)
    _require(digest == canonical_sha256(payload), "checkpoint self-hash mismatch")
    _require(payload.get("schema") == CHECKPOINT_SCHEMA, "checkpoint schema mismatch")
    _require(payload.get("authority_sha256") == authority_hash, "checkpoint authority mismatch")
    _require(
        payload.get("implementation_authority_sha256") == implementation_authority_hash,
        "checkpoint implementation authority mismatch",
    )
    _require(
        payload.get("implementation_sha256") == implementation_hash, "checkpoint implementation mismatch"
    )
    _require(payload.get("runtime_identity") == _runtime_identity(), "checkpoint runtime mismatch")
    gate_rows = payload.get("gate_rows")
    heldout_rows = payload.get("heldout_rows")
    if not isinstance(gate_rows, list) or not isinstance(heldout_rows, list):
        raise ValueError("checkpoint rows missing")
    for row in gate_rows:
        _require_exact_keys(row, GATE_ROW_KEYS, "checkpoint gate row")
    for row in heldout_rows:
        _require_exact_keys(row, HELDOUT_ROW_KEYS, "checkpoint heldout row")
    gate_ids = [int(row["seed"]) for row in gate_rows]
    heldout_ids = [int(row["seed"]) for row in heldout_rows]
    _require(gate_ids == list(seeds[: len(gate_ids)]), "gate checkpoint is not a seed prefix")
    _require(heldout_ids == list(seeds[: len(heldout_ids)]), "heldout checkpoint is not a seed prefix")
    _require(gate_ids == payload.get("gate_seed_ids"), "gate checkpoint index mismatch")
    _require(heldout_ids == payload.get("heldout_seed_ids"), "heldout checkpoint index mismatch")
    _require(
        [canonical_sha256(row) for row in gate_rows] == payload.get("gate_row_sha256"),
        "gate row hash mismatch",
    )
    _require(
        [canonical_sha256(row) for row in heldout_rows] == payload.get("heldout_row_sha256"),
        "heldout row hash mismatch",
    )
    _require(not heldout_rows or len(gate_rows) == len(seeds), "heldout rows exist before gate completion")
    return list(gate_rows), list(heldout_rows)


def _load_checkpoint(
    path: Path,
    authority_hash: str,
    implementation_authority_hash: str,
    implementation_hash: str,
    seeds: Sequence[int],
    max_bytes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    document, _ = _read_json_artifact_snapshot(path, max_bytes, "checkpoint")
    return _validate_checkpoint_document(
        document,
        authority_hash,
        implementation_authority_hash,
        implementation_hash,
        seeds,
    )


def _execution_manifest(
    config: Mapping[str, Any],
    gate_result: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    seeds = [int(seed) for seed in config["seeds"]]
    gate_complete = len(gate_rows) == len(seeds)
    gate_stop = gate_complete and not bool(gate_result.get("passed"))
    heldout_complete = len(heldout_rows) == len(seeds)
    if not gate_complete or (bool(gate_result.get("passed")) and not heldout_complete):
        status = "partial"
        problems = ["execution_incomplete"]
        resumable = True
        stop_reason = None
    elif gate_stop:
        status = "terminal_scientific_stop"
        problems = []
        resumable = False
        stop_reason = str(config["verdict"]["gate_stop_label"])
    else:
        status = "complete"
        problems = []
        resumable = False
        stop_reason = None
    return {
        "execution_status": status,
        "all_ok": not problems,
        "problems": problems,
        "resumable": resumable,
        "terminal_scientific_stop": gate_stop,
        "terminal_stop_reason": stop_reason,
        "completed_gate_seeds": [int(row["seed"]) for row in gate_rows],
        "completed_heldout_seeds": [int(row["seed"]) for row in heldout_rows],
        "required_gate_seeds": seeds,
        "required_heldout_seeds": seeds if bool(gate_result.get("passed")) else [],
    }


def _validate_selected_controls(
    heldout_rows: Sequence[Mapping[str, Any]],
    gate_result: Mapping[str, Any],
) -> None:
    for row in heldout_rows:
        _require(
            row["selected_best_single"] == gate_result.get("selected_best_single"),
            "heldout selected single does not match the gate",
        )
        _require(
            row["selected_recurrent"] == gate_result.get("selected_recurrent"),
            "heldout selected recurrent control does not match the gate",
        )


def _verify_checkpoint_binding(
    binding: Mapping[str, Any],
    checkpoint_path: Path,
    authority_hash: str,
    implementation_authority_hash: str,
    implementation_hash: str,
    seeds: Sequence[int],
    gate_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    max_bytes: int,
) -> None:
    _require_exact_keys(binding, CHECKPOINT_BINDING_KEYS, "checkpoint binding")
    _require_exact_keys(binding["file"], FILE_RECEIPT_KEYS, "checkpoint binding file receipt")
    checkpoint_document, checkpoint_source_receipt = _read_json_artifact_snapshot(
        checkpoint_path,
        max_bytes,
        "checkpoint",
    )
    _require(
        binding["file"] == checkpoint_source_receipt,
        "bound checkpoint file mismatch",
    )
    _require(
        checkpoint_document.get("checkpoint_sha256") == binding["checkpoint_sha256"],
        "bound checkpoint digest mismatch",
    )
    checkpoint_gate_rows, checkpoint_heldout_rows = _validate_checkpoint_document(
        checkpoint_document,
        authority_hash,
        implementation_authority_hash,
        implementation_hash,
        seeds,
    )
    _require(list(checkpoint_gate_rows) == list(gate_rows), "receipt/checkpoint gate row join mismatch")
    _require(
        list(checkpoint_heldout_rows) == list(heldout_rows), "receipt/checkpoint heldout row join mismatch"
    )
    _require(
        binding["gate_row_sha256"] == [canonical_sha256(row) for row in gate_rows],
        "bound gate row hash mismatch",
    )
    _require(
        binding["heldout_row_sha256"] == [canonical_sha256(row) for row in heldout_rows],
        "bound heldout row hash mismatch",
    )


def validate_full_regeneration(
    config: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
    gate_result: Mapping[str, Any],
) -> dict[str, Any]:

    regenerated_gate: list[dict[str, Any]] = []
    for stored in gate_rows:
        regenerated = run_gate_seed(config, int(stored["seed"]))
        _require(
            canonical_bytes(regenerated) == canonical_bytes(stored),
            f"full regeneration gate row mismatch for seed {stored['seed']}",
        )
        regenerated_gate.append(regenerated)
    regenerated_gate_result = aggregate_gate(regenerated_gate, config)
    _require(
        regenerated_gate_result == gate_result,
        "full regeneration gate aggregate mismatch",
    )
    regenerated_heldout: list[dict[str, Any]] = []
    if heldout_rows:
        _require(bool(gate_result.get("passed")), "heldout regeneration requires a passing gate")
        selected_kind = str(gate_result["selected_best_single"])
        selected_recurrent = gate_result["selected_recurrent"]
        for stored in heldout_rows:
            regenerated = run_heldout_seed(
                config,
                int(stored["seed"]),
                selected_kind,
                selected_recurrent,
            )
            _require(
                canonical_bytes(regenerated) == canonical_bytes(stored),
                f"full regeneration heldout row mismatch for seed {stored['seed']}",
            )
            regenerated_heldout.append(regenerated)
    return {
        "mode": OFFICIAL_VERIFIER_MODE,
        "regenerated_gate_seeds": [int(row["seed"]) for row in regenerated_gate],
        "regenerated_heldout_seeds": [int(row["seed"]) for row in regenerated_heldout],
    }


def run_from_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    implementation_authority_path: Path | str = DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
    *,
    implementation_authority_sha256: str | None = None,
    max_new_seeds: int | None = None,
    verifier_mode: str | None = None,
    exploratory: bool = False,
) -> dict[str, Any]:
    _require(
        max_new_seeds is None or max_new_seeds >= 0,
        "max_new_seeds must be nonnegative",
    )
    source = Path(config_path).resolve()
    output = Path(output_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    implementation_authority_source = Path(implementation_authority_path).resolve()
    _require_distinct_paths(
        {
            "config": source,
            "output": output,
            "checkpoint": checkpoint,
            "implementation_authority": implementation_authority_source,
        }
    )
    if exploratory:
        _require(
            output != DEFAULT_OUTPUT_PATH.resolve(),
            "exploratory execution may not use the official proof path",
        )
        _require(
            checkpoint != DEFAULT_CHECKPOINT_PATH.resolve(),
            "exploratory execution may not use the official checkpoint path",
        )
        receipt_verifier_mode = verifier_mode or DIAGNOSTIC_VERIFIER_MODE
        _require(
            receipt_verifier_mode in (OFFICIAL_VERIFIER_MODE, DIAGNOSTIC_VERIFIER_MODE),
            "unknown exploratory verifier mode",
        )
    else:
        receipt_verifier_mode = verifier_mode or OFFICIAL_VERIFIER_MODE
        _require(
            receipt_verifier_mode == OFFICIAL_VERIFIER_MODE,
            "official receipts require full deterministic regeneration",
        )
    config, envelope, config_source_receipt = _load_config_snapshot(
        source,
        exploratory=exploratory,
    )
    implementation_authority, implementation_authority_source_receipt = (
        _load_implementation_authority_snapshot(
            implementation_authority_source,
            config,
            expected_sha256=implementation_authority_sha256,
            exploratory=exploratory,
            config_source_receipt=config_source_receipt,
        )
    )
    prospective_guard = _prospective_artifact_guard(config)
    authority_hash = canonical_sha256(config)
    implementation = list(implementation_authority["files"])
    implementation_authority_hash = str(implementation_authority["manifest_sha256"])
    runtime_identity = _runtime_identity()
    implementation_hash = canonical_sha256(
        {
            "implementation_authority_sha256": implementation_authority_hash,
            "runtime": runtime_identity,
        }
    )
    seeds = [int(seed) for seed in config["seeds"]]
    gate_rows, heldout_rows = _load_checkpoint(
        checkpoint,
        authority_hash,
        implementation_authority_hash,
        implementation_hash,
        seeds,
        int(config["resources"]["max_checkpoint_bytes"]),
    )
    remaining_budget = max_new_seeds
    for seed in seeds[len(gate_rows) :]:
        if remaining_budget is not None and remaining_budget <= 0:
            break
        gate_rows.append(run_gate_seed(config, seed))
        _write_checkpoint(
            checkpoint,
            authority_hash,
            implementation_authority_hash,
            implementation_hash,
            gate_rows,
            heldout_rows,
            int(config["resources"]["max_checkpoint_bytes"]),
        )
        if remaining_budget is not None:
            remaining_budget -= 1
    gate_result = aggregate_gate(gate_rows, config)
    _require(
        not heldout_rows or bool(gate_result.get("passed")),
        "heldout checkpoint rows exist without a passing gate",
    )
    if gate_result.get("passed"):
        selected_kind = str(gate_result["selected_best_single"])
        selected_recurrent = gate_result["selected_recurrent"]
        for seed in seeds[len(heldout_rows) :]:
            if remaining_budget is not None and remaining_budget <= 0:
                break
            heldout_rows.append(run_heldout_seed(config, seed, selected_kind, selected_recurrent))
            _write_checkpoint(
                checkpoint,
                authority_hash,
                implementation_authority_hash,
                implementation_hash,
                gate_rows,
                heldout_rows,
                int(config["resources"]["max_checkpoint_bytes"]),
            )
            if remaining_budget is not None:
                remaining_budget -= 1
    if exploratory:
        aggregate = {
            "status": "exploratory",
            "verdict": "exploratory_only",
            "scientific_promotion": False,
            "gate": gate_result,
        }
    elif gate_result.get("status") == "complete" and not gate_result.get("passed"):
        aggregate = {
            "status": "complementarity-gate-failed",
            "verdict": config["verdict"]["gate_stop_label"],
            "scientific_promotion": False,
            "gate": gate_result,
        }
    else:
        aggregate = aggregate_heldout(heldout_rows, config, gate_result)
    checkpoint_payload = _write_checkpoint(
        checkpoint,
        authority_hash,
        implementation_authority_hash,
        implementation_hash,
        gate_rows,
        heldout_rows,
        int(config["resources"]["max_checkpoint_bytes"]),
    )
    _, checkpoint_source_receipt = _read_written_checkpoint_snapshot(
        checkpoint,
        checkpoint_payload,
        int(config["resources"]["max_checkpoint_bytes"]),
    )
    execution = _execution_manifest(config, gate_result, gate_rows, heldout_rows)
    core = {
        "schema": RECEIPT_SCHEMA,
        "study_id": config["study_id"],
        "claim_scope": CLAIM_SCOPE,
        "strong_null": config["strong_null"],
        "authority": envelope["authority"],
        "authority_sha256": authority_hash,
        "config_source": config_source_receipt,
        "implementation": implementation,
        "implementation_authority": {
            "source": implementation_authority_source_receipt,
            "mode": implementation_authority["mode"],
            "review_status": implementation_authority["review_status"],
            "manifest_sha256": implementation_authority_hash,
        },
        "implementation_authority_sha256": implementation_authority_hash,
        "implementation_sha256": implementation_hash,
        "runtime_identity": runtime_identity,
        "gate_rows": gate_rows,
        "gate": gate_result,
        "heldout_rows": heldout_rows,
        "aggregate": aggregate,
        **execution,
        "abstract_work_contract": config["abstract_work"],
        "empirical_timing": dict(EMPIRICAL_TIMING_CLAIM),
        "resume": {
            "granularity": "phase-and-completed-seed-boundary",
            "checkpoint_path": str(checkpoint),
            "deterministic": True,
        },
        "checkpoint_binding": {
            "file": checkpoint_source_receipt,
            "checkpoint_sha256": checkpoint_payload["checkpoint_sha256"],
            "gate_row_sha256": checkpoint_payload["gate_row_sha256"],
            "heldout_row_sha256": checkpoint_payload["heldout_row_sha256"],
        },
        "prospective_artifact_guard": prospective_guard,
        "exploratory": exploratory,
        "verifier_mode": receipt_verifier_mode,
        "scientific_promotion": False,
    }
    receipt = dict(core)
    receipt["deterministic_core_sha256"] = canonical_sha256(core)
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _require(
        len(canonical_bytes(receipt)) + 1 <= int(config["resources"]["max_receipt_bytes"]),
        "receipt byte envelope exceeded",
    )
    _atomic_json(output, receipt)
    return receipt


def verify_receipt(
    path: Path | str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    checkpoint_path: Path | str | None = None,
    implementation_authority_path: Path | str = DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
    implementation_authority_sha256: str | None = None,
    verifier_mode: str | None = None,
    exploratory: bool = False,
) -> dict[str, Any]:
    config_source = Path(config_path).resolve()
    receipt_path = Path(path).resolve()
    implementation_authority_source = Path(implementation_authority_path).resolve()
    _require_distinct_paths(
        {
            "config": config_source,
            "receipt": receipt_path,
            "implementation_authority": implementation_authority_source,
        }
    )
    config, current_envelope, config_source_receipt = _load_config_snapshot(
        config_source,
        exploratory=exploratory,
    )
    receipt, receipt_source_receipt = _read_json_artifact_snapshot(
        receipt_path,
        int(config["resources"]["max_receipt_bytes"]),
        "receipt",
    )
    _require_exact_keys(receipt, RECEIPT_KEYS, "receipt")
    _require(receipt.get("schema") == RECEIPT_SCHEMA, "receipt schema mismatch")
    receipt_without_hash = dict(receipt)
    receipt_digest = receipt_without_hash.pop("receipt_sha256", None)
    _require(receipt_digest == canonical_sha256(receipt_without_hash), "receipt self-hash mismatch")
    core_digest = receipt_without_hash.pop("deterministic_core_sha256", None)
    _require(core_digest == canonical_sha256(receipt_without_hash), "deterministic core hash mismatch")
    stored_verifier_mode = str(receipt["verifier_mode"])
    selected_verifier_mode = verifier_mode or stored_verifier_mode
    _require(
        selected_verifier_mode == stored_verifier_mode,
        "requested verifier mode does not match receipt authority",
    )
    if exploratory:
        _require(
            selected_verifier_mode in (OFFICIAL_VERIFIER_MODE, DIAGNOSTIC_VERIFIER_MODE),
            "unknown exploratory verifier mode",
        )
    else:
        _require(
            selected_verifier_mode == OFFICIAL_VERIFIER_MODE,
            "official verification requires full deterministic regeneration",
        )
    _require(receipt["study_id"] == config["study_id"], "receipt study id mismatch")
    _require(bool(receipt["exploratory"]) == exploratory, "receipt exploratory mode mismatch")
    _require(receipt["claim_scope"] == CLAIM_SCOPE, "receipt claim scope mismatch")
    _require(receipt["strong_null"] == config["strong_null"], "receipt strong null mismatch")
    _require(receipt["scientific_promotion"] is False, "receipt scientific promotion escaped")
    _require(receipt["abstract_work_contract"] == config["abstract_work"], "abstract work contract mismatch")
    _require_exact_keys(receipt["empirical_timing"], EMPIRICAL_TIMING_KEYS, "receipt empirical timing")
    _require(
        receipt["empirical_timing"] == EMPIRICAL_TIMING_CLAIM,
        "receipt empirical timing claim mismatch",
    )
    _require_exact_keys(receipt["resume"], RESUME_KEYS, "receipt resume")
    _require(
        receipt["resume"]["granularity"] == "phase-and-completed-seed-boundary"
        and receipt["resume"]["deterministic"] is True
        and isinstance(receipt["resume"]["checkpoint_path"], str)
        and Path(receipt["resume"]["checkpoint_path"]).is_absolute(),
        "receipt resume claim mismatch",
    )
    _require_exact_keys(
        receipt["checkpoint_binding"],
        CHECKPOINT_BINDING_KEYS,
        "receipt checkpoint binding",
    )
    _require_exact_keys(receipt["config_source"], FILE_RECEIPT_KEYS, "receipt config source")
    _require(
        receipt["prospective_artifact_guard"] == _prospective_artifact_guard(config),
        "artifact guard mismatch",
    )
    _require(receipt["authority"] == current_envelope["authority"], "receipt authority envelope mismatch")
    _require(receipt["authority_sha256"] == canonical_sha256(config), "receipt authority mismatch")
    _require(
        receipt["config_source"] == config_source_receipt,
        "receipt config source mismatch",
    )
    implementation_authority, implementation_authority_source_receipt = (
        _load_implementation_authority_snapshot(
            implementation_authority_source,
            config,
            expected_sha256=implementation_authority_sha256,
            exploratory=exploratory,
            config_source_receipt=config_source_receipt,
        )
    )
    implementation_authority_hash = str(implementation_authority["manifest_sha256"])
    expected_implementation_authority_receipt = {
        "source": implementation_authority_source_receipt,
        "mode": implementation_authority["mode"],
        "review_status": implementation_authority["review_status"],
        "manifest_sha256": implementation_authority_hash,
    }
    _require_exact_keys(
        receipt["implementation_authority"],
        IMPLEMENTATION_AUTHORITY_RECEIPT_KEYS,
        "receipt implementation authority",
    )
    _require_exact_keys(
        receipt["implementation_authority"]["source"],
        FILE_RECEIPT_KEYS,
        "receipt implementation authority source",
    )
    _require(
        receipt["implementation_authority"] == expected_implementation_authority_receipt,
        "receipt implementation authority mismatch",
    )
    _require(
        receipt["implementation_authority_sha256"] == implementation_authority_hash,
        "receipt implementation authority digest mismatch",
    )
    implementation = list(implementation_authority["files"])
    _require(receipt["implementation"] == implementation, "receipt implementation mismatch")
    runtime_identity = _runtime_identity()
    _require(receipt["runtime_identity"] == runtime_identity, "receipt runtime mismatch")
    _require(
        receipt["implementation_sha256"]
        == canonical_sha256(
            {
                "implementation_authority_sha256": implementation_authority_hash,
                "runtime": runtime_identity,
            }
        ),
        "implementation hash mismatch",
    )
    implementation_hash = receipt["implementation_sha256"]
    gate_rows = receipt["gate_rows"]
    seeds = [int(seed) for seed in config["seeds"]]
    _require(
        [int(row["seed"]) for row in gate_rows] == seeds[: len(gate_rows)],
        "receipt gate rows are not a seed prefix",
    )
    gate_result = aggregate_gate(gate_rows, config)
    _require(receipt["gate"] == gate_result, "stored gate aggregate mismatch")
    heldout_rows = receipt["heldout_rows"]
    _require(
        [int(row["seed"]) for row in heldout_rows] == seeds[: len(heldout_rows)],
        "receipt heldout rows are not a seed prefix",
    )
    _require(not heldout_rows or len(gate_rows) == len(seeds), "receipt heldout rows precede gate completion")
    _require(
        not heldout_rows or bool(gate_result.get("passed")),
        "receipt heldout rows exist without a passing gate",
    )
    for row in heldout_rows:
        validate_heldout_row(row, config)
    _validate_selected_controls(heldout_rows, gate_result)
    expected_execution = _execution_manifest(config, gate_result, gate_rows, heldout_rows)
    for key, expected in expected_execution.items():
        _require(receipt.get(key) == expected, f"receipt execution field mismatch: {key}")
    _require_terminal_execution(expected_execution)
    if selected_verifier_mode == OFFICIAL_VERIFIER_MODE:
        regeneration = validate_full_regeneration(
            config,
            gate_rows,
            heldout_rows,
            gate_result,
        )
    else:
        regeneration = {
            "mode": DIAGNOSTIC_VERIFIER_MODE,
            "regenerated_gate_seeds": [],
            "regenerated_heldout_seeds": [],
        }
    binding = receipt["checkpoint_binding"]
    bound_checkpoint = Path(
        checkpoint_path if checkpoint_path is not None else receipt["resume"]["checkpoint_path"]
    ).resolve()
    _require(
        Path(receipt["resume"]["checkpoint_path"]).resolve() == bound_checkpoint,
        "receipt resume/checkpoint path mismatch",
    )
    _require_distinct_paths(
        {
            "config": config_source,
            "receipt": receipt_path,
            "checkpoint": bound_checkpoint,
            "implementation_authority": implementation_authority_source,
        }
    )
    _verify_checkpoint_binding(
        binding,
        bound_checkpoint,
        receipt["authority_sha256"],
        implementation_authority_hash,
        implementation_hash,
        seeds,
        gate_rows,
        heldout_rows,
        int(config["resources"]["max_checkpoint_bytes"]),
    )
    if receipt["exploratory"]:
        expected_aggregate = receipt["aggregate"]
        _require(expected_aggregate["verdict"] == "exploratory_only", "exploratory verdict escaped")
    elif gate_result.get("status") == "complete" and not gate_result.get("passed"):
        expected_aggregate = {
            "status": "complementarity-gate-failed",
            "verdict": config["verdict"]["gate_stop_label"],
            "scientific_promotion": False,
            "gate": gate_result,
        }
        _require(receipt["aggregate"] == expected_aggregate, "stored gate-stop aggregate mismatch")
    else:
        expected_aggregate = aggregate_heldout(heldout_rows, config, gate_result)
        _require(receipt["aggregate"] == expected_aggregate, "stored heldout aggregate mismatch")
    _require(receipt["aggregate"]["scientific_promotion"] is False, "aggregate scientific promotion escaped")
    return {
        "valid": True,
        "gate_seed_ids": [int(row["seed"]) for row in gate_rows],
        "heldout_seed_ids": [int(row["seed"]) for row in heldout_rows],
        "verdict": receipt["aggregate"]["verdict"],
        "execution_status": receipt["execution_status"],
        "verifier_mode": selected_verifier_mode,
        "regeneration": regeneration,
        "authority_sha256": receipt["authority_sha256"],
        "implementation_authority_sha256": implementation_authority_hash,
        "verified_sources": {
            "receipt": receipt_source_receipt,
            "receipt_path": str(receipt_path),
            "checkpoint": dict(binding["file"]),
            "checkpoint_path": str(bound_checkpoint),
            "config": config_source_receipt,
            "implementation_authority": implementation_authority_source_receipt,
        },
        "scientific_promotion": False,
    }


def _require_terminal_execution(execution: Mapping[str, Any]) -> None:
    _require(
        execution.get("execution_status") in {"complete", "terminal_scientific_stop"}
        and execution.get("all_ok") is True
        and execution.get("problems") == []
        and execution.get("resumable") is False,
        "verification refuses a nonterminal partial receipt",
    )


def _require_terminal_verification_result(result: Mapping[str, Any]) -> None:
    _require(
        result.get("valid") is True
        and result.get("execution_status") in {"complete", "terminal_scientific_stop"}
        and result.get("scientific_promotion") is False,
        "verification artifact requires a valid terminal result",
    )


def build_verification_artifact(result: Mapping[str, Any]) -> dict[str, Any]:

    _require_terminal_verification_result(result)
    core = {
        "schema": VERIFICATION_ARTIFACT_SCHEMA,
        "study_id": "edcm1-event-triggered-heterogeneous-coalition-crossover-v3",
        "claim_scope": CLAIM_SCOPE,
        "verification": dict(result),
        "scientific_promotion": False,
    }
    artifact = dict(core)
    artifact["verification_artifact_sha256"] = canonical_sha256(core)
    return artifact


def write_verification_artifact(
    path: Path | str,
    result: Mapping[str, Any],
    *,
    protected_paths: Mapping[str, Path | str] | None = None,
) -> dict[str, Any]:
    target = Path(path).resolve()
    identities: dict[str, Path] = {"verification_output": target}
    if protected_paths is not None:
        identities.update({label: Path(value).resolve() for label, value in protected_paths.items()})
    _require_distinct_paths(identities)
    artifact = build_verification_artifact(result)
    _require(
        len(canonical_bytes(artifact)) + 1 <= MAX_VERIFICATION_ARTIFACT_BYTES,
        "verification artifact byte envelope exceeded",
    )
    _atomic_json(target, artifact)
    written, _ = _read_json_artifact_snapshot(
        target,
        MAX_VERIFICATION_ARTIFACT_BYTES,
        "verification artifact",
    )
    _require(written == artifact, "written verification artifact snapshot mismatch")
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", "--out", dest="output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--implementation-authority",
        type=Path,
        default=DEFAULT_IMPLEMENTATION_AUTHORITY_PATH,
    )
    parser.add_argument("--implementation-authority-sha256")
    parser.add_argument(
        "--verifier-mode",
        choices=(OFFICIAL_VERIFIER_MODE, DIAGNOSTIC_VERIFIER_MODE),
    )
    parser.add_argument("--max-new-seeds", type=int)
    parser.add_argument("--verify", type=Path)
    parser.add_argument(
        "--verification-out",
        type=Path,
        default=DEFAULT_VERIFICATION_OUTPUT_PATH,
    )
    parser.add_argument("--exploratory", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.verify is not None:
        verification_paths: dict[str, Path] = {
            "config": arguments.config,
            "receipt": arguments.verify,
            "implementation_authority": arguments.implementation_authority,
            "verification_output": arguments.verification_out,
        }
        if arguments.checkpoint is not None:
            verification_paths["checkpoint"] = arguments.checkpoint
        _require_distinct_paths(verification_paths)
        if arguments.exploratory:
            _require(
                arguments.verification_out.resolve() != DEFAULT_VERIFICATION_OUTPUT_PATH.resolve(),
                "exploratory verification may not use the official verification artifact path",
            )
        result = verify_receipt(
            arguments.verify,
            arguments.config,
            checkpoint_path=arguments.checkpoint,
            implementation_authority_path=arguments.implementation_authority,
            implementation_authority_sha256=arguments.implementation_authority_sha256,
            verifier_mode=arguments.verifier_mode,
            exploratory=arguments.exploratory,
        )
        _require_terminal_verification_result(result)
        protected_paths: dict[str, Path | str] = {
            "config": arguments.config,
            "receipt": arguments.verify,
            "implementation_authority": arguments.implementation_authority,
        }
        verified_sources = result.get("verified_sources")
        if isinstance(verified_sources, Mapping):
            checkpoint_source = verified_sources.get("checkpoint_path")
            if isinstance(checkpoint_source, str):
                protected_paths["checkpoint"] = checkpoint_source
        if "checkpoint" not in protected_paths and arguments.checkpoint is not None:
            protected_paths["checkpoint"] = arguments.checkpoint
        write_verification_artifact(
            arguments.verification_out,
            result,
            protected_paths=protected_paths,
        )
    else:
        result = run_from_config(
            arguments.config,
            arguments.output,
            arguments.checkpoint or DEFAULT_CHECKPOINT_PATH,
            arguments.implementation_authority,
            implementation_authority_sha256=arguments.implementation_authority_sha256,
            max_new_seeds=arguments.max_new_seeds,
            verifier_mode=arguments.verifier_mode,
            exploratory=arguments.exploratory,
        )
    print(
        json.dumps(result if arguments.verify is not None else result["aggregate"], sort_keys=True, indent=2)
    )
    return 2 if arguments.verify is None and bool(result["resumable"]) else 0


__all__ = [
    "AbstractWork",
    "ActivationRecord",
    "BudgetLedger",
    "CoalitionController",
    "ContradictionVerifier",
    "DIRECT_INTERVENTIONS",
    "EqualBudgetRecurrentController",
    "EvaluatorTransition",
    "EventSentinel",
    "MAIN_ARMS",
    "OFFICIAL_AUTHORITY_SHA256",
    "OFFICIAL_IMPLEMENTATION_REVIEW_STATUS",
    "OFFICIAL_VERIFIER_MODE",
    "PROPOSER_ORDER",
    "PartialChangePointWorld",
    "PreparedDecision",
    "ProposalMessage",
    "Resolution",
    "VisibleObservation",
    "VisibleTransition",
    "aggregate_gate",
    "aggregate_heldout",
    "accounting_sensitivity",
    "build_parser",
    "build_verification_artifact",
    "build_implementation_authority",
    "canonical_bytes",
    "canonical_sha256",
    "load_config",
    "load_implementation_authority",
    "main",
    "periodic_round_matched_schedule",
    "replay_episode_actions",
    "round_activation_counts",
    "run_from_config",
    "run_gate_seed",
    "run_heldout_seed",
    "shuffled_round_matched_schedule",
    "shuffled_coalition_matched_schedule",
    "validate_arm_summary",
    "validate_full_regeneration",
    "validate_heldout_row",
    "verify_receipt",
    "write_verification_artifact",
    "write_implementation_authority",
]
