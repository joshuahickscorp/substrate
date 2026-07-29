"""Equally resourced controls and deprivation baselines for Cognitive Material Genesis.

Every entry of ``genesis_config.BASELINES`` is a separate ``MaterialBase`` subclass with
its own mechanism identifier. Durable change happens only through propose/apply. No arm
reads a held-out label. Activation stays false.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from substrate import genesis_config as C
from substrate.final_revision_field import multiplication_light_dot, native_low_bit_update
from substrate.genesis_material import (
    Answer,
    MaterialBase,
    Observation,
    Opportunity,
    Probe,
    Proposal,
    Receipt,
    register,
)

_TERNARY = (-1, 0, 1)
_QUINARY = (-2, -1, 0, 1, 2)
_ACTIVATION = False


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _bytes_of(value: Any) -> int:
    return len(_canonical(value))


def _key_of(channel: str, payload: Sequence[int]) -> str:
    return f"{channel}|{','.join(str(int(x)) for x in payload)}"


def _hamming(left: Sequence[int], right: Sequence[int]) -> int:
    score = abs(len(left) - len(right))
    for index in range(min(len(left), len(right))):
        if int(left[index]) != int(right[index]):
            score += 1
    return score


def _trim(value: Sequence[int], arity: int) -> tuple[int, ...]:
    if arity <= 0:
        return tuple(int(x) for x in value)
    items = tuple(int(x) for x in value)
    if len(items) >= arity:
        return items[:arity]
    return items + (0,) * (arity - len(items))


def _prefix_keys(payload: Sequence[int]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Emit (key, value) splits used by the monolithic core for sequence learning."""
    items = tuple(int(x) for x in payload)
    if not items:
        return []
    pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    # Whole payload as a unit value under the empty key and under successive prefixes.
    pairs.append(((), items))
    for cut in range(1, len(items)):
        pairs.append((items[:cut], items[cut:]))
    return pairs


def _nearest(
    probe: Sequence[int],
    candidates: Mapping[str, tuple[int, ...]],
) -> tuple[str, tuple[int, ...]] | None:
    if not candidates:
        return None
    best_key = ""
    best_value: tuple[int, ...] = ()
    best_score = 10**9
    for key, value in candidates.items():
        score = _hamming(probe, value)
        if score < best_score or (score == best_score and key < best_key):
            best_score = score
            best_key = key
            best_value = value
    return best_key, best_value


def _det_weights(seed: str, size: int) -> list[int]:
    digest = hashlib.sha256(seed.encode()).digest()
    values: list[int] = []
    cursor = 0
    while len(values) < size:
        if cursor >= len(digest):
            digest = hashlib.sha256(digest).digest()
            cursor = 0
        values.append(int(digest[cursor]) % 5 - 2)
        cursor += 1
    return values


# --------------------------------------------------------------------------
# S2 — strongest equal-opportunity monolithic control
# --------------------------------------------------------------------------


@dataclass
class S2TaskIndependentMonolithicPersistentCore(MaterialBase):
    """One task-independent transition over owned persistent associative state.

    This is a first-class implementation, not a flag on another material. It stages
    sequence associations from every observation, proposes durable writes for the
    strongest uncommitted associations, and answers by exact then nearest lookup
    over the committed table plus the active buffer.
    """

    table: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_buffer: list[dict[str, Any]] = field(default_factory=list)
    staged: dict[str, dict[str, Any]] = field(default_factory=dict)
    undo: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    transition_counts: dict[str, int] = field(default_factory=dict)
    last_channel: str | None = None
    last_payload: tuple[int, ...] = ()

    def _resize(self) -> None:
        self._opportunity.ledger.resize(_bytes_of(self._durable_state()) + _bytes_of(self._active_state()))

    def _transition(self, observation: Observation) -> None:
        payload = tuple(int(x) for x in observation.payload)
        row = {
            "index": observation.index,
            "channel": observation.channel,
            "payload": payload,
            "teaching": bool(observation.teaching),
            "modality": observation.modality,
        }
        self.active_buffer.append(row)
        # Task-independent transition: accumulate associative candidates and channel transitions.
        for key, value in _prefix_keys(payload):
            address = _key_of(observation.channel, key)
            entry = self.staged.get(address)
            if entry is None or len(value) >= len(entry["value"]):
                self.staged[address] = {
                    "channel": observation.channel,
                    "key": key,
                    "value": value,
                    "count": int(entry["count"]) + 1 if entry is not None else 1,
                    "teaching": bool(observation.teaching) or bool(entry and entry.get("teaching")),
                }
            else:
                entry["count"] = int(entry["count"]) + 1
                entry["teaching"] = bool(entry["teaching"]) or bool(observation.teaching)
        if self.last_channel is not None:
            edge = f"{self.last_channel}->{observation.channel}"
            self.transition_counts[edge] = self.transition_counts.get(edge, 0) + 1
            bridge_key = _key_of(f"edge:{self.last_channel}", self.last_payload[:4])
            self.staged[bridge_key] = {
                "channel": f"edge:{self.last_channel}",
                "key": self.last_payload[:4],
                "value": payload[: max(1, len(payload))],
                "count": self.transition_counts[edge],
                "teaching": bool(observation.teaching),
            }
        self.last_channel = observation.channel
        self.last_payload = payload
        self._resize()

    def _answer(self, probe: Probe) -> Answer:
        arity = max(0, int(probe.arity))
        probe_key = tuple(int(x) for x in probe.probe)
        # Exact durable hit on channel+probe key.
        address = _key_of(probe.channel, probe_key)
        if address in self.table:
            value = _trim(self.table[address]["value"], arity)
            strength = int(self.table[address].get("strength", 0))
            return Answer(probe_index=probe.index, value=value, confidence=max(0, min(255, 128 + 40 * strength)), abstained=False)
        # Prefix match: durable keys that extend the probe under the same channel.
        for key, entry in self.table.items():
            if not key.startswith(f"{probe.channel}|"):
                continue
            stored_key = tuple(entry["key"])
            stored_value = tuple(entry["value"])
            if stored_key == probe_key:
                return Answer(probe_index=probe.index, value=_trim(stored_value, arity), confidence=200, abstained=False)
            if probe_key and stored_value[: len(probe_key)] == probe_key:
                continuation = stored_value[len(probe_key) :]
                if continuation:
                    return Answer(probe_index=probe.index, value=_trim(continuation, arity), confidence=160, abstained=False)
            full = stored_key + stored_value
            if probe_key and full[: len(probe_key)] == probe_key and len(full) > len(probe_key):
                return Answer(probe_index=probe.index, value=_trim(full[len(probe_key) :], arity), confidence=140, abstained=False)
        # Active buffer exact channel match (working memory; not durable).
        for row in reversed(self.active_buffer):
            if row["channel"] == probe.channel:
                payload = tuple(row["payload"])
                if not probe_key or payload[: len(probe_key)] == probe_key:
                    value = payload[len(probe_key) :] if probe_key else payload
                    if value or not probe_key:
                        return Answer(probe_index=probe.index, value=_trim(value if value else payload, arity), confidence=90, abstained=False)
        # Nearest neighbour over durable values.
        durable_values = {key: tuple(entry["value"]) for key, entry in self.table.items()}
        nearest = _nearest(probe_key, durable_values) if probe_key else None
        if nearest is not None:
            return Answer(probe_index=probe.index, value=_trim(nearest[1], arity), confidence=40, abstained=False)
        if arity == 0:
            return Answer(probe_index=probe.index, value=(), confidence=0, abstained=True)
        return Answer(probe_index=probe.index, value=tuple(0 for _ in range(arity)), confidence=0, abstained=True)

    def _propose(self) -> Iterable[Proposal]:
        ranked = sorted(
            self.staged.items(),
            key=lambda item: (-int(item[1]["count"]), -int(bool(item[1]["teaching"])), item[0]),
        )
        proposals: list[Proposal] = []
        for address, entry in ranked:
            current = self.table.get(address)
            value = tuple(int(x) for x in entry["value"])
            if current is not None and tuple(current["value"]) == value:
                continue
            count = int(entry["count"])
            proposals.append(
                Proposal(
                    proposal_id=f"s2:{address}:{count}:{len(proposals)}",
                    kind="monolith_association_write",
                    target=address,
                    delta=value,
                    precision_request="ternary",
                    topology_operation=None,
                    trigger="task_independent_transition",
                    expected_value=float(count) + (1.0 if entry["teaching"] else 0.0),
                    cost_bytes=max(8, _bytes_of(value)),
                )
            )
        return proposals

    def _commit(self, proposal: Proposal) -> None:
        address = proposal.target
        prior = self.table.get(address)
        self.undo[proposal.proposal_id] = None if prior is None else {
            "value": tuple(prior["value"]),
            "strength": int(prior["strength"]),
            "count": int(prior["count"]),
            "key": tuple(prior["key"]),
            "channel": str(prior["channel"]),
        }
        strength = 0 if prior is None else int(prior["strength"])
        strength = native_low_bit_update(strength, -1.0, _TERNARY)
        key_part = address.split("|", 1)
        channel = key_part[0]
        key_tokens = key_part[1].split(",") if len(key_part) > 1 and key_part[1] else []
        key = tuple(int(token) for token in key_tokens if token != "")
        self.table[address] = {
            "channel": channel,
            "key": key,
            "value": tuple(int(x) for x in proposal.delta),
            "strength": strength,
            "count": 1 if prior is None else int(prior["count"]) + 1,
        }
        self._resize()

    def _rollback(self, receipt: Receipt) -> None:
        prior = self.undo.pop(receipt.proposal_id, None)
        if prior is None:
            self.table.pop(receipt.target, None)
        else:
            self.table[receipt.target] = {
                "channel": prior["channel"],
                "key": tuple(prior["key"]),
                "value": tuple(prior["value"]),
                "strength": int(prior["strength"]),
                "count": int(prior["count"]),
            }
        self._resize()

    def _durable_state(self) -> Any:
        return {
            "form": "monolithic_deterministic_state_machine",
            "table": {
                key: {
                    "channel": value["channel"],
                    "key": list(value["key"]),
                    "value": list(value["value"]),
                    "strength": int(value["strength"]),
                    "count": int(value["count"]),
                }
                for key, value in sorted(self.table.items())
            },
            "activation": _ACTIVATION,
        }

    def _active_state(self) -> Any:
        return {
            "buffer": list(self.active_buffer),
            "staged": {
                key: {
                    "channel": value["channel"],
                    "key": list(value["key"]),
                    "value": list(value["value"]),
                    "count": int(value["count"]),
                    "teaching": bool(value["teaching"]),
                }
                for key, value in sorted(self.staged.items())
            },
            "transition_counts": dict(sorted(self.transition_counts.items())),
            "last_channel": self.last_channel,
            "last_payload": list(self.last_payload),
            "activation": _ACTIVATION,
        }

    def _restore_durable(self, state: Any) -> None:
        table_raw = state.get("table", {})
        self.table = {
            key: {
                "channel": str(value["channel"]),
                "key": tuple(int(x) for x in value["key"]),
                "value": tuple(int(x) for x in value["value"]),
                "strength": int(value["strength"]),
                "count": int(value["count"]),
            }
            for key, value in table_raw.items()
        }
        self.undo.clear()

    def _restore_active(self, state: Any) -> None:
        self.active_buffer = list(state.get("buffer", []))
        staged_raw = state.get("staged", {})
        self.staged = {
            key: {
                "channel": str(value["channel"]),
                "key": tuple(int(x) for x in value["key"]),
                "value": tuple(int(x) for x in value["value"]),
                "count": int(value["count"]),
                "teaching": bool(value["teaching"]),
            }
            for key, value in staged_raw.items()
        }
        self.transition_counts = {str(key): int(value) for key, value in state.get("transition_counts", {}).items()}
        self.last_channel = state.get("last_channel")
        self.last_payload = tuple(int(x) for x in state.get("last_payload", ()))


def _build_s2(opportunity: Opportunity, **_options: Any) -> S2TaskIndependentMonolithicPersistentCore:
    return S2TaskIndependentMonolithicPersistentCore(
        name=C.CANONICAL_S2_ID,
        mechanism="monolithic_deterministic_task_independent_persistent_core",
        _opportunity=opportunity,
    )


register(C.CANONICAL_S2_ID, _build_s2)


# --------------------------------------------------------------------------
# FR selected kernel — event-sourced minimal persistent core
# --------------------------------------------------------------------------


@dataclass
class FRSelectedKernel(MaterialBase):
    """S2-derived minimal event-sourced monolithic persistent core.

    Durable state is an append-only event log; the readable core is a pure
    deterministic projection of that log. This is the Final Revision selected
    kernel shape, re-hosted on the genesis material interface.
    """

    events: list[dict[str, Any]] = field(default_factory=list)
    projection: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_buffer: list[dict[str, Any]] = field(default_factory=list)
    staged: list[dict[str, Any]] = field(default_factory=list)
    undo_len: dict[str, int] = field(default_factory=dict)

    def _resize(self) -> None:
        self._opportunity.ledger.resize(_bytes_of(self._durable_state()) + _bytes_of(self._active_state()))

    def _reproject(self) -> None:
        projection: dict[str, dict[str, Any]] = {}
        for event in self.events:
            address = str(event["address"])
            projection[address] = {
                "channel": str(event["channel"]),
                "key": tuple(int(x) for x in event["key"]),
                "value": tuple(int(x) for x in event["value"]),
                "sequence": int(event["sequence"]),
            }
        self.projection = projection

    def _transition(self, observation: Observation) -> None:
        payload = tuple(int(x) for x in observation.payload)
        self.active_buffer.append(
            {
                "index": observation.index,
                "channel": observation.channel,
                "payload": payload,
                "teaching": bool(observation.teaching),
                "modality": observation.modality,
            }
        )
        for key, value in _prefix_keys(payload):
            address = _key_of(observation.channel, key)
            self.staged.append(
                {
                    "address": address,
                    "channel": observation.channel,
                    "key": key,
                    "value": value,
                    "teaching": bool(observation.teaching),
                    "obs_index": observation.index,
                }
            )
        self._resize()

    def _answer(self, probe: Probe) -> Answer:
        arity = max(0, int(probe.arity))
        probe_key = tuple(int(x) for x in probe.probe)
        address = _key_of(probe.channel, probe_key)
        if address in self.projection:
            value = _trim(self.projection[address]["value"], arity)
            return Answer(probe_index=probe.index, value=value, confidence=190, abstained=False)
        for entry in self.projection.values():
            if entry["channel"] != probe.channel:
                continue
            stored_key = tuple(entry["key"])
            stored_value = tuple(entry["value"])
            if stored_key == probe_key:
                return Answer(probe_index=probe.index, value=_trim(stored_value, arity), confidence=180, abstained=False)
            full = stored_key + stored_value
            if probe_key and full[: len(probe_key)] == probe_key and len(full) > len(probe_key):
                return Answer(probe_index=probe.index, value=_trim(full[len(probe_key) :], arity), confidence=150, abstained=False)
        for row in reversed(self.active_buffer):
            if row["channel"] == probe.channel:
                payload = tuple(row["payload"])
                if not probe_key or payload[: len(probe_key)] == probe_key:
                    value = payload[len(probe_key) :] if probe_key else payload
                    return Answer(probe_index=probe.index, value=_trim(value if value else payload, arity), confidence=80, abstained=False)
        if arity == 0:
            return Answer(probe_index=probe.index, value=(), confidence=0, abstained=True)
        return Answer(probe_index=probe.index, value=tuple(0 for _ in range(arity)), confidence=0, abstained=True)

    def _propose(self) -> Iterable[Proposal]:
        seen: set[str] = set()
        proposals: list[Proposal] = []
        for entry in reversed(self.staged):
            address = str(entry["address"])
            if address in seen:
                continue
            seen.add(address)
            current = self.projection.get(address)
            value = tuple(int(x) for x in entry["value"])
            if current is not None and tuple(current["value"]) == value:
                continue
            proposals.append(
                Proposal(
                    proposal_id=f"fr:{address}:{entry['obs_index']}:{len(proposals)}",
                    kind="event_append_projection",
                    target=address,
                    delta=value,
                    topology_operation="append_event",
                    trigger="event_sourced_projection",
                    expected_value=1.5 if entry["teaching"] else 1.0,
                    cost_bytes=max(8, _bytes_of(value) + 16),
                )
            )
        return proposals

    def _commit(self, proposal: Proposal) -> None:
        self.undo_len[proposal.proposal_id] = len(self.events)
        # Recover key from staged/projection address encoding.
        key_part = proposal.target.split("|", 1)
        channel = key_part[0]
        key_tokens = key_part[1].split(",") if len(key_part) > 1 and key_part[1] else []
        key = tuple(int(token) for token in key_tokens if token != "")
        event = {
            "sequence": len(self.events) + 1,
            "address": proposal.target,
            "channel": channel,
            "key": key,
            "value": tuple(int(x) for x in proposal.delta),
            "kind": "association",
        }
        self.events.append(event)
        self._reproject()
        self._resize()

    def _rollback(self, receipt: Receipt) -> None:
        prior_len = self.undo_len.pop(receipt.proposal_id, None)
        if prior_len is None:
            return
        self.events = self.events[:prior_len]
        self._reproject()
        self._resize()

    def _durable_state(self) -> Any:
        return {
            "form": "s2_derived_minimal_event_sourced_monolithic_persistent_core",
            "events": [
                {
                    "sequence": int(event["sequence"]),
                    "address": str(event["address"]),
                    "channel": str(event["channel"]),
                    "key": list(event["key"]),
                    "value": list(event["value"]),
                    "kind": str(event["kind"]),
                }
                for event in self.events
            ],
            "projection": {
                key: {
                    "channel": value["channel"],
                    "key": list(value["key"]),
                    "value": list(value["value"]),
                    "sequence": int(value["sequence"]),
                }
                for key, value in sorted(self.projection.items())
            },
            "activation": _ACTIVATION,
        }

    def _active_state(self) -> Any:
        return {
            "buffer": list(self.active_buffer),
            "staged": [
                {
                    "address": item["address"],
                    "channel": item["channel"],
                    "key": list(item["key"]),
                    "value": list(item["value"]),
                    "teaching": bool(item["teaching"]),
                    "obs_index": int(item["obs_index"]),
                }
                for item in self.staged
            ],
            "activation": _ACTIVATION,
        }

    def _restore_durable(self, state: Any) -> None:
        self.events = [
            {
                "sequence": int(event["sequence"]),
                "address": str(event["address"]),
                "channel": str(event["channel"]),
                "key": tuple(int(x) for x in event["key"]),
                "value": tuple(int(x) for x in event["value"]),
                "kind": str(event["kind"]),
            }
            for event in state.get("events", [])
        ]
        self._reproject()
        self.undo_len.clear()

    def _restore_active(self, state: Any) -> None:
        self.active_buffer = list(state.get("buffer", []))
        self.staged = [
            {
                "address": str(item["address"]),
                "channel": str(item["channel"]),
                "key": tuple(int(x) for x in item["key"]),
                "value": tuple(int(x) for x in item["value"]),
                "teaching": bool(item["teaching"]),
                "obs_index": int(item["obs_index"]),
            }
            for item in state.get("staged", [])
        ]


def _build_fr(opportunity: Opportunity, **_options: Any) -> FRSelectedKernel:
    return FRSelectedKernel(
        name="FR_selected_kernel",
        mechanism="s2_derived_minimal_event_sourced_monolithic_persistent_core",
        _opportunity=opportunity,
    )


register("FR_selected_kernel", _build_fr)


# --------------------------------------------------------------------------
# Deprivation baselines
# --------------------------------------------------------------------------


@dataclass
class StaticFrozenField(MaterialBase):
    """Fixed durable field with no plasticity; answers from frozen weights only."""

    weights: list[int] = field(default_factory=list)
    active_trace: list[tuple[int, ...]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = _det_weights("static_frozen_field/v1", 32)

    def _transition(self, observation: Observation) -> None:
        payload = tuple(int(x) for x in observation.payload)
        self.active_trace.append(payload)
        if len(self.active_trace) > 64:
            self.active_trace = self.active_trace[-64:]
        self._opportunity.ledger.resize(_bytes_of(self._durable_state()) + _bytes_of(self._active_state()))

    def _answer(self, probe: Probe) -> Answer:
        arity = max(0, int(probe.arity))
        features = list(probe.probe) if probe.probe else [0]
        # Pad/truncate features to weight length for a fixed linear map.
        if len(features) < len(self.weights):
            features = features + [0] * (len(self.weights) - len(features))
        else:
            features = features[: len(self.weights)]
        total = int(multiplication_light_dot(features, self.weights)["value"])
        if arity == 0:
            return Answer(probe_index=probe.index, value=(), confidence=10, abstained=False)
        value = tuple(native_low_bit_update(0, float((total >> shift) & 3) - 1.0, _QUINARY) for shift in range(arity))
        return Answer(probe_index=probe.index, value=value, confidence=10, abstained=False)

    def _propose(self) -> Iterable[Proposal]:
        return ()

    def _commit(self, proposal: Proposal) -> None:
        return None

    def _rollback(self, receipt: Receipt) -> None:
        return None

    def _durable_state(self) -> Any:
        return {"form": "static_frozen_field", "weights": list(self.weights), "activation": _ACTIVATION}

    def _active_state(self) -> Any:
        return {"trace": [list(item) for item in self.active_trace], "activation": _ACTIVATION}

    def _restore_durable(self, state: Any) -> None:
        self.weights = [int(x) for x in state.get("weights", _det_weights("static_frozen_field/v1", 32))]

    def _restore_active(self, state: Any) -> None:
        self.active_trace = [tuple(int(x) for x in item) for item in state.get("trace", [])]


def _build_static(opportunity: Opportunity, **_options: Any) -> StaticFrozenField:
    return StaticFrozenField(
        name="static_frozen_field",
        mechanism="static_frozen_field_no_plasticity",
        _opportunity=opportunity,
    )


register("static_frozen_field", _build_static)


@dataclass
class ReplayFullHistory(MaterialBase):
    """Answers by replaying the entire observation history; no durable development."""

    history: list[dict[str, Any]] = field(default_factory=list)

    def _transition(self, observation: Observation) -> None:
        self.history.append(
            {
                "index": observation.index,
                "channel": observation.channel,
                "payload": tuple(int(x) for x in observation.payload),
                "teaching": bool(observation.teaching),
                "modality": observation.modality,
            }
        )
        self._opportunity.ledger.resize(_bytes_of(self._durable_state()) + _bytes_of(self._active_state()))

    def _answer(self, probe: Probe) -> Answer:
        arity = max(0, int(probe.arity))
        probe_key = tuple(int(x) for x in probe.probe)
        # Full replay: scan history in order, last exact channel match wins.
        hit: tuple[int, ...] | None = None
        for row in self.history:
            if row["channel"] != probe.channel:
                continue
            payload = tuple(row["payload"])
            if not probe_key:
                hit = payload
            elif payload[: len(probe_key)] == probe_key:
                hit = payload[len(probe_key) :] if len(payload) > len(probe_key) else payload
            elif _hamming(payload, probe_key) <= max(1, len(probe_key) // 4):
                hit = payload
        if hit is None:
            return Answer(probe_index=probe.index, value=tuple(0 for _ in range(arity)) if arity else (), confidence=0, abstained=True)
        return Answer(probe_index=probe.index, value=_trim(hit, arity), confidence=70, abstained=False)

    def _propose(self) -> Iterable[Proposal]:
        return ()

    def _commit(self, proposal: Proposal) -> None:
        return None

    def _rollback(self, receipt: Receipt) -> None:
        return None

    def _durable_state(self) -> Any:
        return {"form": "replay_full_history", "durable": "none", "activation": _ACTIVATION}

    def _active_state(self) -> Any:
        return {
            "history": [
                {
                    "index": row["index"],
                    "channel": row["channel"],
                    "payload": list(row["payload"]),
                    "teaching": row["teaching"],
                    "modality": row["modality"],
                }
                for row in self.history
            ],
            "activation": _ACTIVATION,
        }

    def _restore_durable(self, state: Any) -> None:
        return None

    def _restore_active(self, state: Any) -> None:
        self.history = [
            {
                "index": int(row["index"]),
                "channel": str(row["channel"]),
                "payload": tuple(int(x) for x in row["payload"]),
                "teaching": bool(row["teaching"]),
                "modality": str(row["modality"]),
            }
            for row in state.get("history", [])
        ]


def _build_replay(opportunity: Opportunity, **_options: Any) -> ReplayFullHistory:
    return ReplayFullHistory(
        name="replay_full_history",
        mechanism="replay_full_observation_history",
        _opportunity=opportunity,
    )


register("replay_full_history", _build_replay)


@dataclass
class SummaryReplay(MaterialBase):
    """Answers from a bounded summary of history; no durable development."""

    summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_seen: int = 0
    capacity: int = 16

    def _transition(self, observation: Observation) -> None:
        self.total_seen += 1
        payload = tuple(int(x) for x in observation.payload)
        self.summary[observation.channel] = {
            "payload": payload,
            "count": int(self.summary.get(observation.channel, {}).get("count", 0)) + 1,
            "teaching": bool(observation.teaching) or bool(self.summary.get(observation.channel, {}).get("teaching")),
            "last_index": observation.index,
            "checksum": int(sum(payload) + observation.index) & 0xFFFF,
        }
        if len(self.summary) > self.capacity:
            ordered = sorted(self.summary.items(), key=lambda item: int(item[1]["last_index"]))
            for key, _ in ordered[: len(self.summary) - self.capacity]:
                self.summary.pop(key, None)
        self._opportunity.ledger.resize(_bytes_of(self._durable_state()) + _bytes_of(self._active_state()))

    def _answer(self, probe: Probe) -> Answer:
        arity = max(0, int(probe.arity))
        entry = self.summary.get(probe.channel)
        if entry is None:
            return Answer(probe_index=probe.index, value=tuple(0 for _ in range(arity)) if arity else (), confidence=0, abstained=True)
        payload = tuple(entry["payload"])
        probe_key = tuple(int(x) for x in probe.probe)
        if probe_key and payload[: len(probe_key)] == probe_key and len(payload) > len(probe_key):
            value = payload[len(probe_key) :]
        else:
            value = payload
        return Answer(probe_index=probe.index, value=_trim(value, arity), confidence=60, abstained=False)

    def _propose(self) -> Iterable[Proposal]:
        return ()

    def _commit(self, proposal: Proposal) -> None:
        return None

    def _rollback(self, receipt: Receipt) -> None:
        return None

    def _durable_state(self) -> Any:
        return {"form": "summary_replay", "durable": "none", "activation": _ACTIVATION}

    def _active_state(self) -> Any:
        return {
            "summary": {
                key: {
                    "payload": list(value["payload"]),
                    "count": int(value["count"]),
                    "teaching": bool(value["teaching"]),
                    "last_index": int(value["last_index"]),
                    "checksum": int(value["checksum"]),
                }
                for key, value in sorted(self.summary.items())
            },
            "total_seen": self.total_seen,
            "activation": _ACTIVATION,
        }

    def _restore_durable(self, state: Any) -> None:
        return None

    def _restore_active(self, state: Any) -> None:
        self.total_seen = int(state.get("total_seen", 0))
        self.summary = {
            str(key): {
                "payload": tuple(int(x) for x in value["payload"]),
                "count": int(value["count"]),
                "teaching": bool(value["teaching"]),
                "last_index": int(value["last_index"]),
                "checksum": int(value["checksum"]),
            }
            for key, value in state.get("summary", {}).items()
        }


def _build_summary(opportunity: Opportunity, **_options: Any) -> SummaryReplay:
    return SummaryReplay(
        name="summary_replay",
        mechanism="bounded_summary_replay",
        _opportunity=opportunity,
    )


register("summary_replay", _build_summary)


@dataclass
class RetrievalOnly(MaterialBase):
    """Nearest-match retrieval over a bounded store; no developmental rewrites."""

    store: dict[str, tuple[int, ...]] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    capacity: int = 32

    def _put(self, key: str, value: tuple[int, ...]) -> None:
        if key in self.order:
            self.order.remove(key)
        self.order.append(key)
        self.store[key] = value
        while len(self.order) > self.capacity:
            evicted = self.order.pop(0)
            self.store.pop(evicted, None)

    def _transition(self, observation: Observation) -> None:
        payload = tuple(int(x) for x in observation.payload)
        self._put(_key_of(observation.channel, ()), payload)
        if payload:
            self._put(_key_of(observation.channel, payload[:1]), payload)
        if observation.teaching and len(payload) > 1:
            mid = len(payload) // 2
            self._put(_key_of(observation.channel, payload[:mid]), payload[mid:])
        self._opportunity.ledger.resize(_bytes_of(self._durable_state()) + _bytes_of(self._active_state()))

    def _answer(self, probe: Probe) -> Answer:
        arity = max(0, int(probe.arity))
        probe_key = tuple(int(x) for x in probe.probe)
        address = _key_of(probe.channel, probe_key)
        if address in self.store:
            return Answer(probe_index=probe.index, value=_trim(self.store[address], arity), confidence=100, abstained=False)
        channel_hits = {key: value for key, value in self.store.items() if key.startswith(f"{probe.channel}|")}
        nearest = _nearest(probe_key if probe_key else (0,), channel_hits) if channel_hits else None
        if nearest is None:
            nearest = _nearest(probe_key if probe_key else (0,), self.store) if self.store else None
        if nearest is None:
            return Answer(probe_index=probe.index, value=tuple(0 for _ in range(arity)) if arity else (), confidence=0, abstained=True)
        return Answer(probe_index=probe.index, value=_trim(nearest[1], arity), confidence=55, abstained=False)

    def _propose(self) -> Iterable[Proposal]:
        return ()

    def _commit(self, proposal: Proposal) -> None:
        return None

    def _rollback(self, receipt: Receipt) -> None:
        return None

    def _durable_state(self) -> Any:
        return {"form": "retrieval_only", "durable": "none", "activation": _ACTIVATION}

    def _active_state(self) -> Any:
        return {
            "store": {key: list(self.store[key]) for key in self.order if key in self.store},
            "order": list(self.order),
            "activation": _ACTIVATION,
        }

    def _restore_durable(self, state: Any) -> None:
        return None

    def _restore_active(self, state: Any) -> None:
        self.order = [str(key) for key in state.get("order", [])]
        store_raw = state.get("store", {})
        self.store = {str(key): tuple(int(x) for x in value) for key, value in store_raw.items()}


def _build_retrieval(opportunity: Opportunity, **_options: Any) -> RetrievalOnly:
    return RetrievalOnly(
        name="retrieval_only",
        mechanism="nearest_match_retrieval_only",
        _opportunity=opportunity,
    )


register("retrieval_only", _build_retrieval)


@dataclass
class PrecompiledProcedureBank(MaterialBase):
    """Fixed bank of procedures; no new procedures are admitted."""

    procedures: dict[str, str] = field(default_factory=dict)
    active_buffer: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.procedures:
            self.procedures = {
                "identity": "identity",
                "prefix": "prefix",
                "sum_mod": "sum_mod",
                "delta": "delta",
                "const_zero": "const_zero",
            }

    def _transition(self, observation: Observation) -> None:
        self.active_buffer.append(
            {
                "channel": observation.channel,
                "payload": tuple(int(x) for x in observation.payload),
                "teaching": bool(observation.teaching),
            }
        )
        if len(self.active_buffer) > 48:
            self.active_buffer = self.active_buffer[-48:]
        self._opportunity.ledger.resize(_bytes_of(self._durable_state()) + _bytes_of(self._active_state()))

    def _select_procedure(self, probe: Probe) -> str:
        digest = hashlib.sha256(f"{probe.channel}:{probe.family}".encode()).digest()
        names = sorted(self.procedures)
        return names[digest[0] % len(names)]

    def _run(self, name: str, probe: Probe) -> tuple[int, ...]:
        arity = max(0, int(probe.arity))
        items = tuple(int(x) for x in probe.probe)
        if name == "identity":
            return _trim(items, arity)
        if name == "prefix":
            return _trim(items[: max(1, len(items) // 2)] if items else (0,), arity)
        if name == "sum_mod":
            total = sum(items) if items else 0
            return _trim(tuple((total + index) % 5 - 2 for index in range(max(arity, 1))), arity)
        if name == "delta":
            if len(items) >= 2:
                diffs = tuple(items[index] - items[index - 1] for index in range(1, len(items)))
                return _trim(diffs, arity)
            return _trim(items, arity)
        return tuple(0 for _ in range(arity)) if arity else ()

    def _answer(self, probe: Probe) -> Answer:
        name = self._select_procedure(probe)
        # Prefer a channel-matched active payload when the fixed procedure is const_zero and buffer has data.
        for row in reversed(self.active_buffer):
            if row["channel"] == probe.channel and name in {"identity", "prefix"}:
                payload = tuple(row["payload"])
                probe_key = tuple(int(x) for x in probe.probe)
                if not probe_key or payload[: len(probe_key)] == probe_key:
                    value = payload[len(probe_key) :] if probe_key else payload
                    return Answer(probe_index=probe.index, value=_trim(value if value else payload, probe.arity), confidence=50, abstained=False)
        value = self._run(name, probe)
        return Answer(probe_index=probe.index, value=value, confidence=30, abstained=False)

    def _propose(self) -> Iterable[Proposal]:
        return ()

    def _commit(self, proposal: Proposal) -> None:
        return None

    def _rollback(self, receipt: Receipt) -> None:
        return None

    def _durable_state(self) -> Any:
        return {
            "form": "precompiled_procedure_bank",
            "procedures": dict(sorted(self.procedures.items())),
            "activation": _ACTIVATION,
        }

    def _active_state(self) -> Any:
        return {
            "buffer": [
                {"channel": row["channel"], "payload": list(row["payload"]), "teaching": row["teaching"]}
                for row in self.active_buffer
            ],
            "activation": _ACTIVATION,
        }

    def _restore_durable(self, state: Any) -> None:
        self.procedures = {str(key): str(value) for key, value in state.get("procedures", {}).items()}

    def _restore_active(self, state: Any) -> None:
        self.active_buffer = [
            {
                "channel": str(row["channel"]),
                "payload": tuple(int(x) for x in row["payload"]),
                "teaching": bool(row["teaching"]),
            }
            for row in state.get("buffer", [])
        ]


def _build_procedure_bank(opportunity: Opportunity, **_options: Any) -> PrecompiledProcedureBank:
    return PrecompiledProcedureBank(
        name="precompiled_procedure_bank",
        mechanism="fixed_precompiled_procedure_bank",
        _opportunity=opportunity,
    )


register("precompiled_procedure_bank", _build_procedure_bank)


@dataclass
class WrongHistoryPlastic(MaterialBase):
    """Fully plastic core that accepts whatever history the harness supplies.

    Keying deliberately differs from S2 so the mechanism is distinct under an
    identical stream; the harness alone supplies wrong history. This material
    never reshuffles or rewrites its input order.
    """

    table: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_buffer: list[dict[str, Any]] = field(default_factory=list)
    staged: dict[str, dict[str, Any]] = field(default_factory=dict)
    undo: dict[str, dict[str, Any] | None] = field(default_factory=dict)

    def _address(self, channel: str, key: Sequence[int]) -> str:
        # Inverted channel token + reversed key: distinct from S2 under the same stream.
        inverted = channel[::-1]
        reversed_key = tuple(reversed(tuple(int(x) for x in key)))
        return f"wh:{inverted}|{','.join(str(x) for x in reversed_key)}"

    def _resize(self) -> None:
        self._opportunity.ledger.resize(_bytes_of(self._durable_state()) + _bytes_of(self._active_state()))

    def _transition(self, observation: Observation) -> None:
        payload = tuple(int(x) for x in observation.payload)
        self.active_buffer.append({"channel": observation.channel, "payload": payload, "teaching": bool(observation.teaching)})
        for key, value in _prefix_keys(payload):
            address = self._address(observation.channel, key)
            entry = self.staged.get(address)
            self.staged[address] = {
                "channel": observation.channel,
                "key": key,
                "value": tuple(reversed(value)) if value else value,
                "count": int(entry["count"]) + 1 if entry else 1,
            }
        self._resize()

    def _answer(self, probe: Probe) -> Answer:
        arity = max(0, int(probe.arity))
        probe_key = tuple(int(x) for x in probe.probe)
        address = self._address(probe.channel, probe_key)
        if address in self.table:
            # Stored values were reversed at stage time; reverse back for emission.
            raw = tuple(self.table[address]["value"])
            value = tuple(reversed(raw)) if raw else raw
            return Answer(probe_index=probe.index, value=_trim(value, arity), confidence=170, abstained=False)
        for row in reversed(self.active_buffer):
            if row["channel"] == probe.channel:
                payload = tuple(row["payload"])
                if not probe_key or payload[: len(probe_key)] == probe_key:
                    value = payload[len(probe_key) :] if probe_key else payload
                    return Answer(probe_index=probe.index, value=_trim(value if value else payload, arity), confidence=85, abstained=False)
        return Answer(probe_index=probe.index, value=tuple(0 for _ in range(arity)) if arity else (), confidence=0, abstained=True)

    def _propose(self) -> Iterable[Proposal]:
        proposals: list[Proposal] = []
        for address, entry in sorted(self.staged.items(), key=lambda item: (-int(item[1]["count"]), item[0])):
            current = self.table.get(address)
            value = tuple(int(x) for x in entry["value"])
            if current is not None and tuple(current["value"]) == value:
                continue
            proposals.append(
                Proposal(
                    proposal_id=f"wh:{address}:{entry['count']}:{len(proposals)}",
                    kind="wrong_history_association_write",
                    target=address,
                    delta=value,
                    trigger="plastic_on_supplied_history",
                    expected_value=float(entry["count"]),
                    cost_bytes=max(8, _bytes_of(value)),
                )
            )
        return proposals

    def _commit(self, proposal: Proposal) -> None:
        prior = self.table.get(proposal.target)
        self.undo[proposal.proposal_id] = None if prior is None else {
            "value": tuple(prior["value"]),
            "count": int(prior["count"]),
            "channel": str(prior["channel"]),
            "key": tuple(prior["key"]),
        }
        self.table[proposal.target] = {
            "channel": proposal.target,
            "key": (),
            "value": tuple(int(x) for x in proposal.delta),
            "count": 1 if prior is None else int(prior["count"]) + 1,
        }
        self._resize()

    def _rollback(self, receipt: Receipt) -> None:
        prior = self.undo.pop(receipt.proposal_id, None)
        if prior is None:
            self.table.pop(receipt.target, None)
        else:
            self.table[receipt.target] = {
                "channel": prior["channel"],
                "key": tuple(prior["key"]),
                "value": tuple(prior["value"]),
                "count": int(prior["count"]),
            }
        self._resize()

    def _durable_state(self) -> Any:
        return {
            "form": "wrong_history_plastic",
            "table": {
                key: {"value": list(value["value"]), "count": int(value["count"])}
                for key, value in sorted(self.table.items())
            },
            "activation": _ACTIVATION,
        }

    def _active_state(self) -> Any:
        return {
            "buffer": [{"channel": row["channel"], "payload": list(row["payload"]), "teaching": row["teaching"]} for row in self.active_buffer],
            "staged": {
                key: {"value": list(value["value"]), "count": int(value["count"]), "channel": value["channel"], "key": list(value["key"])}
                for key, value in sorted(self.staged.items())
            },
            "activation": _ACTIVATION,
        }

    def _restore_durable(self, state: Any) -> None:
        self.table = {
            str(key): {
                "channel": str(key),
                "key": (),
                "value": tuple(int(x) for x in value["value"]),
                "count": int(value["count"]),
            }
            for key, value in state.get("table", {}).items()
        }
        self.undo.clear()

    def _restore_active(self, state: Any) -> None:
        self.active_buffer = [
            {"channel": str(row["channel"]), "payload": tuple(int(x) for x in row["payload"]), "teaching": bool(row["teaching"])}
            for row in state.get("buffer", [])
        ]
        self.staged = {
            str(key): {
                "channel": str(value["channel"]),
                "key": tuple(int(x) for x in value["key"]),
                "value": tuple(int(x) for x in value["value"]),
                "count": int(value["count"]),
            }
            for key, value in state.get("staged", {}).items()
        }


def _build_wrong_history(opportunity: Opportunity, **_options: Any) -> WrongHistoryPlastic:
    return WrongHistoryPlastic(
        name="wrong_history_plastic",
        mechanism="plastic_core_on_harness_supplied_wrong_history",
        _opportunity=opportunity,
    )


register("wrong_history_plastic", _build_wrong_history)


@dataclass
class ShuffledHistoryPlastic(MaterialBase):
    """Fully plastic core; order-invariant multiset keys make the mechanism distinct.

    The harness alone supplies shuffled order. This material does not reshuffle.
    """

    table: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_buffer: list[dict[str, Any]] = field(default_factory=list)
    staged: dict[str, dict[str, Any]] = field(default_factory=dict)
    undo: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    bag: dict[str, int] = field(default_factory=dict)

    def _address(self, channel: str, payload: Sequence[int]) -> str:
        multiset = ",".join(str(x) for x in sorted(int(v) for v in payload))
        return f"sh:{channel}#{multiset}"

    def _resize(self) -> None:
        self._opportunity.ledger.resize(_bytes_of(self._durable_state()) + _bytes_of(self._active_state()))

    def _transition(self, observation: Observation) -> None:
        payload = tuple(int(x) for x in observation.payload)
        self.active_buffer.append({"channel": observation.channel, "payload": payload, "teaching": bool(observation.teaching)})
        token = f"{observation.channel}:{sum(payload)}"
        self.bag[token] = self.bag.get(token, 0) + 1
        address = self._address(observation.channel, payload)
        entry = self.staged.get(address)
        self.staged[address] = {
            "channel": observation.channel,
            "payload": payload,
            "value": payload,
            "count": int(entry["count"]) + 1 if entry else 1,
        }
        # Order-invariant co-occurrence: bag snapshot as a secondary key.
        bag_sig = tuple(sorted((key, count) for key, count in self.bag.items()))
        bag_address = f"sh:bag|{hashlib.sha256(repr(bag_sig).encode()).hexdigest()[:16]}"
        self.staged[bag_address] = {
            "channel": "bag",
            "payload": payload,
            "value": payload,
            "count": sum(self.bag.values()),
        }
        self._resize()

    def _answer(self, probe: Probe) -> Answer:
        arity = max(0, int(probe.arity))
        probe_key = tuple(int(x) for x in probe.probe)
        address = self._address(probe.channel, probe_key)
        if address in self.table:
            return Answer(probe_index=probe.index, value=_trim(self.table[address]["value"], arity), confidence=165, abstained=False)
        # Multiset match against durable payloads.
        probe_multi = tuple(sorted(probe_key))
        for entry in self.table.values():
            if entry["channel"] == probe.channel and tuple(sorted(entry["value"])) == probe_multi:
                return Answer(probe_index=probe.index, value=_trim(entry["value"], arity), confidence=120, abstained=False)
        for row in reversed(self.active_buffer):
            if row["channel"] == probe.channel:
                return Answer(probe_index=probe.index, value=_trim(row["payload"], arity), confidence=80, abstained=False)
        return Answer(probe_index=probe.index, value=tuple(0 for _ in range(arity)) if arity else (), confidence=0, abstained=True)

    def _propose(self) -> Iterable[Proposal]:
        proposals: list[Proposal] = []
        for address, entry in sorted(self.staged.items(), key=lambda item: (-int(item[1]["count"]), item[0])):
            current = self.table.get(address)
            value = tuple(int(x) for x in entry["value"])
            if current is not None and tuple(current["value"]) == value:
                continue
            proposals.append(
                Proposal(
                    proposal_id=f"sh:{address}:{entry['count']}:{len(proposals)}",
                    kind="shuffled_history_multiset_write",
                    target=address,
                    delta=value,
                    trigger="plastic_on_supplied_order",
                    expected_value=float(entry["count"]),
                    cost_bytes=max(8, _bytes_of(value)),
                )
            )
        return proposals

    def _commit(self, proposal: Proposal) -> None:
        prior = self.table.get(proposal.target)
        self.undo[proposal.proposal_id] = None if prior is None else {
            "channel": str(prior["channel"]),
            "value": tuple(prior["value"]),
            "count": int(prior["count"]),
        }
        channel = proposal.target.split(":", 1)[-1].split("#", 1)[0].split("|", 1)[0]
        if channel.startswith("sh"):
            channel = "bag"
        self.table[proposal.target] = {
            "channel": channel if not proposal.target.startswith("sh:bag") else "bag",
            "value": tuple(int(x) for x in proposal.delta),
            "count": 1 if prior is None else int(prior["count"]) + 1,
        }
        self._resize()

    def _rollback(self, receipt: Receipt) -> None:
        prior = self.undo.pop(receipt.proposal_id, None)
        if prior is None:
            self.table.pop(receipt.target, None)
        else:
            self.table[receipt.target] = {
                "channel": prior["channel"],
                "value": tuple(prior["value"]),
                "count": int(prior["count"]),
            }
        self._resize()

    def _durable_state(self) -> Any:
        return {
            "form": "shuffled_history_plastic",
            "table": {
                key: {"channel": value["channel"], "value": list(value["value"]), "count": int(value["count"])}
                for key, value in sorted(self.table.items())
            },
            "activation": _ACTIVATION,
        }

    def _active_state(self) -> Any:
        return {
            "buffer": [{"channel": row["channel"], "payload": list(row["payload"]), "teaching": row["teaching"]} for row in self.active_buffer],
            "staged": {
                key: {"channel": value["channel"], "value": list(value["value"]), "count": int(value["count"]), "payload": list(value["payload"])}
                for key, value in sorted(self.staged.items())
            },
            "bag": dict(sorted(self.bag.items())),
            "activation": _ACTIVATION,
        }

    def _restore_durable(self, state: Any) -> None:
        self.table = {
            str(key): {
                "channel": str(value["channel"]),
                "value": tuple(int(x) for x in value["value"]),
                "count": int(value["count"]),
            }
            for key, value in state.get("table", {}).items()
        }
        self.undo.clear()

    def _restore_active(self, state: Any) -> None:
        self.active_buffer = [
            {"channel": str(row["channel"]), "payload": tuple(int(x) for x in row["payload"]), "teaching": bool(row["teaching"])}
            for row in state.get("buffer", [])
        ]
        self.staged = {
            str(key): {
                "channel": str(value["channel"]),
                "payload": tuple(int(x) for x in value["payload"]),
                "value": tuple(int(x) for x in value["value"]),
                "count": int(value["count"]),
            }
            for key, value in state.get("staged", {}).items()
        }
        self.bag = {str(key): int(value) for key, value in state.get("bag", {}).items()}


def _build_shuffled(opportunity: Opportunity, **_options: Any) -> ShuffledHistoryPlastic:
    return ShuffledHistoryPlastic(
        name="shuffled_history_plastic",
        mechanism="plastic_core_on_harness_supplied_shuffled_history",
        _opportunity=opportunity,
    )


register("shuffled_history_plastic", _build_shuffled)


@dataclass
class RandomGrowthPlastic(MaterialBase):
    """Grows durable structure at random rather than on verified value."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_buffer: list[dict[str, Any]] = field(default_factory=list)
    pending_seeds: list[str] = field(default_factory=list)
    undo: dict[str, str | None] = field(default_factory=dict)
    rng_state: int = 0xC0FFEE

    def _next_rand(self) -> int:
        # Deterministic LCG; not crypto. Used only to grow structure without value signals.
        self.rng_state = (1_103_515_245 * self.rng_state + 12_345) & 0x7FFFFFFF
        return self.rng_state

    def _resize(self) -> None:
        self._opportunity.ledger.resize(_bytes_of(self._durable_state()) + _bytes_of(self._active_state()))

    def _transition(self, observation: Observation) -> None:
        payload = tuple(int(x) for x in observation.payload)
        self.active_buffer.append({"channel": observation.channel, "payload": payload, "index": observation.index})
        seed = hashlib.sha256(f"{observation.index}:{observation.channel}:{payload}".encode()).hexdigest()[:12]
        self.pending_seeds.append(seed)
        self._resize()

    def _answer(self, probe: Probe) -> Answer:
        arity = max(0, int(probe.arity))
        if not self.nodes:
            for row in reversed(self.active_buffer):
                if row["channel"] == probe.channel:
                    return Answer(probe_index=probe.index, value=_trim(row["payload"], arity), confidence=20, abstained=False)
            return Answer(probe_index=probe.index, value=tuple(0 for _ in range(arity)) if arity else (), confidence=0, abstained=True)
        # Pick a node by probe hash — not by verified utility.
        names = sorted(self.nodes)
        index = int(hashlib.sha256(f"{probe.channel}:{probe.probe}".encode()).hexdigest()[:8], 16) % len(names)
        node = self.nodes[names[index]]
        return Answer(probe_index=probe.index, value=_trim(node["vector"], arity), confidence=25, abstained=False)

    def _propose(self) -> Iterable[Proposal]:
        proposals: list[Proposal] = []
        for seed in self.pending_seeds:
            if any(node.get("seed") == seed for node in self.nodes.values()):
                continue
            rumble = self._next_rand()
            vector = tuple((rumble >> shift) % 5 - 2 for shift in range(0, 20, 4))
            target = f"rand:{seed}"
            proposals.append(
                Proposal(
                    proposal_id=f"rg:{seed}:{len(proposals)}",
                    kind="random_structure_growth",
                    target=target,
                    delta=vector,
                    topology_operation="allocate",
                    trigger="random_growth",
                    expected_value=0.0,
                    cost_bytes=max(8, _bytes_of(vector)),
                )
            )
        return proposals

    def _commit(self, proposal: Proposal) -> None:
        prior_present = proposal.target in self.nodes
        self.undo[proposal.proposal_id] = proposal.target if prior_present else None
        seed = proposal.target.split(":", 1)[-1]
        self.nodes[proposal.target] = {
            "seed": seed,
            "vector": tuple(int(x) for x in proposal.delta),
            "stamp": self._next_rand(),
        }
        self._resize()

    def _rollback(self, receipt: Receipt) -> None:
        marker = self.undo.pop(receipt.proposal_id, "missing")
        if marker is None or marker == "missing":
            self.nodes.pop(receipt.target, None)
        self._resize()

    def _durable_state(self) -> Any:
        return {
            "form": "random_growth_plastic",
            "nodes": {
                key: {"seed": value["seed"], "vector": list(value["vector"]), "stamp": int(value["stamp"])}
                for key, value in sorted(self.nodes.items())
            },
            "activation": _ACTIVATION,
        }

    def _active_state(self) -> Any:
        return {
            "buffer": [{"channel": row["channel"], "payload": list(row["payload"]), "index": row["index"]} for row in self.active_buffer],
            "pending_seeds": list(self.pending_seeds),
            "rng_state": self.rng_state,
            "activation": _ACTIVATION,
        }

    def _restore_durable(self, state: Any) -> None:
        self.nodes = {
            str(key): {
                "seed": str(value["seed"]),
                "vector": tuple(int(x) for x in value["vector"]),
                "stamp": int(value["stamp"]),
            }
            for key, value in state.get("nodes", {}).items()
        }
        self.undo.clear()

    def _restore_active(self, state: Any) -> None:
        self.active_buffer = [
            {"channel": str(row["channel"]), "payload": tuple(int(x) for x in row["payload"]), "index": int(row["index"])}
            for row in state.get("buffer", [])
        ]
        self.pending_seeds = [str(item) for item in state.get("pending_seeds", [])]
        self.rng_state = int(state.get("rng_state", 0xC0FFEE))


def _build_random_growth(opportunity: Opportunity, **_options: Any) -> RandomGrowthPlastic:
    return RandomGrowthPlastic(
        name="random_growth_plastic",
        mechanism="random_structure_growth_without_verified_value",
        _opportunity=opportunity,
    )


register("random_growth_plastic", _build_random_growth)


@dataclass
class RecordStoreNull(MaterialBase):
    """Append-only record store that copies observed labelled fields and nothing else.

    Must score at chance on development measures. It can only answer when the
    probe target already appeared as a labelled field in its input.
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    labelled_fields: dict[str, tuple[int, ...]] = field(default_factory=dict)
    active_rows: list[dict[str, Any]] = field(default_factory=list)
    staged_fields: dict[str, tuple[int, ...]] = field(default_factory=dict)
    undo: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _resize(self) -> None:
        self._opportunity.ledger.resize(_bytes_of(self._durable_state()) + _bytes_of(self._active_state()))

    def _transition(self, observation: Observation) -> None:
        payload = tuple(int(x) for x in observation.payload)
        row = {
            "index": observation.index,
            "channel": observation.channel,
            "payload": payload,
            "teaching": bool(observation.teaching),
            "modality": observation.modality,
        }
        self.active_rows.append(row)
        # Labelled fields: teaching events, and explicit field-shaped channels.
        is_labelled = bool(observation.teaching) or observation.channel.startswith("field:") or observation.channel.startswith("label:")
        if is_labelled:
            field_name = observation.channel
            if observation.channel.startswith("field:"):
                field_name = observation.channel[len("field:") :]
            elif observation.channel.startswith("label:"):
                field_name = observation.channel[len("label:") :]
            self.staged_fields[field_name] = payload
            self.staged_fields[observation.channel] = payload
        self._resize()

    def _field_present(self, target: str) -> bool:
        if target in self.labelled_fields or target in self.staged_fields:
            return True
        if f"field:{target}" in self.labelled_fields or f"label:{target}" in self.labelled_fields:
            return True
        return f"field:{target}" in self.staged_fields or f"label:{target}" in self.staged_fields

    def _lookup_field(self, target: str) -> tuple[int, ...] | None:
        for key in (target, f"field:{target}", f"label:{target}"):
            if key in self.labelled_fields:
                return self.labelled_fields[key]
            if key in self.staged_fields:
                return self.staged_fields[key]
        return None

    def _answer(self, probe: Probe) -> Answer:
        arity = max(0, int(probe.arity))
        # Targets: probe.channel and, when present, a field name encoded in the probe family.
        targets = [probe.channel]
        if probe.family:
            targets.append(probe.family)
        if probe.probe:
            # Numeric probes are not field names; only channel/family name a field.
            pass
        for target in targets:
            if not self._field_present(target):
                continue
            value = self._lookup_field(target)
            if value is None:
                continue
            return Answer(probe_index=probe.index, value=_trim(value, arity), confidence=100, abstained=False)
        # No labelled field for this target ever appeared: abstain. Do not invent.
        return Answer(probe_index=probe.index, value=tuple(0 for _ in range(arity)) if arity else (), confidence=0, abstained=True)

    def _propose(self) -> Iterable[Proposal]:
        proposals: list[Proposal] = []
        for name, value in sorted(self.staged_fields.items()):
            if name in self.labelled_fields and self.labelled_fields[name] == value:
                continue
            proposals.append(
                Proposal(
                    proposal_id=f"rs:{name}:{len(proposals)}",
                    kind="record_append",
                    target=name,
                    delta=value,
                    trigger="copy_observed_labelled_field",
                    expected_value=0.0,
                    cost_bytes=max(4, _bytes_of(value)),
                )
            )
        return proposals

    def _commit(self, proposal: Proposal) -> None:
        prior = self.labelled_fields.get(proposal.target)
        self.undo[proposal.proposal_id] = {
            "had": prior is not None,
            "value": None if prior is None else list(prior),
        }
        value = tuple(int(x) for x in proposal.delta)
        self.labelled_fields[proposal.target] = value
        self.records.append({"field": proposal.target, "value": value, "sequence": len(self.records) + 1})
        self._resize()

    def _rollback(self, receipt: Receipt) -> None:
        prior = self.undo.pop(receipt.proposal_id, None)
        if prior is None:
            return
        if not prior["had"]:
            self.labelled_fields.pop(receipt.target, None)
        else:
            self.labelled_fields[receipt.target] = tuple(int(x) for x in prior["value"])
        if self.records and self.records[-1]["field"] == receipt.target:
            self.records.pop()
        self._resize()

    def _durable_state(self) -> Any:
        return {
            "form": "record_store_null",
            "records": [
                {"field": row["field"], "value": list(row["value"]), "sequence": int(row["sequence"])}
                for row in self.records
            ],
            "labelled_fields": {key: list(value) for key, value in sorted(self.labelled_fields.items())},
            "activation": _ACTIVATION,
        }

    def _active_state(self) -> Any:
        return {
            "rows": [
                {
                    "index": row["index"],
                    "channel": row["channel"],
                    "payload": list(row["payload"]),
                    "teaching": row["teaching"],
                    "modality": row["modality"],
                }
                for row in self.active_rows
            ],
            "staged_fields": {key: list(value) for key, value in sorted(self.staged_fields.items())},
            "activation": _ACTIVATION,
        }

    def _restore_durable(self, state: Any) -> None:
        self.records = [
            {
                "field": str(row["field"]),
                "value": tuple(int(x) for x in row["value"]),
                "sequence": int(row["sequence"]),
            }
            for row in state.get("records", [])
        ]
        self.labelled_fields = {
            str(key): tuple(int(x) for x in value) for key, value in state.get("labelled_fields", {}).items()
        }
        self.undo.clear()

    def _restore_active(self, state: Any) -> None:
        self.active_rows = [
            {
                "index": int(row["index"]),
                "channel": str(row["channel"]),
                "payload": tuple(int(x) for x in row["payload"]),
                "teaching": bool(row["teaching"]),
                "modality": str(row["modality"]),
            }
            for row in state.get("rows", [])
        ]
        self.staged_fields = {
            str(key): tuple(int(x) for x in value) for key, value in state.get("staged_fields", {}).items()
        }


def _build_record_store(opportunity: Opportunity, **_options: Any) -> RecordStoreNull:
    return RecordStoreNull(
        name="record_store_null",
        mechanism="append_only_labelled_field_record_store",
        _opportunity=opportunity,
    )


register("record_store_null", _build_record_store)


@dataclass
class OracleMaterial(MaterialBase):
    """Upper reference that may use harness-supplied generating structure."""

    generating_structure: dict[str, tuple[int, ...]] = field(default_factory=dict)
    model: dict[str, tuple[int, ...]] = field(default_factory=dict)
    active_buffer: list[dict[str, Any]] = field(default_factory=list)
    staged: dict[str, tuple[int, ...]] = field(default_factory=dict)
    undo: dict[str, tuple[int, ...] | None] = field(default_factory=dict)

    def _structure_key(self, family: str, channel: str, probe: Sequence[int]) -> str:
        return f"{family}|{channel}|{','.join(str(int(x)) for x in probe)}"

    def _resize(self) -> None:
        self._opportunity.ledger.resize(_bytes_of(self._durable_state()) + _bytes_of(self._active_state()))

    def _transition(self, observation: Observation) -> None:
        payload = tuple(int(x) for x in observation.payload)
        self.active_buffer.append(
            {
                "channel": observation.channel,
                "payload": payload,
                "teaching": bool(observation.teaching),
                "index": observation.index,
            }
        )
        # Oracle stages a complete generative copy of everything it sees plus structure.
        self.staged[_key_of(observation.channel, ())] = payload
        for key, value in _prefix_keys(payload):
            self.staged[_key_of(observation.channel, key)] = value
        for structure_key, structure_value in self.generating_structure.items():
            self.staged[f"structure:{structure_key}"] = tuple(int(x) for x in structure_value)
        self._resize()

    def _answer(self, probe: Probe) -> Answer:
        arity = max(0, int(probe.arity))
        structure_key = self._structure_key(probe.family, probe.channel, probe.probe)
        if structure_key in self.generating_structure:
            return Answer(
                probe_index=probe.index,
                value=_trim(self.generating_structure[structure_key], arity),
                confidence=255,
                abstained=False,
            )
        alt = f"{probe.channel}|{','.join(str(int(x)) for x in probe.probe)}"
        if alt in self.generating_structure:
            return Answer(probe_index=probe.index, value=_trim(self.generating_structure[alt], arity), confidence=255, abstained=False)
        address = _key_of(probe.channel, probe.probe)
        if address in self.model:
            return Answer(probe_index=probe.index, value=_trim(self.model[address], arity), confidence=240, abstained=False)
        if f"structure:{structure_key}" in self.model:
            return Answer(probe_index=probe.index, value=_trim(self.model[f"structure:{structure_key}"], arity), confidence=240, abstained=False)
        for row in reversed(self.active_buffer):
            if row["channel"] == probe.channel:
                return Answer(probe_index=probe.index, value=_trim(row["payload"], arity), confidence=200, abstained=False)
        return Answer(probe_index=probe.index, value=tuple(0 for _ in range(arity)) if arity else (), confidence=0, abstained=True)

    def _propose(self) -> Iterable[Proposal]:
        proposals: list[Proposal] = []
        for address, value in sorted(self.staged.items()):
            if address in self.model and self.model[address] == value:
                continue
            proposals.append(
                Proposal(
                    proposal_id=f"oracle:{address}:{len(proposals)}",
                    kind="oracle_structure_write",
                    target=address,
                    delta=value,
                    trigger="oracle_generating_structure",
                    expected_value=10.0,
                    cost_bytes=max(8, _bytes_of(value)),
                )
            )
        return proposals

    def _commit(self, proposal: Proposal) -> None:
        self.undo[proposal.proposal_id] = self.model.get(proposal.target)
        self.model[proposal.target] = tuple(int(x) for x in proposal.delta)
        self._resize()

    def _rollback(self, receipt: Receipt) -> None:
        if receipt.proposal_id not in self.undo:
            self.model.pop(receipt.target, None)
            self._resize()
            return
        prior = self.undo.pop(receipt.proposal_id)
        if prior is None:
            self.model.pop(receipt.target, None)
        else:
            self.model[receipt.target] = prior
        self._resize()

    def _durable_state(self) -> Any:
        return {
            "form": "oracle",
            "model": {key: list(value) for key, value in sorted(self.model.items())},
            "structure_keys": sorted(self.generating_structure),
            "activation": _ACTIVATION,
        }

    def _active_state(self) -> Any:
        return {
            "buffer": [
                {
                    "channel": row["channel"],
                    "payload": list(row["payload"]),
                    "teaching": row["teaching"],
                    "index": row["index"],
                }
                for row in self.active_buffer
            ],
            "staged": {key: list(value) for key, value in sorted(self.staged.items())},
            "activation": _ACTIVATION,
        }

    def _restore_durable(self, state: Any) -> None:
        self.model = {str(key): tuple(int(x) for x in value) for key, value in state.get("model", {}).items()}
        self.undo.clear()

    def _restore_active(self, state: Any) -> None:
        self.active_buffer = [
            {
                "channel": str(row["channel"]),
                "payload": tuple(int(x) for x in row["payload"]),
                "teaching": bool(row["teaching"]),
                "index": int(row["index"]),
            }
            for row in state.get("buffer", [])
        ]
        self.staged = {str(key): tuple(int(x) for x in value) for key, value in state.get("staged", {}).items()}


def _build_oracle(opportunity: Opportunity, **options: Any) -> OracleMaterial:
    structure_raw = options.get("generating_structure") or options.get("structure") or {}
    structure: dict[str, tuple[int, ...]] = {}
    if isinstance(structure_raw, Mapping):
        for key, value in structure_raw.items():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                structure[str(key)] = tuple(int(x) for x in value)
    return OracleMaterial(
        name="oracle",
        mechanism="oracle_generating_structure_reference",
        _opportunity=opportunity,
        generating_structure=structure,
    )


register("oracle", _build_oracle)


def baseline_opportunity(
    name: str,
    *,
    envelope: str,
    observations: Sequence[Observation],
    sensor_channels: Sequence[str],
    operation_budget: int,
    durable_write_budget: int,
) -> Opportunity:
    """Build the constitutionally correct opportunity vector for a baseline."""
    from substrate.genesis_material import equal_opportunity

    if name not in C.BASELINE_DEPRIVATION:
        raise KeyError(f"unknown baseline {name!r}")
    return equal_opportunity(
        envelope=envelope,
        observations=observations,
        sensor_channels=sensor_channels,
        operation_budget=operation_budget,
        durable_write_budget=durable_write_budget,
        deprived=C.BASELINE_DEPRIVATION[name],
    )


__all__ = [
    "FRSelectedKernel",
    "OracleMaterial",
    "PrecompiledProcedureBank",
    "RandomGrowthPlastic",
    "RecordStoreNull",
    "ReplayFullHistory",
    "RetrievalOnly",
    "S2TaskIndependentMonolithicPersistentCore",
    "ShuffledHistoryPlastic",
    "StaticFrozenField",
    "SummaryReplay",
    "WrongHistoryPlastic",
    "baseline_opportunity",
]
