"""Structural cognitive materials: continuous time, adaptive topology, mixed radix, event sourcing.

Each class is a separate implementation with its own durable-change law. No material
reads held-out labels; durable writes route only through propose/apply except for K4's
harness-driven continuous-time physics (``advance``), which is that material's exclusive
mechanism and is never self-clocked from wall time.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from substrate import genesis_config as C
from substrate.final_revision_field import (
    PRECISION_ALPHABETS,
    native_low_bit_update,
    optimal_group_size,
    pack_radix,
)
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

FIELD_WIDTH = 16
DEFAULT_ALPHABET = PRECISION_ALPHABETS["ternary"]
DEFAULT_PRECISION = "ternary"
REGION_LADDER: tuple[str, ...] = ("binary", "ternary", "quinary", "4_bit", "8_bit")
MECH_K4 = "elapsed_time_driven_decay_and_expiry"
MECH_K6 = "unfrozen_allocate_split_merge_prune_under_rent"
MECH_K7 = "per_region_radix_selection_under_rent"
MECH_K8 = "append_only_projection_as_the_only_durable_path"

# Cap on proposals emitted per consolidation cycle. Explicit and matched across structural materials.
PROPOSALS_PER_CYCLE = 32


def _clamp(value: int, precision: str) -> int:
    alphabet = PRECISION_ALPHABETS.get(precision, DEFAULT_ALPHABET)
    return min(alphabet, key=lambda symbol: (abs(symbol - value), symbol))


def _pack_values(values: Sequence[int], precision: str) -> dict[str, Any]:
    alphabet = PRECISION_ALPHABETS.get(precision, DEFAULT_ALPHABET)
    group = optimal_group_size(len(alphabet))
    native = tuple(_clamp(int(v), precision) for v in values)
    packed = pack_radix(native, alphabet, group_size=group)
    return {
        "alphabet": list(packed.alphabet),
        "count": packed.count,
        "group_size": packed.group_size,
        "bit_length": packed.bit_length,
        "payload_hex": packed.payload_hex,
        "precision": precision,
    }


def _payload_bytes(document: Mapping[str, Any]) -> int:
    hex_payload = str(document.get("payload_hex", ""))
    return (len(hex_payload) // 2) if hex_payload else 0


def _encode_signal(payload: Sequence[int], width: int = FIELD_WIDTH) -> list[int]:
    signal = [0] * width
    for index, raw in enumerate(payload):
        signal[index % width] = _clamp(int(raw), DEFAULT_PRECISION)
    if not payload:
        return signal
    for index in range(width):
        signal[index] = _clamp(signal[index] + (index % 3) - 1, DEFAULT_PRECISION)
    return signal


def _mix(left: Sequence[int], right: Sequence[int], precision: str = DEFAULT_PRECISION) -> list[int]:
    alphabet = PRECISION_ALPHABETS.get(precision, DEFAULT_ALPHABET)
    return [
        native_low_bit_update(int(a), -float(b), alphabet, learning_rate=0.5)
        for a, b in zip(left, right, strict=False)
    ][: len(left)]


def _empty_topology() -> dict[str, Any]:
    return {"nodes": [], "edges": [], "archive": []}


def _empty_procedures() -> list[dict[str, Any]]:
    return []


# --------------------------------------------------------------------------
# K4 — continuous-time plastic field
# --------------------------------------------------------------------------


@dataclass
class K4_continuous_time_plastic_field(MaterialBase):
    """Real elapsed time drives decay, consolidation and prediction expiry."""

    _plastic: list[int] = field(default_factory=lambda: [0] * FIELD_WIDTH)
    _consolidation: list[int] = field(default_factory=lambda: [0] * FIELD_WIDTH)
    _predictions: list[dict[str, Any]] = field(default_factory=list)
    _precision_map: dict[str, str] = field(default_factory=lambda: {"field": DEFAULT_PRECISION})
    _topology: dict[str, Any] = field(default_factory=_empty_topology)
    _compiled_procedures: list[dict[str, Any]] = field(default_factory=_empty_procedures)
    _active_trace: list[int] = field(default_factory=lambda: [0] * FIELD_WIDTH)
    _last_channel: str = ""
    _active_pending_ms: int = 0
    _durable_clock_ms: int = 0
    _undo: dict[str, Any] = field(default_factory=dict)
    _proposal_serial: int = 0

    def advance(self, elapsed_ms: int) -> None:
        """Apply harness-supplied elapsed time. Never reads wall clocks."""
        if elapsed_ms < 0:
            raise ValueError("elapsed_ms must be non-negative")
        self._opportunity.ledger.spend(1)
        delta = int(elapsed_ms)
        self.elapsed_ms += delta
        self._durable_clock_ms += delta
        self._active_pending_ms += delta
        self._apply_time_physics(delta)
        self._sync_resident()

    def _apply_time_physics(self, elapsed_ms: int) -> None:
        """Durable continuous-time law. Only advance() and admitted commits call this."""
        if MECH_K4 in self.frozen_mechanisms or elapsed_ms <= 0:
            return
        alphabet = PRECISION_ALPHABETS[self._precision_map.get("field", DEFAULT_PRECISION)]
        steps = max(1, min(64, elapsed_ms // 25))
        decay = max(0.05, min(2.0, elapsed_ms / 100.0))
        for _ in range(steps):
            self._plastic = [
                native_low_bit_update(value, float(value) * decay, alphabet, learning_rate=0.25) for value in self._plastic
            ]
        for index, value in enumerate(self._plastic):
            if value != 0 and self._consolidation[index] < 2 and elapsed_ms >= 50:
                self._consolidation[index] = _clamp(self._consolidation[index] + 1, "quinary")
        self._predictions = [prediction for prediction in self._predictions if int(prediction["expires_at_ms"]) > self._durable_clock_ms]

    def _decay_active_only(self, elapsed_ms: int) -> None:
        if elapsed_ms <= 0:
            return
        alphabet = PRECISION_ALPHABETS[self._precision_map.get("field", DEFAULT_PRECISION)]
        decay = max(0.05, min(2.0, elapsed_ms / 100.0))
        self._active_trace = [
            native_low_bit_update(value, float(value) * decay * 0.5, alphabet, learning_rate=0.25)
            for value in self._active_trace
        ]

    def _transition(self, observation: Observation) -> None:
        signal = _encode_signal(observation.payload)
        alphabet = PRECISION_ALPHABETS[self._precision_map.get("field", DEFAULT_PRECISION)]
        self._active_trace = [
            native_low_bit_update(current, -float(incoming), alphabet, learning_rate=1.0)
            for current, incoming in zip(self._active_trace, signal, strict=True)
        ]
        self._last_channel = observation.channel
        if observation.elapsed_ms > 0:
            # Active staging only; durable physics waits for advance() or commit.
            self._active_pending_ms += int(observation.elapsed_ms)
            self._decay_active_only(int(observation.elapsed_ms))
        self._active_trace = _mix(self._active_trace, signal)

    def _answer(self, probe: Probe) -> Answer:
        signal = _encode_signal(probe.probe)
        score = sum(a * b for a, b in zip(self._plastic, signal, strict=False))
        score += sum(int(p.get("value", 0)) for p in self._predictions)
        score += sum(self._consolidation)
        value = tuple(_clamp(score + int(item), DEFAULT_PRECISION) for item in signal[: max(1, probe.arity)])
        confidence = _clamp(abs(score), "quinary")
        return Answer(probe_index=probe.index, value=value, confidence=confidence, abstained=False)

    def proposals_per_cycle(self) -> int:
        return PROPOSALS_PER_CYCLE

    def _propose(self) -> Iterable[Proposal]:
        if not self._opportunity.plasticity_enabled:
            return
        self._opportunity.ledger.spend(FIELD_WIDTH * 2)
        candidates: list[tuple[float, Proposal]] = []

        def plastic_delta(trace: Sequence[int], scale: float) -> tuple[int, ...]:
            return tuple(
                native_low_bit_update(p, -float(a) * scale, DEFAULT_ALPHABET, learning_rate=1.0) - p
                for p, a in zip(self._plastic, trace, strict=True)
            )

        # Full-field time-aware rewrites at several magnitudes of the active trace.
        full_trace = list(self._active_trace)
        for scale, tag in ((1.0, "full"), (2.0, "scale2"), (0.5, "scale_half"), (-1.0, "invert")):
            delta = plastic_delta(full_trace, scale)
            if all(x == 0 for x in delta):
                continue
            self._proposal_serial += 1
            candidates.append(
                (
                    float(sum(abs(x) for x in delta)) + abs(scale),
                    Proposal(
                        proposal_id=f"k4-{self.observations_seen}-{self._proposal_serial}:{tag}",
                        kind="time_aware_plastic_rewrite",
                        target="field",
                        delta=delta,
                        precision_request=DEFAULT_PRECISION,
                        trigger=f"channel:{self._last_channel}|pending_ms:{self._active_pending_ms}|{tag}",
                        expected_value=float(sum(abs(x) for x in delta)),
                        cost_bytes=max(1, _payload_bytes(_pack_values(self._plastic, DEFAULT_PRECISION))),
                    ),
                )
            )

        # Region-local plastic rewrites: only a contiguous half/quarter of the field.
        region_slices = (
            (0, FIELD_WIDTH // 2, "lo"),
            (FIELD_WIDTH // 2, FIELD_WIDTH // 2, "hi"),
            (0, FIELD_WIDTH // 4, "q0"),
            (FIELD_WIDTH // 4, FIELD_WIDTH // 4, "q1"),
        )
        for start, width, tag in region_slices:
            masked = [full_trace[i] if start <= i < start + width else 0 for i in range(FIELD_WIDTH)]
            delta = plastic_delta(masked, 1.0)
            if all(x == 0 for x in delta):
                continue
            self._proposal_serial += 1
            candidates.append(
                (
                    float(sum(abs(x) for x in delta)),
                    Proposal(
                        proposal_id=f"k4-{self.observations_seen}-{self._proposal_serial}:{tag}",
                        kind="time_aware_plastic_rewrite",
                        target=f"field:{tag}",
                        delta=delta,
                        precision_request=DEFAULT_PRECISION,
                        trigger=f"channel:{self._last_channel}|region:{tag}",
                        expected_value=float(sum(abs(x) for x in delta)),
                        cost_bytes=max(1, _payload_bytes(_pack_values(self._plastic, DEFAULT_PRECISION))),
                    ),
                )
            )

        # Axis-local rewrites on the largest active coordinates.
        ranked_axes = sorted(range(FIELD_WIDTH), key=lambda i: (-abs(full_trace[i]), i))
        for axis in ranked_axes:
            if full_trace[axis] == 0:
                continue
            masked = [0] * FIELD_WIDTH
            masked[axis] = full_trace[axis]
            delta = plastic_delta(masked, 1.0)
            if all(x == 0 for x in delta):
                continue
            self._proposal_serial += 1
            candidates.append(
                (
                    float(abs(full_trace[axis])),
                    Proposal(
                        proposal_id=f"k4-{self.observations_seen}-{self._proposal_serial}:ax{axis}",
                        kind="time_aware_plastic_rewrite",
                        target=f"field:ax{axis}",
                        delta=delta,
                        precision_request=DEFAULT_PRECISION,
                        trigger=f"channel:{self._last_channel}|axis:{axis}",
                        expected_value=float(abs(full_trace[axis])),
                        cost_bytes=max(1, _payload_bytes(_pack_values(self._plastic, DEFAULT_PRECISION))),
                    ),
                )
            )

        # Continuous-time exclusive: schedule predictions at several expiry horizons when time is pending.
        if self._active_pending_ms >= 25 and MECH_K4 not in self.frozen_mechanisms:
            for horizon_factor, tag in ((1, "near"), (2, "mid"), (4, "far")):
                horizon = self._durable_clock_ms + max(50, self._active_pending_ms * horizon_factor)
                self._proposal_serial += 1
                candidates.append(
                    (
                        1.0 / horizon_factor + 0.1,
                        Proposal(
                            proposal_id=f"k4-pred-{self.observations_seen}-{self._proposal_serial}:{tag}",
                            kind="schedule_prediction",
                            target="predictions",
                            delta=tuple(self._active_trace[:4]),
                            trigger=f"expiry_horizon_ms:{horizon}|{tag}",
                            expected_value=1.0 / horizon_factor,
                            cost_bytes=8,
                        ),
                    )
                )

        candidates.sort(key=lambda row: (-row[0], row[1].proposal_id))
        for _score, proposal in candidates[:PROPOSALS_PER_CYCLE]:
            yield proposal

    def _commit(self, proposal: Proposal) -> None:
        self._undo[proposal.proposal_id] = self._snapshot_durable()
        if proposal.kind == "time_aware_plastic_rewrite":
            updated = [
                _clamp(int(base) + int(delta), self._precision_map.get("field", DEFAULT_PRECISION))
                for base, delta in zip(self._plastic, proposal.delta, strict=False)
            ]
            if len(updated) < FIELD_WIDTH:
                updated.extend(self._plastic[len(updated) :])
            self._plastic = updated[:FIELD_WIDTH]
            # Fold staged active time into durable physics on admitted rewrite.
            if self._active_pending_ms > 0 and MECH_K4 not in self.frozen_mechanisms:
                staged = int(self._active_pending_ms)
                self._durable_clock_ms += staged
                self._apply_time_physics(staged)
            self._active_pending_ms = 0
        elif proposal.kind == "schedule_prediction" and MECH_K4 not in self.frozen_mechanisms:
            expires = self._durable_clock_ms + max(50, abs(sum(proposal.delta)) * 10 + 50)
            self._predictions.append(
                {
                    "value": int(sum(proposal.delta)),
                    "expires_at_ms": int(expires),
                    "channel": self._last_channel,
                }
            )
        self._sync_resident()

    def _rollback(self, receipt: Receipt) -> None:
        prior = self._undo.pop(receipt.proposal_id, None)
        if prior is None:
            return
        self._restore_durable(prior)

    def _snapshot_durable(self) -> dict[str, Any]:
        return copy.deepcopy(self._durable_state())

    def _durable_state(self) -> Any:
        precision = self._precision_map.get("field", DEFAULT_PRECISION)
        return {
            "form": "continuous_time_field",
            "plastic_packed": _pack_values(self._plastic, precision),
            "consolidation": list(self._consolidation),
            "predictions": copy.deepcopy(self._predictions),
            "precision_map": dict(self._precision_map),
            "topology": copy.deepcopy(self._topology),
            "compiled_procedures": copy.deepcopy(self._compiled_procedures),
            "durable_clock_ms": int(self._durable_clock_ms),
            "mechanism": self.mechanism,
            "activation": False,
        }

    def _active_state(self) -> Any:
        return {
            "trace": list(self._active_trace),
            "last_channel": self._last_channel,
            "active_pending_ms": int(self._active_pending_ms),
            "activation": False,
        }

    def _restore_durable(self, state: Any) -> None:
        from substrate.final_revision_field import PackedRadix, unpack_radix

        document = dict(state)
        packed = document["plastic_packed"]
        values = unpack_radix(
            PackedRadix(
                alphabet=tuple(packed["alphabet"]),
                count=int(packed["count"]),
                group_size=int(packed["group_size"]),
                bit_length=int(packed["bit_length"]),
                payload_hex=str(packed["payload_hex"]),
            )
        )
        self._plastic = list(values)[:FIELD_WIDTH]
        while len(self._plastic) < FIELD_WIDTH:
            self._plastic.append(0)
        self._consolidation = list(document.get("consolidation", [0] * FIELD_WIDTH))[:FIELD_WIDTH]
        while len(self._consolidation) < FIELD_WIDTH:
            self._consolidation.append(0)
        self._predictions = copy.deepcopy(list(document.get("predictions", [])))
        self._precision_map = dict(document.get("precision_map", {"field": DEFAULT_PRECISION}))
        self._topology = copy.deepcopy(document.get("topology", _empty_topology()))
        self._compiled_procedures = copy.deepcopy(list(document.get("compiled_procedures", [])))
        self._durable_clock_ms = int(document.get("durable_clock_ms", 0))
        self._sync_resident()

    def _restore_active(self, state: Any) -> None:
        document = dict(state)
        self._active_trace = list(document.get("trace", [0] * FIELD_WIDTH))[:FIELD_WIDTH]
        while len(self._active_trace) < FIELD_WIDTH:
            self._active_trace.append(0)
        self._last_channel = str(document.get("last_channel", ""))
        self._active_pending_ms = int(document.get("active_pending_ms", 0))

    def _sync_resident(self) -> None:
        durable = self._durable_state()
        size = _payload_bytes(durable["plastic_packed"])
        size += 8 * len(self._predictions)
        size += 2 * len(self._consolidation)
        size += len(str(durable["topology"]))
        size += len(str(durable["compiled_procedures"]))
        self._opportunity.ledger.resize(size)


# --------------------------------------------------------------------------
# K6 — adaptive topology under rent
# --------------------------------------------------------------------------


@dataclass
class K6_adaptive_topology_field(MaterialBase):
    """Unfrozen allocate/split/merge/prune/archive; growth pays rent or is pruned."""

    _plastic: list[int] = field(default_factory=lambda: [0] * FIELD_WIDTH)
    _nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    _edges: list[dict[str, str]] = field(default_factory=list)
    _archive: list[dict[str, Any]] = field(default_factory=list)
    _rent: dict[str, dict[str, Any]] = field(default_factory=dict)
    _precision_map: dict[str, str] = field(default_factory=lambda: {"field": DEFAULT_PRECISION})
    _compiled_procedures: list[dict[str, Any]] = field(default_factory=_empty_procedures)
    _active_trace: list[int] = field(default_factory=lambda: [0] * FIELD_WIDTH)
    _last_channel: str = ""
    _commit_index: int = 0
    _proposal_serial: int = 0
    _undo: dict[str, Any] = field(default_factory=dict)
    _growth_enabled: bool = True

    def _transition(self, observation: Observation) -> None:
        signal = _encode_signal(observation.payload)
        self._active_trace = _mix(self._active_trace, signal)
        self._last_channel = observation.channel

    def _answer(self, probe: Probe) -> Answer:
        signal = _encode_signal(probe.probe)
        node_bias = sum(int(node.get("activation", 0)) for node in self._nodes.values())
        edge_bias = len(self._edges)
        score = sum(a * b for a, b in zip(self._plastic, signal, strict=False)) + node_bias - edge_bias // 2
        value = tuple(_clamp(score + int(item), DEFAULT_PRECISION) for item in signal[: max(1, probe.arity)])
        return Answer(probe_index=probe.index, value=value, confidence=_clamp(abs(score), "quinary"), abstained=False)

    def _mechanism_open(self) -> bool:
        return MECH_K6 not in self.frozen_mechanisms and self._growth_enabled

    def proposals_per_cycle(self) -> int:
        return PROPOSALS_PER_CYCLE

    def _propose(self) -> Iterable[Proposal]:
        if not self._opportunity.plasticity_enabled:
            return
        # Enumeration cost is proportional to the inspected field and the ranked topology sample, not the full archive.
        self._opportunity.ledger.spend(FIELD_WIDTH + min(PROPOSALS_PER_CYCLE, max(1, len(self._nodes))) + 2)
        candidates: list[tuple[float, Proposal]] = []

        # Plastic rewrites conditioned on current topology, at several magnitudes/regions.
        full_trace = list(self._active_trace)
        for scale, tag in ((1.0, "full"), (2.0, "scale2"), (-1.0, "invert")):
            delta = tuple(
                native_low_bit_update(p, -float(a) * scale, DEFAULT_ALPHABET) - p
                for p, a in zip(self._plastic, full_trace, strict=True)
            )
            if all(x == 0 for x in delta):
                continue
            self._proposal_serial += 1
            candidates.append(
                (
                    float(sum(abs(x) for x in delta)) + 1.0,
                    Proposal(
                        proposal_id=f"k6-plastic-{self.observations_seen}-{self._proposal_serial}:{tag}",
                        kind="topology_conditioned_plastic_rewrite",
                        target="field",
                        delta=delta,
                        trigger=f"channel:{self._last_channel}|{tag}",
                        expected_value=float(sum(abs(x) for x in delta)),
                        cost_bytes=max(1, _payload_bytes(_pack_values(self._plastic, DEFAULT_PRECISION))),
                    ),
                )
            )
        axis_rank = sorted(range(FIELD_WIDTH), key=lambda i: (-abs(full_trace[i]), i))
        for axis in axis_rank[:8]:
            if full_trace[axis] == 0:
                continue
            masked = [full_trace[i] if i == axis else 0 for i in range(FIELD_WIDTH)]
            delta = tuple(
                native_low_bit_update(p, -float(a), DEFAULT_ALPHABET) - p
                for p, a in zip(self._plastic, masked, strict=True)
            )
            if all(x == 0 for x in delta):
                continue
            self._proposal_serial += 1
            candidates.append(
                (
                    float(abs(full_trace[axis])),
                    Proposal(
                        proposal_id=f"k6-plastic-{self.observations_seen}-{self._proposal_serial}:ax{axis}",
                        kind="topology_conditioned_plastic_rewrite",
                        target=f"field:ax{axis}",
                        delta=delta,
                        trigger=f"channel:{self._last_channel}|axis:{axis}",
                        expected_value=float(abs(full_trace[axis])),
                        cost_bytes=max(1, _payload_bytes(_pack_values(self._plastic, DEFAULT_PRECISION))),
                    ),
                )
            )

        if self._mechanism_open():
            # Rent audit: prune structures that failed to earn value (best-first, capped).
            prune_rows: list[tuple[float, str]] = []
            for node_id, meta in self._rent.items():
                age = self._commit_index - int(meta["born_at"])
                utility = float(meta.get("verified_utility", 0.0))
                if age >= C.PRECISION_AUDIT_WINDOW and utility <= 0.0:
                    prune_rows.append((2.0 + age - utility, node_id))
            for score, node_id in sorted(prune_rows, key=lambda row: (-row[0], row[1]))[:4]:
                self._proposal_serial += 1
                candidates.append(
                    (
                        score,
                        Proposal(
                            proposal_id=f"k6-prune-{node_id}-{self._proposal_serial}",
                            kind="topology_prune",
                            target=node_id,
                            topology_operation="prune",
                            trigger="rent_default",
                            expected_value=1.0,
                            cost_bytes=0,
                        ),
                    )
                )
            # Growth: a few allocate targets whose rent could be earned under current demand.
            # Hard cap of 8 live nodes is part of the material's rent law.
            demand = sum(abs(x) for x in self._active_trace)
            free_slots = max(0, 8 - len(self._nodes))
            if demand > 0 and free_slots > 0:
                base = len(self._nodes) + len(self._archive)
                for offset in range(1, min(free_slots, 3) + 1):
                    node_id = f"n{base + offset}"
                    self._proposal_serial += 1
                    candidates.append(
                        (
                            float(demand) / offset + 0.5,
                            Proposal(
                                proposal_id=f"k6-alloc-{node_id}-{self._proposal_serial}",
                                kind="topology_allocate",
                                target=node_id,
                                topology_operation="allocate",
                                trigger=f"demand:{demand}:slot{offset}",
                                expected_value=float(demand) / offset,
                                cost_bytes=16,
                            ),
                        )
                    )
            # Split the highest-activation nodes only (several, not the entire graph).
            ordered = sorted(
                self._nodes,
                key=lambda nid: (-abs(int(self._nodes[nid].get("activation", 0))), nid),
            )
            for rank, node_id in enumerate(ordered[:3]):
                activation = abs(int(self._nodes[node_id].get("activation", 0)))
                self._proposal_serial += 1
                candidates.append(
                    (
                        0.5 + activation - 0.01 * rank,
                        Proposal(
                            proposal_id=f"k6-split-{node_id}-{self._proposal_serial}",
                            kind="topology_split",
                            target=node_id,
                            topology_operation="split",
                            trigger="capacity",
                            expected_value=0.5 + activation,
                            cost_bytes=24,
                        ),
                    )
                )
            # Merge a few aged low-activation pairs only when both have failed to earn rent.
            if len(self._nodes) >= 2:
                by_activation = sorted(
                    (
                        nid
                        for nid, meta in self._rent.items()
                        if (self._commit_index - int(meta["born_at"])) >= C.PRECISION_AUDIT_WINDOW
                        and float(meta.get("verified_utility", 0.0)) <= 0.0
                        and nid in self._nodes
                    ),
                    key=lambda nid: (abs(int(self._nodes[nid].get("activation", 0))), nid),
                )
                for left, right in list(zip(by_activation, by_activation[1:], strict=False))[:2]:
                    self._proposal_serial += 1
                    candidates.append(
                        (
                            0.4,
                            Proposal(
                                proposal_id=f"k6-merge-{left}+{right}-{self._proposal_serial}",
                                kind="topology_merge",
                                target=f"{left}+{right}",
                                topology_operation="merge",
                                trigger="compress",
                                expected_value=0.4,
                                cost_bytes=8,
                            ),
                        )
                    )
            # Archive aged rent-default nodes (same licence as prune; distinct topology op).
            for score, node_id in sorted(prune_rows, key=lambda row: (-row[0], row[1]))[:2]:
                self._proposal_serial += 1
                candidates.append(
                    (
                        score - 0.1,
                        Proposal(
                            proposal_id=f"k6-archive-{node_id}-{self._proposal_serial}",
                            kind="topology_archive",
                            target=node_id,
                            topology_operation="archive",
                            trigger="rent_default",
                            expected_value=0.3,
                            cost_bytes=0,
                        ),
                    )
                )

        candidates.sort(key=lambda row: (-row[0], row[1].proposal_id))
        for _score, proposal in candidates[:PROPOSALS_PER_CYCLE]:
            yield proposal

    def _recompute_rent_utilities(self) -> None:
        """Verified value is the sum of committed improvements since birth (from receipts)."""
        for _node_id, meta in self._rent.items():
            born = int(meta["born_at"])
            # Receipts are appended after each apply step; index is chronological.
            utility = 0.0
            for index, receipt in enumerate(self.receipts, start=1):
                if index < born:
                    continue
                if receipt.committed and receipt.improvement > 0.0:
                    utility += float(receipt.improvement)
            meta["verified_utility"] = utility

    def _commit(self, proposal: Proposal) -> None:
        self._undo[proposal.proposal_id] = self._snapshot_durable()
        self._commit_index += 1
        self._recompute_rent_utilities()
        if proposal.kind == "topology_conditioned_plastic_rewrite":
            updated = [
                _clamp(int(base) + int(delta), DEFAULT_PRECISION)
                for base, delta in zip(self._plastic, proposal.delta, strict=False)
            ]
            if len(updated) < FIELD_WIDTH:
                updated.extend(self._plastic[len(updated) :])
            self._plastic = updated[:FIELD_WIDTH]
            for node_id, node in self._nodes.items():
                node["activation"] = _clamp(
                    int(node.get("activation", 0)) + self._plastic[hash(node_id) % FIELD_WIDTH],
                    DEFAULT_PRECISION,
                )
        elif proposal.kind == "topology_allocate" and self._mechanism_open():
            node_id = proposal.target
            self._nodes[node_id] = {
                "id": node_id,
                "activation": _clamp(int(sum(self._active_trace[:4])), DEFAULT_PRECISION),
                "precision": DEFAULT_PRECISION,
            }
            self._rent[node_id] = {
                "born_at": self._commit_index,
                "verified_utility": 0.0,
                "cost_bytes": int(proposal.cost_bytes),
            }
            if len(self._nodes) > 1:
                others = [nid for nid in self._nodes if nid != node_id]
                self._edges.append({"source": node_id, "target": others[0]})
        elif proposal.kind == "topology_split" and self._mechanism_open():
            parent = proposal.target
            if parent in self._nodes:
                child = f"{parent}.s{self._commit_index}"
                parent_act = int(self._nodes[parent].get("activation", 0))
                self._nodes[parent]["activation"] = _clamp(parent_act // 2, DEFAULT_PRECISION)
                self._nodes[child] = {
                    "id": child,
                    "activation": _clamp(parent_act - parent_act // 2, DEFAULT_PRECISION),
                    "precision": DEFAULT_PRECISION,
                }
                self._edges.append({"source": parent, "target": child})
                self._rent[child] = {
                    "born_at": self._commit_index,
                    "verified_utility": 0.0,
                    "cost_bytes": int(proposal.cost_bytes),
                }
        elif proposal.kind == "topology_merge" and self._mechanism_open():
            parts = proposal.target.split("+")
            if len(parts) == 2 and parts[0] in self._nodes and parts[1] in self._nodes:
                kept, drop = parts[0], parts[1]
                self._nodes[kept]["activation"] = _clamp(
                    int(self._nodes[kept]["activation"]) + int(self._nodes[drop]["activation"]),
                    DEFAULT_PRECISION,
                )
                del self._nodes[drop]
                self._rent.pop(drop, None)
                self._edges = [edge for edge in self._edges if edge["source"] != drop and edge["target"] != drop]
        elif proposal.kind == "topology_prune":
            self._prune_node(proposal.target)
        elif proposal.kind == "topology_archive" and self._mechanism_open():
            node_id = proposal.target
            if node_id in self._nodes:
                self._archive.append({"node": self._nodes.pop(node_id), "at": self._commit_index})
                self._rent.pop(node_id, None)
                self._edges = [edge for edge in self._edges if edge["source"] != node_id and edge["target"] != node_id]
        self._enforce_rent()
        self._sync_resident()

    def _prune_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            return
        self._archive.append({"node": self._nodes.pop(node_id), "at": self._commit_index, "reason": "rent_default"})
        self._rent.pop(node_id, None)
        self._edges = [edge for edge in self._edges if edge["source"] != node_id and edge["target"] != node_id]

    def _enforce_rent(self) -> None:
        if not self._mechanism_open():
            return
        self._recompute_rent_utilities()
        for node_id, meta in list(self._rent.items()):
            age = self._commit_index - int(meta["born_at"])
            utility = float(meta.get("verified_utility", 0.0))
            if age >= C.PRECISION_AUDIT_WINDOW and utility <= 0.0:
                self._prune_node(node_id)

    def _rollback(self, receipt: Receipt) -> None:
        prior = self._undo.pop(receipt.proposal_id, None)
        if prior is None:
            return
        self._restore_durable(prior)

    def _snapshot_durable(self) -> dict[str, Any]:
        return copy.deepcopy(self._durable_state())

    def _durable_state(self) -> Any:
        return {
            "form": "adaptive_topology_field",
            "plastic_packed": _pack_values(self._plastic, self._precision_map.get("field", DEFAULT_PRECISION)),
            "nodes": {key: dict(value) for key, value in sorted(self._nodes.items())},
            "edges": list(self._edges),
            "archive": copy.deepcopy(self._archive),
            "rent": {key: dict(value) for key, value in sorted(self._rent.items())},
            "precision_map": dict(self._precision_map),
            "compiled_procedures": copy.deepcopy(self._compiled_procedures),
            "commit_index": self._commit_index,
            "mechanism": self.mechanism,
            "activation": False,
        }

    def _active_state(self) -> Any:
        return {"trace": list(self._active_trace), "last_channel": self._last_channel, "activation": False}

    def _restore_durable(self, state: Any) -> None:
        from substrate.final_revision_field import PackedRadix, unpack_radix

        document = dict(state)
        packed = document["plastic_packed"]
        values = unpack_radix(
            PackedRadix(
                alphabet=tuple(packed["alphabet"]),
                count=int(packed["count"]),
                group_size=int(packed["group_size"]),
                bit_length=int(packed["bit_length"]),
                payload_hex=str(packed["payload_hex"]),
            )
        )
        self._plastic = list(values)[:FIELD_WIDTH]
        while len(self._plastic) < FIELD_WIDTH:
            self._plastic.append(0)
        self._nodes = {key: dict(value) for key, value in dict(document.get("nodes", {})).items()}
        self._edges = list(document.get("edges", []))
        self._archive = copy.deepcopy(list(document.get("archive", [])))
        self._rent = {key: dict(value) for key, value in dict(document.get("rent", {})).items()}
        self._precision_map = dict(document.get("precision_map", {"field": DEFAULT_PRECISION}))
        self._compiled_procedures = copy.deepcopy(list(document.get("compiled_procedures", [])))
        self._commit_index = int(document.get("commit_index", 0))
        self._sync_resident()

    def _restore_active(self, state: Any) -> None:
        document = dict(state)
        self._active_trace = list(document.get("trace", [0] * FIELD_WIDTH))[:FIELD_WIDTH]
        while len(self._active_trace) < FIELD_WIDTH:
            self._active_trace.append(0)
        self._last_channel = str(document.get("last_channel", ""))

    def _sync_resident(self) -> None:
        durable = self._durable_state()
        size = _payload_bytes(durable["plastic_packed"])
        size += 32 * len(self._nodes)
        size += 16 * len(self._edges)
        size += 24 * len(self._archive)
        size += 16 * len(self._rent)
        size += len(str(durable["compiled_procedures"]))
        self._opportunity.ledger.resize(size)


# --------------------------------------------------------------------------
# K7 — native mixed-radix field
# --------------------------------------------------------------------------


@dataclass
class K7_native_mixed_radix_field(MaterialBase):
    """Per-region radix selection with promote/demote under earn-your-bits rent."""

    _regions: dict[str, list[int]] = field(
        default_factory=lambda: {
            "routing": [0] * (FIELD_WIDTH // 2),
            "influence": [0] * (FIELD_WIDTH // 2),
            "memory": [0] * FIELD_WIDTH,
        }
    )
    _precision_map: dict[str, str] = field(
        default_factory=lambda: {
            "routing": "ternary",
            "influence": "ternary",
            "memory": "ternary",
        }
    )
    _precision_rent: dict[str, dict[str, Any]] = field(default_factory=dict)
    _topology: dict[str, Any] = field(default_factory=_empty_topology)
    _compiled_procedures: list[dict[str, Any]] = field(default_factory=_empty_procedures)
    _active_trace: list[int] = field(default_factory=lambda: [0] * FIELD_WIDTH)
    _last_channel: str = ""
    _commit_index: int = 0
    _proposal_serial: int = 0
    _undo: dict[str, Any] = field(default_factory=dict)

    def _transition(self, observation: Observation) -> None:
        signal = _encode_signal(observation.payload)
        self._active_trace = _mix(self._active_trace, signal)
        self._last_channel = observation.channel

    def _answer(self, probe: Probe) -> Answer:
        signal = _encode_signal(probe.probe)
        flat = self._flatten()
        score = sum(a * b for a, b in zip(flat, signal, strict=False))
        # Precision mass enters the answer so radix policy is behaviorally visible.
        score += sum(REGION_LADDER.index(self._precision_map[name]) for name in sorted(self._precision_map))
        value = tuple(_clamp(score + int(item), DEFAULT_PRECISION) for item in signal[: max(1, probe.arity)])
        return Answer(probe_index=probe.index, value=value, confidence=_clamp(abs(score), "quinary"), abstained=False)

    def _flatten(self) -> list[int]:
        out = [0] * FIELD_WIDTH
        for name, values in self._regions.items():
            for index, value in enumerate(values):
                out[index % FIELD_WIDTH] = _clamp(out[index % FIELD_WIDTH] + int(value), self._precision_map[name])
        return out

    def _mechanism_open(self) -> bool:
        return MECH_K7 not in self.frozen_mechanisms

    def proposals_per_cycle(self) -> int:
        return PROPOSALS_PER_CYCLE

    def _propose(self) -> Iterable[Proposal]:
        if not self._opportunity.plasticity_enabled:
            return
        self._opportunity.ledger.spend(sum(len(v) for v in self._regions.values()) + len(self._regions) * 2)
        candidates: list[tuple[float, Proposal]] = []
        demand = sum(abs(x) for x in self._active_trace)

        # Region plastic rewrites at each region's native radix, plus scaled/axis variants.
        for name, values in sorted(self._regions.items()):
            precision = self._precision_map[name]
            alphabet = PRECISION_ALPHABETS[precision]
            slice_trace = list(self._active_trace[: len(values)])
            while len(slice_trace) < len(values):
                slice_trace.append(0)
            for scale, tag in ((1.0, "full"), (2.0, "scale2"), (-1.0, "invert")):
                delta = tuple(
                    native_low_bit_update(v, -float(t) * scale, alphabet) - v
                    for v, t in zip(values, slice_trace, strict=True)
                )
                if all(x == 0 for x in delta):
                    continue
                self._proposal_serial += 1
                candidates.append(
                    (
                        float(sum(abs(x) for x in delta)),
                        Proposal(
                            proposal_id=f"k7-write-{name}-{self.observations_seen}-{self._proposal_serial}:{tag}",
                            kind="region_plastic_rewrite",
                            target=name,
                            delta=delta,
                            precision_request=precision,
                            trigger=f"channel:{self._last_channel}|{tag}",
                            expected_value=float(sum(abs(x) for x in delta)),
                            cost_bytes=max(1, _payload_bytes(_pack_values(values, precision))),
                        ),
                    )
                )
            for axis in sorted(range(len(values)), key=lambda i: (-abs(slice_trace[i]), i)):
                if slice_trace[axis] == 0:
                    continue
                masked = [slice_trace[i] if i == axis else 0 for i in range(len(values))]
                delta = tuple(
                    native_low_bit_update(v, -float(t), alphabet) - v for v, t in zip(values, masked, strict=True)
                )
                if all(x == 0 for x in delta):
                    continue
                self._proposal_serial += 1
                candidates.append(
                    (
                        float(abs(slice_trace[axis])),
                        Proposal(
                            proposal_id=f"k7-write-{name}-{self.observations_seen}-{self._proposal_serial}:ax{axis}",
                            kind="region_plastic_rewrite",
                            target=name,
                            delta=delta,
                            precision_request=precision,
                            trigger=f"channel:{self._last_channel}|axis:{axis}",
                            expected_value=float(abs(slice_trace[axis])),
                            cost_bytes=max(1, _payload_bytes(_pack_values(values, precision))),
                        ),
                    )
                )

        if self._mechanism_open():
            # Promotion for every region whose demand could pay the radix rent.
            for name, precision in sorted(self._precision_map.items()):
                if name in self._precision_rent:
                    continue
                rank = REGION_LADDER.index(precision)
                if demand >= (rank + 1) * 2 and rank < len(REGION_LADDER) - 1:
                    higher = REGION_LADDER[rank + 1]
                    before = _payload_bytes(_pack_values(self._regions[name], precision))
                    projected = [_clamp(v, higher) for v in self._regions[name]]
                    after = _payload_bytes(_pack_values(projected, higher))
                    added_bytes = max(1, after - before)
                    self._proposal_serial += 1
                    candidates.append(
                        (
                            float(demand) / float(added_bytes),
                            Proposal(
                                proposal_id=f"k7-promote-{name}-{self._proposal_serial}",
                                kind="precision_promote",
                                target=name,
                                precision_request=higher,
                                trigger=f"demand:{demand}",
                                expected_value=float(demand) / float(added_bytes),
                                cost_bytes=added_bytes,
                            ),
                        )
                    )
            # Demotion proposals for rent defaulters (also auto-enforced on commit).
            for name, meta in sorted(self._precision_rent.items()):
                age = self._commit_index - int(meta["born_at"])
                utility = float(meta.get("verified_utility", 0.0))
                added = max(1, int(meta.get("added_bytes", 1)))
                rate = utility / float(added)
                if age >= C.PRECISION_AUDIT_WINDOW and rate < C.MINIMUM_UTILITY_PER_ADDED_BYTE:
                    self._proposal_serial += 1
                    candidates.append(
                        (
                            2.0,
                            Proposal(
                                proposal_id=f"k7-demote-{name}-{self._proposal_serial}",
                                kind="precision_demote",
                                target=name,
                                precision_request=str(meta.get("prior_precision", "ternary")),
                                trigger="rent_default",
                                expected_value=1.0,
                                cost_bytes=0,
                            ),
                        )
                    )

        candidates.sort(key=lambda row: (-row[0], row[1].proposal_id))
        for _score, proposal in candidates[:PROPOSALS_PER_CYCLE]:
            yield proposal

    def _commit(self, proposal: Proposal) -> None:
        self._undo[proposal.proposal_id] = self._snapshot_durable()
        self._commit_index += 1
        self._recompute_precision_rent()
        if proposal.kind == "region_plastic_rewrite":
            name = proposal.target
            precision = self._precision_map[name]
            values = self._regions[name]
            updated = [
                _clamp(int(base) + int(delta), precision) for base, delta in zip(values, proposal.delta, strict=False)
            ]
            if len(updated) < len(values):
                updated.extend(values[len(updated) :])
            self._regions[name] = updated[: len(values)]
        elif proposal.kind == "precision_promote" and self._mechanism_open():
            name = proposal.target
            prior = self._precision_map[name]
            higher = str(proposal.precision_request or prior)
            if higher in REGION_LADDER and REGION_LADDER.index(higher) > REGION_LADDER.index(prior):
                self._regions[name] = [_clamp(v, higher) for v in self._regions[name]]
                self._precision_map[name] = higher
                self._precision_rent[name] = {
                    "born_at": self._commit_index,
                    "verified_utility": 0.0,
                    "added_bytes": max(1, int(proposal.cost_bytes)),
                    "prior_precision": prior,
                }
        elif proposal.kind == "precision_demote" and self._mechanism_open():
            self._demote_region(proposal.target)
        self._enforce_precision_rent()
        self._sync_resident()

    def _recompute_precision_rent(self) -> None:
        for _name, meta in self._precision_rent.items():
            born = int(meta["born_at"])
            utility = 0.0
            for index, receipt in enumerate(self.receipts, start=1):
                if index < born:
                    continue
                if receipt.committed and receipt.improvement > 0.0:
                    utility += float(receipt.improvement)
            meta["verified_utility"] = utility

    def _demote_region(self, name: str) -> None:
        if name not in self._precision_map:
            return
        meta = self._precision_rent.get(name, {})
        prior = str(meta.get("prior_precision", "ternary"))
        if prior not in REGION_LADDER:
            rank = max(0, REGION_LADDER.index(self._precision_map[name]) - 1)
            prior = REGION_LADDER[rank]
        self._precision_map[name] = prior
        self._regions[name] = [_clamp(v, prior) for v in self._regions[name]]
        self._precision_rent.pop(name, None)

    def _enforce_precision_rent(self) -> None:
        if not self._mechanism_open():
            return
        self._recompute_precision_rent()
        for name, meta in list(self._precision_rent.items()):
            age = self._commit_index - int(meta["born_at"])
            utility = float(meta.get("verified_utility", 0.0))
            added = max(1, int(meta.get("added_bytes", 1)))
            if age >= C.PRECISION_AUDIT_WINDOW and (utility / float(added)) < C.MINIMUM_UTILITY_PER_ADDED_BYTE:
                self._demote_region(name)

    def _rollback(self, receipt: Receipt) -> None:
        prior = self._undo.pop(receipt.proposal_id, None)
        if prior is None:
            return
        self._restore_durable(prior)

    def _snapshot_durable(self) -> dict[str, Any]:
        return copy.deepcopy(self._durable_state())

    def _durable_state(self) -> Any:
        packed_regions = {
            name: _pack_values(values, self._precision_map[name]) for name, values in sorted(self._regions.items())
        }
        return {
            "form": "native_mixed_radix_field",
            "regions_packed": packed_regions,
            "precision_map": dict(sorted(self._precision_map.items())),
            "precision_rent": {key: dict(value) for key, value in sorted(self._precision_rent.items())},
            "topology": copy.deepcopy(self._topology),
            "compiled_procedures": copy.deepcopy(self._compiled_procedures),
            "commit_index": self._commit_index,
            "mechanism": self.mechanism,
            "activation": False,
        }

    def _active_state(self) -> Any:
        return {"trace": list(self._active_trace), "last_channel": self._last_channel, "activation": False}

    def _restore_durable(self, state: Any) -> None:
        from substrate.final_revision_field import PackedRadix, unpack_radix

        document = dict(state)
        self._precision_map = dict(document.get("precision_map", {}))
        regions: dict[str, list[int]] = {}
        for name, packed in dict(document.get("regions_packed", {})).items():
            values = unpack_radix(
                PackedRadix(
                    alphabet=tuple(packed["alphabet"]),
                    count=int(packed["count"]),
                    group_size=int(packed["group_size"]),
                    bit_length=int(packed["bit_length"]),
                    payload_hex=str(packed["payload_hex"]),
                )
            )
            regions[name] = list(values)
        self._regions = regions
        self._precision_rent = {key: dict(value) for key, value in dict(document.get("precision_rent", {})).items()}
        self._topology = copy.deepcopy(document.get("topology", _empty_topology()))
        self._compiled_procedures = copy.deepcopy(list(document.get("compiled_procedures", [])))
        self._commit_index = int(document.get("commit_index", 0))
        self._sync_resident()

    def _restore_active(self, state: Any) -> None:
        document = dict(state)
        self._active_trace = list(document.get("trace", [0] * FIELD_WIDTH))[:FIELD_WIDTH]
        while len(self._active_trace) < FIELD_WIDTH:
            self._active_trace.append(0)
        self._last_channel = str(document.get("last_channel", ""))

    def _sync_resident(self) -> None:
        durable = self._durable_state()
        size = sum(_payload_bytes(packed) for packed in durable["regions_packed"].values())
        size += 12 * len(self._precision_map)
        size += 16 * len(self._precision_rent)
        size += len(str(durable["topology"]))
        size += len(str(durable["compiled_procedures"]))
        # precision_bits channel: sum of alphabet bit widths times region lengths.
        bits = 0
        for name, values in self._regions.items():
            alphabet = PRECISION_ALPHABETS[self._precision_map[name]]
            bits += max(1, (len(alphabet) - 1).bit_length()) * len(values)
        self._opportunity.ledger.precision_bits = bits
        self._opportunity.ledger.resize(size)


# --------------------------------------------------------------------------
# K8 — event-sourced plastic field
# --------------------------------------------------------------------------


@dataclass
class K8_event_sourced_plastic_field(MaterialBase):
    """Append-only archive is the only durable path; projection is deterministic."""

    _archive: list[dict[str, Any]] = field(default_factory=list)
    _projection: dict[str, Any] = field(default_factory=dict)
    _precision_map: dict[str, str] = field(default_factory=lambda: {"field": DEFAULT_PRECISION})
    _compiled_procedures: list[dict[str, Any]] = field(default_factory=_empty_procedures)
    _active_trace: list[int] = field(default_factory=lambda: [0] * FIELD_WIDTH)
    _last_channel: str = ""
    _proposal_serial: int = 0
    _undo: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._projection:
            self._projection = self.project_from_archive(self._archive)

    @staticmethod
    def project_from_archive(archive: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Deterministic projection rebuild. Pure function of the archive."""
        plastic = [0] * FIELD_WIDTH
        topology = _empty_topology()
        precision_map = {"field": DEFAULT_PRECISION}
        procedures: list[dict[str, Any]] = []
        for event in archive:
            kind = str(event.get("kind", ""))
            if kind == "plastic_event":
                delta = list(event.get("delta", ()))
                for index, step in enumerate(delta):
                    plastic[index % FIELD_WIDTH] = _clamp(plastic[index % FIELD_WIDTH] + int(step), precision_map["field"])
            elif kind == "topology_event":
                node = str(event.get("node", ""))
                if node and node not in topology["nodes"]:
                    topology["nodes"].append(node)
                edge = event.get("edge")
                if isinstance(edge, dict):
                    topology["edges"].append(dict(edge))
            elif kind == "precision_event":
                precision_map["field"] = str(event.get("precision", precision_map["field"]))
                plastic = [_clamp(v, precision_map["field"]) for v in plastic]
            elif kind == "procedure_event":
                procedures.append({"id": event.get("procedure_id"), "body": event.get("body", ())})
        return {
            "plastic": plastic,
            "topology": topology,
            "precision_map": precision_map,
            "compiled_procedures": procedures,
        }

    def rebuild_projection(self) -> dict[str, Any]:
        """Public rebuild used by tests to prove archive-only reconstruction."""
        return self.project_from_archive(self._archive)

    def _transition(self, observation: Observation) -> None:
        signal = _encode_signal(observation.payload)
        self._active_trace = _mix(self._active_trace, signal)
        self._last_channel = observation.channel

    def _answer(self, probe: Probe) -> Answer:
        signal = _encode_signal(probe.probe)
        plastic = list(self._projection.get("plastic", [0] * FIELD_WIDTH))
        score = sum(a * b for a, b in zip(plastic, signal, strict=False))
        score += len(self._archive)
        score += len(self._projection.get("topology", {}).get("nodes", []))
        value = tuple(_clamp(score + int(item), DEFAULT_PRECISION) for item in signal[: max(1, probe.arity)])
        return Answer(probe_index=probe.index, value=value, confidence=_clamp(abs(score), "quinary"), abstained=False)

    def _mechanism_open(self) -> bool:
        return MECH_K8 not in self.frozen_mechanisms

    def proposals_per_cycle(self) -> int:
        return PROPOSALS_PER_CYCLE

    def _propose(self) -> Iterable[Proposal]:
        if not self._opportunity.plasticity_enabled:
            return
        if not self._mechanism_open():
            # Even with mechanism frozen, no durable path exists except archive events.
            return
        self._opportunity.ledger.spend(FIELD_WIDTH * 2)
        candidates: list[tuple[float, Proposal]] = []
        trace = list(self._active_trace)
        mass = sum(abs(x) for x in trace)

        # Plastic events: full, scaled, inverted, and axis-local archive appends.
        for scale, tag in ((1.0, "full"), (2.0, "scale2"), (-1.0, "invert"), (0.5, "half")):
            delta = tuple(native_low_bit_update(0, -float(a) * scale, DEFAULT_ALPHABET) for a in trace)
            if all(x == 0 for x in delta):
                continue
            self._proposal_serial += 1
            candidates.append(
                (
                    float(sum(abs(x) for x in delta)),
                    Proposal(
                        proposal_id=f"k8-event-{self.observations_seen}-{self._proposal_serial}:{tag}",
                        kind="plastic_event",
                        target="archive",
                        delta=delta,
                        trigger=f"channel:{self._last_channel}|{tag}",
                        expected_value=float(sum(abs(x) for x in delta)),
                        cost_bytes=max(1, 4 + 2 * len(delta)),
                    ),
                )
            )
        for axis in sorted(range(FIELD_WIDTH), key=lambda i: (-abs(trace[i]), i)):
            if trace[axis] == 0:
                continue
            delta = tuple(
                native_low_bit_update(0, -float(trace[i] if i == axis else 0), DEFAULT_ALPHABET) for i in range(FIELD_WIDTH)
            )
            if all(x == 0 for x in delta):
                continue
            self._proposal_serial += 1
            candidates.append(
                (
                    float(abs(trace[axis])),
                    Proposal(
                        proposal_id=f"k8-event-{self.observations_seen}-{self._proposal_serial}:ax{axis}",
                        kind="plastic_event",
                        target="archive",
                        delta=delta,
                        trigger=f"channel:{self._last_channel}|axis:{axis}",
                        expected_value=float(abs(trace[axis])),
                        cost_bytes=max(1, 4 + 2 * len(delta)),
                    ),
                )
            )

        # Topology / precision / procedure events licensed by the event-sourced path.
        if mass > 2:
            for offset in range(1, 5):
                node = f"e{len(self._archive) + offset}"
                self._proposal_serial += 1
                candidates.append(
                    (
                        1.0 / offset + mass * 0.01,
                        Proposal(
                            proposal_id=f"k8-topo-{self.observations_seen}-{self._proposal_serial}:n{offset}",
                            kind="topology_event",
                            target=node,
                            topology_operation="append_node",
                            delta=(offset,),
                            trigger="structure_from_event",
                            expected_value=1.0 / offset,
                            cost_bytes=8,
                        ),
                    )
                )
        if mass >= 3:
            for precision in ("quinary", "4_bit"):
                if precision == self._precision_map.get("field", DEFAULT_PRECISION):
                    continue
                self._proposal_serial += 1
                candidates.append(
                    (
                        0.75,
                        Proposal(
                            proposal_id=f"k8-prec-{self.observations_seen}-{self._proposal_serial}:{precision}",
                            kind="precision_event",
                            target="precision",
                            precision_request=precision,
                            delta=(mass,),
                            trigger=f"precision:{precision}",
                            expected_value=0.75,
                            cost_bytes=4,
                        ),
                    )
                )
        if mass >= 2:
            for body_scale in (1, 2):
                body = tuple(_clamp(int(x) * body_scale, DEFAULT_PRECISION) for x in trace[:4])
                if all(x == 0 for x in body):
                    continue
                self._proposal_serial += 1
                candidates.append(
                    (
                        0.5 * body_scale,
                        Proposal(
                            proposal_id=f"k8-proc-{self.observations_seen}-{self._proposal_serial}:s{body_scale}",
                            kind="procedure_event",
                            target=f"proc:{self.observations_seen}:{body_scale}",
                            delta=body,
                            trigger="compile_from_event",
                            expected_value=0.5 * body_scale,
                            cost_bytes=max(1, 2 * len(body)),
                        ),
                    )
                )

        candidates.sort(key=lambda row: (-row[0], row[1].proposal_id))
        for _score, proposal in candidates[:PROPOSALS_PER_CYCLE]:
            yield proposal

    def _commit(self, proposal: Proposal) -> None:
        self._undo[proposal.proposal_id] = self._snapshot_durable()
        if not self._mechanism_open():
            return
        if proposal.kind == "plastic_event":
            event = {
                "kind": "plastic_event",
                "delta": list(proposal.delta),
                "channel": self._last_channel,
                "seq": len(self._archive),
            }
            self._archive.append(event)
        elif proposal.kind == "topology_event":
            nodes = self._projection.get("topology", {}).get("nodes", [])
            source = nodes[-1] if nodes else "root"
            event = {
                "kind": "topology_event",
                "node": proposal.target,
                "edge": {"source": source, "target": proposal.target},
                "seq": len(self._archive),
            }
            self._archive.append(event)
        elif proposal.kind == "precision_event":
            self._archive.append(
                {
                    "kind": "precision_event",
                    "precision": proposal.precision_request or DEFAULT_PRECISION,
                    "seq": len(self._archive),
                }
            )
        elif proposal.kind == "procedure_event":
            self._archive.append(
                {
                    "kind": "procedure_event",
                    "procedure_id": proposal.target,
                    "body": list(proposal.delta),
                    "seq": len(self._archive),
                }
            )
        # Projection is always rebuilt; never mutated independently.
        self._projection = self.project_from_archive(self._archive)
        self._precision_map = dict(self._projection["precision_map"])
        self._compiled_procedures = list(self._projection["compiled_procedures"])
        self._sync_resident()

    def _rollback(self, receipt: Receipt) -> None:
        prior = self._undo.pop(receipt.proposal_id, None)
        if prior is None:
            return
        self._restore_durable(prior)

    def _snapshot_durable(self) -> dict[str, Any]:
        return copy.deepcopy(self._durable_state())

    def _durable_state(self) -> Any:
        # Durable truth is the archive; projection is included only as a deterministic view.
        projection = self.project_from_archive(self._archive)
        plastic = list(projection["plastic"])
        return {
            "form": "event_sourced_plastic_field",
            "archive": copy.deepcopy(list(self._archive)),
            "projection_packed": _pack_values(plastic, projection["precision_map"].get("field", DEFAULT_PRECISION)),
            "projection_topology": copy.deepcopy(projection["topology"]),
            "precision_map": dict(projection["precision_map"]),
            "compiled_procedures": copy.deepcopy(projection["compiled_procedures"]),
            "mechanism": self.mechanism,
            "activation": False,
        }

    def _active_state(self) -> Any:
        return {"trace": list(self._active_trace), "last_channel": self._last_channel, "activation": False}

    def _restore_durable(self, state: Any) -> None:
        document = dict(state)
        self._archive = copy.deepcopy(list(document.get("archive", [])))
        self._projection = self.project_from_archive(self._archive)
        self._precision_map = dict(self._projection["precision_map"])
        self._compiled_procedures = list(self._projection["compiled_procedures"])
        self._sync_resident()

    def _restore_active(self, state: Any) -> None:
        document = dict(state)
        self._active_trace = list(document.get("trace", [0] * FIELD_WIDTH))[:FIELD_WIDTH]
        while len(self._active_trace) < FIELD_WIDTH:
            self._active_trace.append(0)
        self._last_channel = str(document.get("last_channel", ""))

    def _sync_resident(self) -> None:
        size = sum(8 + 2 * len(event.get("delta", ())) for event in self._archive)
        size += _payload_bytes(_pack_values(self._projection.get("plastic", [0] * FIELD_WIDTH), DEFAULT_PRECISION))
        size += 16 * len(self._projection.get("topology", {}).get("nodes", []))
        size += len(str(self._compiled_procedures))
        self._opportunity.ledger.resize(size)


# --------------------------------------------------------------------------
# Factories and registration
# --------------------------------------------------------------------------


def _make_k4(opportunity: Opportunity, **options: Any) -> K4_continuous_time_plastic_field:
    material = K4_continuous_time_plastic_field(
        name="K4_continuous_time_plastic_field",
        mechanism=MECH_K4,
        _opportunity=opportunity,
    )
    material._sync_resident()
    return material


def _make_k6(opportunity: Opportunity, **options: Any) -> K6_adaptive_topology_field:
    material = K6_adaptive_topology_field(
        name="K6_adaptive_topology_field",
        mechanism=MECH_K6,
        _opportunity=opportunity,
    )
    material._sync_resident()
    return material


def _make_k7(opportunity: Opportunity, **options: Any) -> K7_native_mixed_radix_field:
    material = K7_native_mixed_radix_field(
        name="K7_native_mixed_radix_field",
        mechanism=MECH_K7,
        _opportunity=opportunity,
    )
    material._sync_resident()
    return material


def _make_k8(opportunity: Opportunity, **options: Any) -> K8_event_sourced_plastic_field:
    material = K8_event_sourced_plastic_field(
        name="K8_event_sourced_plastic_field",
        mechanism=MECH_K8,
        _opportunity=opportunity,
    )
    material._sync_resident()
    return material


register("K4_continuous_time_plastic_field", _make_k4)
register("K6_adaptive_topology_field", _make_k6)
register("K7_native_mixed_radix_field", _make_k7)
register("K8_event_sourced_plastic_field", _make_k8)

__all__ = [
    "PROPOSALS_PER_CYCLE",
    "K4_continuous_time_plastic_field",
    "K6_adaptive_topology_field",
    "K7_native_mixed_radix_field",
    "K8_event_sourced_plastic_field",
    "MECH_K4",
    "MECH_K6",
    "MECH_K7",
    "MECH_K8",
]
