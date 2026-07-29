"""Genesis basic cognitive materials: K1, K2, K3, K5.

Each material is a separate class with its own durable-change law. Materials never
see held-out labels; durable writes only happen through propose/apply; observe and
answer touch only active state.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from substrate.final_revision_field import (
    PRECISION_ALPHABETS,
    PackedRadix,
    multiplication_light_dot,
    native_low_bit_update,
    optimal_group_size,
    pack_radix,
    unpack_radix,
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

# Native bulk precision for plastic mass. Exact symbolic is reserved for identity/provenance only.
_BULK = "quinary"
_ALPHABET = PRECISION_ALPHABETS[_BULK]
_GROUP = optimal_group_size(len(_ALPHABET))


def _clamp(value: int) -> int:
    return min(_ALPHABET, key=lambda symbol: (abs(symbol - int(value)), symbol))


def _pack(values: Sequence[int]) -> PackedRadix:
    clamped = [_clamp(v) for v in values]
    return pack_radix(clamped, _ALPHABET, group_size=_GROUP)


def _unpack(packed: PackedRadix | MappingLike) -> list[int]:
    if isinstance(packed, PackedRadix):
        return unpack_radix(packed)
    return unpack_radix(
        PackedRadix(
            alphabet=tuple(packed["alphabet"]),
            count=int(packed["count"]),
            group_size=int(packed["group_size"]),
            bit_length=int(packed["bit_length"]),
            payload_hex=str(packed["payload_hex"]),
        )
    )


def _packed_document(packed: PackedRadix) -> dict[str, Any]:
    return {
        "alphabet": list(packed.alphabet),
        "count": packed.count,
        "group_size": packed.group_size,
        "bit_length": packed.bit_length,
        "payload_hex": packed.payload_hex,
        "byte_length": packed.byte_length,
    }


def _packed_bytes(packed: PackedRadix) -> int:
    return packed.byte_length


def _payload_features(payload: Sequence[int], dim: int) -> list[int]:
    features = [0] * dim
    if not payload:
        return features
    for index, raw in enumerate(payload):
        features[index % dim] = _clamp(features[index % dim] + int(raw))
    return features


def _channel_index(channel: str, modulus: int) -> int:
    if modulus <= 0:
        return 0
    acc = 0
    for char in channel:
        acc = (acc * 33 + ord(char)) % (10**9 + 7)
    return acc % modulus


MappingLike = dict[str, Any]


# ---------------------------------------------------------------------------
# K1 — monolithic dense plastic field (no topology)
# ---------------------------------------------------------------------------


class K1_monolithic_plastic_field(MaterialBase):
    """One dense plastic accumulation over the whole field; no structure, no locality, no graph."""

    MECHANISM = "dense_global_plastic_accumulation"

    def __init__(self, opportunity: Opportunity, *, field_dim: int = 16, **_options: Any) -> None:
        super().__init__(
            name="K1_monolithic_plastic_field",
            mechanism=self.MECHANISM,
            _opportunity=opportunity,
        )
        self._field_dim = int(field_dim)
        self._field = [0] * self._field_dim
        self._precision_map = {"field": _BULK}
        self._compiled_procedures: list[dict[str, Any]] = []
        self._last_features = [0] * self._field_dim
        self._last_channel = ""
        self._proposal_seq = 0
        self._undo: dict[str, Any] = {}
        self._resize()

    def _resize(self) -> None:
        packed = _pack(self._field)
        # Honest resident estimate from packed bulk plus small symbolic shell.
        shell = 64 + 8 * len(self._compiled_procedures) + sum(len(k) + len(v) for k, v in self._precision_map.items())
        self._opportunity.ledger.resize(_packed_bytes(packed) + shell)

    def _transition(self, observation: Observation) -> None:
        self._opportunity.ledger.spend(self._field_dim)
        features = _payload_features(observation.payload, self._field_dim)
        channel_bias = _channel_index(observation.channel, self._field_dim)
        features[channel_bias] = _clamp(features[channel_bias] + (1 if observation.teaching else 0))
        self._last_features = features
        self._last_channel = observation.channel

    def _answer(self, probe: Probe) -> Answer:
        self._opportunity.ledger.spend(self._field_dim, precision_bits=int(len(_ALPHABET).bit_length() * self._field_dim))
        features = _payload_features(probe.probe, self._field_dim)
        dot = int(multiplication_light_dot(self._field, features)["value"])
        arity = max(1, probe.arity)
        value = tuple(_clamp(dot + features[i % self._field_dim]) for i in range(arity))
        confidence = min(127, abs(dot))
        return Answer(probe_index=probe.index, value=value, confidence=confidence, abstained=False)

    def _propose(self) -> Iterable[Proposal]:
        self._opportunity.ledger.spend(self._field_dim)
        # Dense outer-style accumulation: add last features into the whole field.
        delta = tuple(self._last_features)
        if all(item == 0 for item in delta):
            delta = tuple(1 if i == 0 else 0 for i in range(self._field_dim))
        self._proposal_seq += 1
        proposal_id = f"k1:{self.observations_seen}:{self._proposal_seq}"
        packed = _pack(delta)
        yield Proposal(
            proposal_id=proposal_id,
            kind="dense_field_accumulate",
            target="field",
            delta=delta,
            precision_request=_BULK,
            topology_operation=None,
            trigger=self._last_channel or "bootstrap",
            expected_value=0.0,
            cost_bytes=_packed_bytes(packed),
        )

    def _commit(self, proposal: Proposal) -> None:
        self._undo[proposal.proposal_id] = copy.deepcopy(self._durable_state())
        self._opportunity.ledger.spend(self._field_dim, precision_bits=self._field_dim * 3)
        delta = list(proposal.delta) if proposal.delta else [0] * self._field_dim
        if len(delta) < self._field_dim:
            delta = delta + [0] * (self._field_dim - len(delta))
        self._field = [
            native_low_bit_update(current, -float(step), _ALPHABET, learning_rate=1.0)
            for current, step in zip(self._field, delta[: self._field_dim], strict=True)
        ]
        self._resize()

    def _rollback(self, receipt: Receipt) -> None:
        prior = self._undo.pop(receipt.proposal_id, None)
        if prior is None:
            raise RuntimeError(f"{self.name} missing undo state for {receipt.proposal_id}")
        self._restore_durable(prior)

    def _durable_state(self) -> Any:
        packed = _pack(self._field)
        return {
            "form": "monolithic_plastic_field",
            "field_dim": self._field_dim,
            "field_packed": _packed_document(packed),
            "precision_map": dict(self._precision_map),
            "compiled_procedures": copy.deepcopy(self._compiled_procedures),
            "topology": None,
            "activation": False,
        }

    def _active_state(self) -> Any:
        return {
            "last_features": list(self._last_features),
            "last_channel": self._last_channel,
            "proposal_seq": self._proposal_seq,
        }

    def _restore_durable(self, state: Any) -> None:
        self._field_dim = int(state["field_dim"])
        self._field = _unpack(state["field_packed"])
        self._precision_map = dict(state["precision_map"])
        self._compiled_procedures = copy.deepcopy(state["compiled_procedures"])
        self._resize()

    def _restore_active(self, state: Any) -> None:
        self._last_features = list(state["last_features"])
        self._last_channel = str(state["last_channel"])
        self._proposal_seq = int(state["proposal_seq"])


# ---------------------------------------------------------------------------
# K2 — typed relation graph; durable change is an edge rewrite
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _TypedEdge:
    src: int
    dst: int
    etype: str
    value: int
    scope: str
    precision: str
    provenance: str


class K2_graph_plastic_field(MaterialBase):
    """Typed relation graph: each edge carries plastic value, scope, precision, provenance."""

    MECHANISM = "typed_per_edge_plastic_value_scope_and_precision"
    OWNED = "typed_per_edge_plastic_value_scope_and_precision"

    def __init__(self, opportunity: Opportunity, *, n_nodes: int = 8, **_options: Any) -> None:
        super().__init__(
            name="K2_graph_plastic_field",
            mechanism=self.MECHANISM,
            _opportunity=opportunity,
        )
        self._n_nodes = int(n_nodes)
        self._edges: list[_TypedEdge] = self._bootstrap_edges()
        self._precision_map = {"edge_value": _BULK}
        self._compiled_procedures: list[dict[str, Any]] = []
        self._node_act = [0] * self._n_nodes
        self._last_payload: tuple[int, ...] = ()
        self._last_channel = ""
        self._proposal_seq = 0
        self._undo: dict[str, Any] = {}
        self._hops = 2
        self._resize()

    def _bootstrap_edges(self) -> list[_TypedEdge]:
        edges: list[_TypedEdge] = []
        types = ("associates", "causes", "inhibits", "binds")
        for src in range(self._n_nodes):
            dst = (src + 1) % self._n_nodes
            edges.append(
                _TypedEdge(
                    src=src,
                    dst=dst,
                    etype=types[src % len(types)],
                    value=0,
                    scope=f"node:{src}",
                    precision=_BULK,
                    provenance="bootstrap",
                )
            )
            # A second typed edge giving non-local reach for multi-hop and rewire tests.
            far = (src + 3) % self._n_nodes
            if far != dst:
                edges.append(
                    _TypedEdge(
                        src=src,
                        dst=far,
                        etype=types[(src + 1) % len(types)],
                        value=0,
                        scope=f"node:{src}:far",
                        precision=_BULK,
                        provenance="bootstrap",
                    )
                )
        return edges

    def _resize(self) -> None:
        values = [edge.value for edge in self._edges]
        packed = _pack(values) if values else _pack([0])
        topology_bytes = 16 * len(self._edges)
        shell = 64 + sum(len(k) + len(v) for k, v in self._precision_map.items())
        self._opportunity.ledger.resize(_packed_bytes(packed) + topology_bytes + shell)

    def _propagate(self, seeds: Sequence[int]) -> list[int]:
        act = [_clamp(v) for v in seeds]
        for _ in range(self._hops):
            nxt = [0] * self._n_nodes
            for edge in self._edges:
                # Multi-hop typed message: source activation scaled by edge plastic value.
                message = int(act[edge.src]) * int(edge.value)
                if edge.etype == "inhibits":
                    message = -message
                nxt[edge.dst] = _clamp(nxt[edge.dst] + message + (1 if edge.value else 0) * int(act[edge.src] != 0))
            act = nxt
            self._opportunity.ledger.spend(len(self._edges))
        return act

    def _transition(self, observation: Observation) -> None:
        self._opportunity.ledger.spend(self._n_nodes + len(self._edges))
        features = _payload_features(observation.payload, self._n_nodes)
        channel_node = _channel_index(observation.channel, self._n_nodes)
        features[channel_node] = _clamp(features[channel_node] + 1)
        self._node_act = self._propagate(features)
        self._last_payload = tuple(int(x) for x in observation.payload)
        self._last_channel = observation.channel

    def _answer(self, probe: Probe) -> Answer:
        self._opportunity.ledger.spend(self._n_nodes, retrievals=1)
        seeds = _payload_features(probe.probe, self._n_nodes)
        activity = self._propagate(seeds)
        # Blend durable edge mass into the readout so edge rewrites affect answers.
        edge_mass = sum(edge.value for edge in self._edges)
        arity = max(1, probe.arity)
        value = tuple(_clamp(activity[i % self._n_nodes] + edge_mass) for i in range(arity))
        confidence = min(127, sum(abs(v) for v in activity))
        return Answer(probe_index=probe.index, value=value, confidence=confidence, abstained=False)

    def _propose(self) -> Iterable[Proposal]:
        if self.OWNED in self.frozen_mechanisms:
            return
        self._opportunity.ledger.spend(len(self._edges))
        # Choose the edge whose endpoints are most co-active under the last activation.
        best_index = 0
        best_score = -10**9
        for index, edge in enumerate(self._edges):
            score = abs(self._node_act[edge.src]) + abs(self._node_act[edge.dst])
            if self._last_channel and edge.etype[0] == (self._last_channel[:1] or "a"):
                score += 1
            if score > best_score:
                best_score = score
                best_index = index
        edge = self._edges[best_index]
        step = 1 if (self._node_act[edge.src] + self._node_act[edge.dst]) >= 0 else -1
        new_value = native_low_bit_update(edge.value, -float(step), _ALPHABET)
        self._proposal_seq += 1
        proposal_id = f"k2:{self.observations_seen}:{self._proposal_seq}"
        yield Proposal(
            proposal_id=proposal_id,
            kind="edge_plastic_rewrite",
            target=f"edge:{best_index}",
            delta=(best_index, new_value, edge.src, edge.dst),
            precision_request=_BULK,
            topology_operation="rewrite_edge_value",
            trigger=f"{edge.etype}:{edge.scope}",
            expected_value=0.0,
            cost_bytes=8,
        )

    def _commit(self, proposal: Proposal) -> None:
        if self.OWNED in self.frozen_mechanisms:
            return
        self._undo[proposal.proposal_id] = copy.deepcopy(self._durable_state())
        self._opportunity.ledger.spend(1, precision_bits=3)
        index = int(proposal.delta[0])
        new_value = _clamp(int(proposal.delta[1]))
        old = self._edges[index]
        self._edges[index] = _TypedEdge(
            src=old.src,
            dst=old.dst,
            etype=old.etype,
            value=new_value,
            scope=old.scope,
            precision=old.precision,
            provenance=f"rewrite:{proposal.proposal_id}",
        )
        self._resize()

    def _rollback(self, receipt: Receipt) -> None:
        prior = self._undo.pop(receipt.proposal_id, None)
        if prior is None:
            raise RuntimeError(f"{self.name} missing undo state for {receipt.proposal_id}")
        self._restore_durable(prior)

    def rewire_edge(self, edge_index: int, new_dst: int) -> None:
        """Topology rewrite used by locality separation tests (and legal K2 structure ops)."""
        old = self._edges[edge_index]
        self._edges[edge_index] = _TypedEdge(
            src=old.src,
            dst=int(new_dst) % self._n_nodes,
            etype=old.etype,
            value=old.value,
            scope=old.scope,
            precision=old.precision,
            provenance=f"rewire:{old.provenance}",
        )
        self._resize()

    def _durable_state(self) -> Any:
        values = [edge.value for edge in self._edges] or [0]
        packed = _pack(values)
        return {
            "form": "graph_plastic_field",
            "n_nodes": self._n_nodes,
            "edges": [
                {
                    "src": edge.src,
                    "dst": edge.dst,
                    "etype": edge.etype,
                    "value": edge.value,
                    "scope": edge.scope,
                    "precision": edge.precision,
                    "provenance": edge.provenance,
                }
                for edge in self._edges
            ],
            "edge_values_packed": _packed_document(packed),
            "precision_map": dict(self._precision_map),
            "compiled_procedures": copy.deepcopy(self._compiled_procedures),
            "hops": self._hops,
            "activation": False,
        }

    def _active_state(self) -> Any:
        return {
            "node_act": list(self._node_act),
            "last_payload": list(self._last_payload),
            "last_channel": self._last_channel,
            "proposal_seq": self._proposal_seq,
        }

    def _restore_durable(self, state: Any) -> None:
        self._n_nodes = int(state["n_nodes"])
        self._hops = int(state.get("hops", 2))
        self._edges = [
            _TypedEdge(
                src=int(row["src"]),
                dst=int(row["dst"]),
                etype=str(row["etype"]),
                value=_clamp(int(row["value"])),
                scope=str(row["scope"]),
                precision=str(row["precision"]),
                provenance=str(row["provenance"]),
            )
            for row in state["edges"]
        ]
        self._precision_map = dict(state["precision_map"])
        self._compiled_procedures = copy.deepcopy(state["compiled_procedures"])
        self._resize()

    def _restore_active(self, state: Any) -> None:
        self._node_act = list(state["node_act"])
        self._last_payload = tuple(int(x) for x in state["last_payload"])
        self._last_channel = str(state["last_channel"])
        self._proposal_seq = int(state["proposal_seq"])


# ---------------------------------------------------------------------------
# K3 — cellular lattice; durable change is a synchronous local neighbourhood rule
# ---------------------------------------------------------------------------


class K3_cellular_plastic_field(MaterialBase):
    """Fixed lattice of cells; durable updates are local within a bounded influence radius."""

    MECHANISM = "bounded_radius_local_neighbourhood_rule"
    OWNED = "bounded_radius_local_neighbourhood_rule"

    def __init__(
        self,
        opportunity: Opportunity,
        *,
        height: int = 4,
        width: int = 4,
        radius: int = 1,
        **_options: Any,
    ) -> None:
        super().__init__(
            name="K3_cellular_plastic_field",
            mechanism=self.MECHANISM,
            _opportunity=opportunity,
        )
        self._height = int(height)
        self._width = int(width)
        self._radius = int(radius)
        self._n_cells = self._height * self._width
        self._cells = [0] * self._n_cells
        self._precision_map = {"cell": _BULK}
        self._compiled_procedures: list[dict[str, Any]] = []
        # Active injection only. Non-local wiring is deliberately NOT durable and NOT used by the rule.
        self._active_injection = [0] * self._n_cells
        self._last_channel = ""
        self._proposal_seq = 0
        self._undo: dict[str, Any] = {}
        # Phantom long-range edge table: present so a non-local collapse can be detected by tests.
        # Correct K3 never reads this for durable state or the neighbourhood rule.
        self._non_local_wiring: list[tuple[int, int]] = []
        self._resize()

    def _index(self, row: int, col: int) -> int:
        return row * self._width + col

    def _coords(self, index: int) -> tuple[int, int]:
        return divmod(index, self._width)

    def _chebyshev(self, left: int, right: int) -> int:
        r0, c0 = self._coords(left)
        r1, c1 = self._coords(right)
        return max(abs(r0 - r1), abs(c0 - c1))

    def _neighbours(self, index: int) -> list[int]:
        row, col = self._coords(index)
        found: list[int] = []
        for dr in range(-self._radius, self._radius + 1):
            for dc in range(-self._radius, self._radius + 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = row + dr, col + dc
                if 0 <= rr < self._height and 0 <= cc < self._width:
                    found.append(self._index(rr, cc))
        return found

    def _resize(self) -> None:
        packed = _pack(self._cells)
        shell = 64 + 16 + sum(len(k) + len(v) for k, v in self._precision_map.items())
        self._opportunity.ledger.resize(_packed_bytes(packed) + shell)

    def _local_rule(self, cells: Sequence[int], injection: Sequence[int]) -> list[int]:
        """Synchronous neighbourhood update. Only cells within radius contribute."""
        nxt = [0] * self._n_cells
        for index in range(self._n_cells):
            total = int(cells[index]) + int(injection[index])
            for neighbour in self._neighbours(index):
                total += int(cells[neighbour])
            # Intentionally ignore self._non_local_wiring — that is the K2/K3 separation.
            nxt[index] = _clamp(total)
        self._opportunity.ledger.spend(self._n_cells * (1 + 4 * self._radius * self._radius))
        return nxt

    def _transition(self, observation: Observation) -> None:
        self._opportunity.ledger.spend(self._n_cells)
        injection = [0] * self._n_cells
        features = _payload_features(observation.payload, self._n_cells)
        channel_cell = _channel_index(observation.channel, self._n_cells)
        for i, value in enumerate(features):
            injection[i] = _clamp(value)
        injection[channel_cell] = _clamp(injection[channel_cell] + 1)
        self._active_injection = injection
        self._last_channel = observation.channel

    def _answer(self, probe: Probe) -> Answer:
        self._opportunity.ledger.spend(self._n_cells, retrievals=1)
        seeds = _payload_features(probe.probe, self._n_cells)
        # Readout uses local neighbourhood sums around seed peaks — not a global edge sum.
        peak = max(range(self._n_cells), key=lambda i: abs(seeds[i]))
        local = [self._cells[peak]] + [self._cells[n] for n in self._neighbours(peak)]
        mass = sum(local)
        arity = max(1, probe.arity)
        value = tuple(_clamp(mass + seeds[i % self._n_cells]) for i in range(arity))
        confidence = min(127, abs(mass))
        return Answer(probe_index=probe.index, value=value, confidence=confidence, abstained=False)

    def _propose(self) -> Iterable[Proposal]:
        if self.OWNED in self.frozen_mechanisms:
            return
        self._opportunity.ledger.spend(self._n_cells)
        proposed = self._local_rule(self._cells, self._active_injection)
        delta = tuple(proposed)
        self._proposal_seq += 1
        proposal_id = f"k3:{self.observations_seen}:{self._proposal_seq}"
        packed = _pack(delta)
        yield Proposal(
            proposal_id=proposal_id,
            kind="local_neighbourhood_sync",
            target="lattice",
            delta=delta,
            precision_request=_BULK,
            topology_operation=None,
            trigger=self._last_channel or "bootstrap",
            expected_value=0.0,
            cost_bytes=_packed_bytes(packed),
        )

    def _commit(self, proposal: Proposal) -> None:
        if self.OWNED in self.frozen_mechanisms:
            return
        self._undo[proposal.proposal_id] = copy.deepcopy(self._durable_state())
        self._opportunity.ledger.spend(self._n_cells, precision_bits=self._n_cells * 3)
        delta = list(proposal.delta)
        if len(delta) != self._n_cells:
            delta = (delta + [0] * self._n_cells)[: self._n_cells]
        # Commit is the synchronous local rule result (already local); store as cell state.
        self._cells = [_clamp(v) for v in delta]
        self._resize()

    def _rollback(self, receipt: Receipt) -> None:
        prior = self._undo.pop(receipt.proposal_id, None)
        if prior is None:
            raise RuntimeError(f"{self.name} missing undo state for {receipt.proposal_id}")
        self._restore_durable(prior)

    def rewire_non_neighbour(self, src: int, dst: int) -> None:
        """Record a non-local link. Correct K3 durable digest must ignore this."""
        if self._chebyshev(src, dst) <= self._radius:
            raise ValueError("pair is inside the influence radius; use a non-neighbour pair")
        self._non_local_wiring.append((int(src), int(dst)))

    def find_non_neighbour_pair(self) -> tuple[int, int]:
        for left in range(self._n_cells):
            for right in range(left + 1, self._n_cells):
                if self._chebyshev(left, right) > self._radius:
                    return left, right
        raise RuntimeError("lattice too small to expose a non-neighbour pair")

    def _durable_state(self) -> Any:
        packed = _pack(self._cells)
        return {
            "form": "cellular_field",
            "height": self._height,
            "width": self._width,
            "radius": self._radius,
            "cells_packed": _packed_document(packed),
            "precision_map": dict(self._precision_map),
            "compiled_procedures": copy.deepcopy(self._compiled_procedures),
            # Topology is the fixed lattice geometry only — never an arbitrary edge list.
            "topology": {"kind": "fixed_lattice", "height": self._height, "width": self._width, "radius": self._radius},
            "activation": False,
        }

    def _active_state(self) -> Any:
        return {
            "active_injection": list(self._active_injection),
            "last_channel": self._last_channel,
            "proposal_seq": self._proposal_seq,
            # Exposed so tests can see it is non-durable; not part of durable digest.
            "non_local_wiring": list(self._non_local_wiring),
        }

    def _restore_durable(self, state: Any) -> None:
        self._height = int(state["height"])
        self._width = int(state["width"])
        self._radius = int(state["radius"])
        self._n_cells = self._height * self._width
        self._cells = _unpack(state["cells_packed"])
        self._precision_map = dict(state["precision_map"])
        self._compiled_procedures = copy.deepcopy(state["compiled_procedures"])
        self._resize()

    def _restore_active(self, state: Any) -> None:
        self._active_injection = list(state["active_injection"])
        self._last_channel = str(state["last_channel"])
        self._proposal_seq = int(state["proposal_seq"])
        self._non_local_wiring = [tuple(pair) for pair in state.get("non_local_wiring", ())]


# ---------------------------------------------------------------------------
# K5 — input-dependent bounded recurrence; durable change updates recurrence params
# ---------------------------------------------------------------------------


class K5_recurrent_state_space_plastic_field(MaterialBase):
    """Small latent carried across observations; durable change is a slow param update."""

    MECHANISM = "input_dependent_bounded_recurrence"
    OWNED = "input_dependent_bounded_recurrence"

    def __init__(self, opportunity: Opportunity, *, latent_dim: int = 4, **_options: Any) -> None:
        super().__init__(
            name="K5_recurrent_state_space_plastic_field",
            mechanism=self.MECHANISM,
            _opportunity=opportunity,
        )
        self._latent_dim = int(latent_dim)
        # Durable recurrence parameters A (d x d row-major) and B (d), readout C (d).
        self._A = [0] * (self._latent_dim * self._latent_dim)
        self._B = [0] * self._latent_dim
        self._C = [1 if i == 0 else 0 for i in range(self._latent_dim)]
        # Seed a mild identity-like diagonal so recurrence is input-dependent and non-zero.
        for i in range(self._latent_dim):
            self._A[i * self._latent_dim + i] = 1
        self._precision_map = {"A": _BULK, "B": _BULK, "C": _BULK}
        self._compiled_procedures: list[dict[str, Any]] = []
        self._z = [0] * self._latent_dim
        self._last_input = [0] * self._latent_dim
        self._last_channel = ""
        self._proposal_seq = 0
        self._undo: dict[str, Any] = {}
        self._resize()

    def _resize(self) -> None:
        packed = _pack(self._A + self._B + self._C)
        shell = 64 + sum(len(k) + len(v) for k, v in self._precision_map.items())
        self._opportunity.ledger.resize(_packed_bytes(packed) + shell)

    def _step_latent(self, z: Sequence[int], x: Sequence[int]) -> list[int]:
        nxt = [0] * self._latent_dim
        for row in range(self._latent_dim):
            row_weights = self._A[row * self._latent_dim : (row + 1) * self._latent_dim]
            recurrent = int(multiplication_light_dot(row_weights, z)["value"])
            drive = int(self._B[row]) * int(x[row])
            nxt[row] = _clamp(recurrent + drive)
        self._opportunity.ledger.spend(self._latent_dim * self._latent_dim)
        return nxt

    def _transition(self, observation: Observation) -> None:
        self._opportunity.ledger.spend(self._latent_dim)
        x = _payload_features(observation.payload, self._latent_dim)
        channel_i = _channel_index(observation.channel, self._latent_dim)
        x[channel_i] = _clamp(x[channel_i] + 1)
        self._last_input = x
        self._last_channel = observation.channel
        # Active recurrence only — parameters stay durable and unchanged here.
        self._z = self._step_latent(self._z, x)

    def _answer(self, probe: Probe) -> Answer:
        self._opportunity.ledger.spend(self._latent_dim, retrievals=1)
        x = _payload_features(probe.probe, self._latent_dim)
        z_probe = self._step_latent(self._z, x)
        readout = int(multiplication_light_dot(self._C, z_probe)["value"])
        arity = max(1, probe.arity)
        value = tuple(_clamp(readout + z_probe[i % self._latent_dim]) for i in range(arity))
        confidence = min(127, abs(readout))
        return Answer(probe_index=probe.index, value=value, confidence=confidence, abstained=False)

    def _propose(self) -> Iterable[Proposal]:
        if self.OWNED in self.frozen_mechanisms:
            return
        self._opportunity.ledger.spend(self._latent_dim * self._latent_dim)
        # Slow param update from last input residual: push B toward last input, A toward outer(z, z).
        residual = [_clamp(int(self._last_input[i]) - int(self._z[i])) for i in range(self._latent_dim)]
        new_B = [
            native_low_bit_update(self._B[i], -float(residual[i]), _ALPHABET, learning_rate=1.0)
            for i in range(self._latent_dim)
        ]
        new_A = list(self._A)
        for row in range(self._latent_dim):
            for col in range(self._latent_dim):
                idx = row * self._latent_dim + col
                grad = float(self._z[row]) * float(residual[col]) if self._z[row] else float(residual[row])
                new_A[idx] = native_low_bit_update(self._A[idx], -grad, _ALPHABET, learning_rate=0.5)
        new_C = [
            native_low_bit_update(self._C[i], -float(residual[i]), _ALPHABET, learning_rate=0.5)
            for i in range(self._latent_dim)
        ]
        delta = tuple(new_A + new_B + new_C)
        self._proposal_seq += 1
        proposal_id = f"k5:{self.observations_seen}:{self._proposal_seq}"
        packed = _pack(delta)
        yield Proposal(
            proposal_id=proposal_id,
            kind="recurrence_parameter_update",
            target="recurrence_params",
            delta=delta,
            precision_request=_BULK,
            topology_operation=None,
            trigger=self._last_channel or "bootstrap",
            expected_value=0.0,
            cost_bytes=_packed_bytes(packed),
        )

    def _commit(self, proposal: Proposal) -> None:
        if self.OWNED in self.frozen_mechanisms:
            return
        self._undo[proposal.proposal_id] = copy.deepcopy(self._durable_state())
        self._opportunity.ledger.spend(len(proposal.delta), precision_bits=len(proposal.delta) * 3)
        d = self._latent_dim
        data = list(proposal.delta)
        need = d * d + d + d
        if len(data) < need:
            data = data + [0] * (need - len(data))
        self._A = [_clamp(v) for v in data[: d * d]]
        self._B = [_clamp(v) for v in data[d * d : d * d + d]]
        self._C = [_clamp(v) for v in data[d * d + d : d * d + 2 * d]]
        self._resize()

    def _rollback(self, receipt: Receipt) -> None:
        prior = self._undo.pop(receipt.proposal_id, None)
        if prior is None:
            raise RuntimeError(f"{self.name} missing undo state for {receipt.proposal_id}")
        self._restore_durable(prior)

    def _durable_state(self) -> Any:
        packed = _pack(self._A + self._B + self._C)
        return {
            "form": "recurrent_state_space_field",
            "latent_dim": self._latent_dim,
            "params_packed": _packed_document(packed),
            "precision_map": dict(self._precision_map),
            "compiled_procedures": copy.deepcopy(self._compiled_procedures),
            "topology": None,
            "activation": False,
        }

    def _active_state(self) -> Any:
        return {
            "z": list(self._z),
            "last_input": list(self._last_input),
            "last_channel": self._last_channel,
            "proposal_seq": self._proposal_seq,
        }

    def _restore_durable(self, state: Any) -> None:
        self._latent_dim = int(state["latent_dim"])
        params = _unpack(state["params_packed"])
        d = self._latent_dim
        self._A = params[: d * d]
        self._B = params[d * d : d * d + d]
        self._C = params[d * d + d : d * d + 2 * d]
        self._precision_map = dict(state["precision_map"])
        self._compiled_procedures = copy.deepcopy(state["compiled_procedures"])
        self._resize()

    def _restore_active(self, state: Any) -> None:
        self._z = list(state["z"])
        self._last_input = list(state["last_input"])
        self._last_channel = str(state["last_channel"])
        self._proposal_seq = int(state["proposal_seq"])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _factory_k1(opportunity: Opportunity, **options: Any) -> K1_monolithic_plastic_field:
    return K1_monolithic_plastic_field(opportunity, **options)


def _factory_k2(opportunity: Opportunity, **options: Any) -> K2_graph_plastic_field:
    return K2_graph_plastic_field(opportunity, **options)


def _factory_k3(opportunity: Opportunity, **options: Any) -> K3_cellular_plastic_field:
    return K3_cellular_plastic_field(opportunity, **options)


def _factory_k5(opportunity: Opportunity, **options: Any) -> K5_recurrent_state_space_plastic_field:
    return K5_recurrent_state_space_plastic_field(opportunity, **options)


register("K1_monolithic_plastic_field", _factory_k1)
register("K2_graph_plastic_field", _factory_k2)
register("K3_cellular_plastic_field", _factory_k3)
register("K5_recurrent_state_space_plastic_field", _factory_k5)


__all__ = [
    "K1_monolithic_plastic_field",
    "K2_graph_plastic_field",
    "K3_cellular_plastic_field",
    "K5_recurrent_state_space_plastic_field",
]
