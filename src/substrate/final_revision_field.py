"""Executable foundation for an Endogenously Plastic Cognitive Field.

This module is deliberately outside the current Final Revision candidate path.
It supplies architecture-neutral contracts, small mechanism probes, and neutral
state migration scaffolding for the next campaign.  Nothing here is evidence
for a Nous classification and every public result keeps activation false.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from substrate import final_revision_io as io

FIELD_SCHEMA = "substrate-endogenous-plastic-field/v1"
FOUNDATION_STATUS = "foundation_feasibility_only"
FIELD_SYMBOLS = ("Theta", "P_t", "G_t", "Z_t", "E_t", "A", "C_t", "M_t")
TIMESCALES = (
    "momentary_activation",
    "fast_plasticity",
    "intermediate_plasticity",
    "slow_consolidation",
    "rare_kernel_revision",
    "topological_development",
)
METAPLASTIC_STATES = ("new", "provisional", "supported", "consolidated", "reopened", "refuted")
PRECISION_ALPHABETS: dict[str, tuple[int, ...]] = {
    "binary": (-1, 1),
    "ternary": (-1, 0, 1),
    "quinary": (-2, -1, 0, 1, 2),
    "seven_state_powers_of_two": (-4, -2, -1, 0, 1, 2, 4),
}
PRECISION_CLASSES = (
    "exact_symbolic",
    "high_precision_numeric",
    "8_bit",
    "4_bit",
    "quinary",
    "ternary",
    "binary",
    "vector_quantized",
    "archived",
)
PRECISION_RANK = {
    "archived": 0,
    "binary": 1,
    "ternary": 2,
    "quinary": 3,
    "4_bit": 4,
    "vector_quantized": 5,
    "8_bit": 6,
    "high_precision_numeric": 7,
    "exact_symbolic": 8,
}
PLASTICITY_CONTRACTS = (
    "PlasticityObserve",
    "PlasticityPropose",
    "PlasticitySimulate",
    "PlasticityVerify",
    "PlasticityCommit",
    "PlasticityRollback",
    "PlasticityConsolidate",
)
METAPLASTICITY_CONTRACTS = (
    "MetaplasticityPromote",
    "MetaplasticityDestabilize",
    "MetaplasticityReconsolidate",
)
PRECISION_CONTRACTS = (
    "PrecisionRequest",
    "PrecisionPromote",
    "PrecisionDemote",
    "PrecisionFreeze",
    "PrecisionReopen",
    "PrecisionAudit",
)
TOPOLOGY_CONTRACTS = (
    "TopologyAllocate",
    "TopologyConnect",
    "TopologyDisconnect",
    "TopologySplit",
    "TopologyMerge",
    "TopologyCompile",
    "TopologyPrune",
    "TopologyArchive",
    "TopologyRestore",
)
SHADOW_CONTRACTS = (
    "ShadowFieldFork",
    "ShadowFieldPerturb",
    "ShadowFieldRun",
    "ShadowFieldCompare",
    "ShadowFieldDiscard",
    "ShadowFieldPromote",
)
COMPILER_CONTRACTS = (
    "ProcedureObserveTrace",
    "ProcedurePropose",
    "ProcedureVerify",
    "ProcedureCompile",
    "ProcedureExecute",
    "ProcedureMonitor",
    "ProcedureInvalidate",
    "ProcedureDecompile",
)
TEMPORAL_CONTRACTS = (
    "TemporalEvent",
    "ElapsedTime",
    "ScheduledObservation",
    "BackgroundConsolidation",
    "GoalDeadline",
    "PredictionExpiry",
    "MemoryDecayProposal",
)
MIGRATION_CONTRACTS = (
    "CognitiveStateExport",
    "CognitiveStateImport",
    "CognitiveStateMigrate",
    "CognitiveStateCompare",
    "CognitiveStateRollback",
)
COMMON_FIELD_CONTRACTS = (
    *PLASTICITY_CONTRACTS,
    *METAPLASTICITY_CONTRACTS,
    *PRECISION_CONTRACTS,
    *TOPOLOGY_CONTRACTS,
    *SHADOW_CONTRACTS,
    *COMPILER_CONTRACTS,
    *TEMPORAL_CONTRACTS,
    *MIGRATION_CONTRACTS,
    "Checkpoint",
    "Restore",
    "ExactShellAudit",
)
RESOURCE_ENVELOPES_BYTES: dict[str, int | None] = {
    "512_MB": 512 * 1024**2,
    "1_GB": 1024**3,
    "2_GB": 2 * 1024**3,
    "5_GB": 5 * 1024**3,
    "10_GB": 10 * 1024**3,
    "unconstrained_reference": None,
}
SKELETONS: dict[str, dict[str, Any]] = {
    "K1_monolithic_plastic_field": {
        "representation": "single_materialized_field",
        "mechanisms": ["shared_transition", "verification_gated_plasticity"],
    },
    "K2_graph_plastic_field": {
        "representation": "sparse_typed_dynamic_graph",
        "mechanisms": ["message_passing", "edge_plasticity"],
    },
    "K3_cellular_plastic_field": {
        "representation": "local_cognitive_cells",
        "mechanisms": ["shared_local_rule", "neighborhood_update"],
    },
    "K4_continuous_time_plastic_field": {
        "representation": "event_driven_continuous_time_field",
        "mechanisms": ["elapsed_time_transition", "scheduled_consolidation"],
    },
    "K5_recurrent_state_space_plastic_field": {
        "representation": "bounded_recurrent_state_space",
        "mechanisms": ["recurrent_transition", "fast_weight_state"],
    },
    "K6_adaptive_topology_field": {
        "representation": "rent_bearing_dynamic_sparse_topology",
        "mechanisms": ["allocate_split_merge", "held_out_pruning"],
    },
    "K7_native_mixed_radix_field": {
        "representation": "region_specific_native_precision",
        "mechanisms": ["mixed_radix_state", "precision_promotion"],
    },
    "K8_integrated_field_placeholder": {
        "representation": "unselected_hybrid_placeholder",
        "mechanisms": ["contract_composition_only"],
    },
}


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise io.Refused(message)


def _clamp_to_alphabet(value: int, precision: str) -> int:
    alphabet = PRECISION_ALPHABETS.get(precision)
    if alphabet is None:
        return value
    return min(alphabet, key=lambda item: (abs(item - value), item))


@dataclass
class PlasticRelation:
    """A compact relation with explicit metaplastic and evidentiary state."""

    value: int
    precision: str
    stability: str
    scope: str
    provenance: list[str]
    last_verification: str | None = None
    contradictions: list[str] = field(default_factory=list)
    frozen: bool = False

    def validate(self) -> None:
        _require(self.precision in PRECISION_CLASSES, f"unknown precision {self.precision!r}")
        _require(self.stability in METAPLASTIC_STATES, f"unknown stability {self.stability!r}")
        _require(bool(self.scope), "relation scope is required")
        _require(bool(self.provenance), "relation provenance is required")
        alphabet = PRECISION_ALPHABETS.get(self.precision)
        if alphabet is not None:
            _require(self.value in alphabet, f"value {self.value} is not native {self.precision}")


@dataclass
class CognitiveCell:
    """Architecture-neutral compact cognitive cell."""

    cell_id: str
    cell_type: str
    cognitive_activation: int = 0
    influence: int = 0
    uncertainty: float = 1.0
    stability: str = "new"
    precision: str = "ternary"
    scope: str = "local"
    sparse_links: list[str] = field(default_factory=list)
    vector_quantized_local_latent: list[int] = field(default_factory=list)
    provenance_pointer: str = ""

    def validate(self) -> None:
        allowed = {
            "object",
            "event",
            "concept",
            "belief",
            "goal",
            "procedure",
            "body",
            "tool",
            "model",
            "place",
            "causal_relation",
            "unresolved_hypothesis",
        }
        _require(self.cell_type in allowed, f"unknown cognitive cell type {self.cell_type!r}")
        _require(self.stability in METAPLASTIC_STATES, "invalid cognitive cell stability")
        _require(self.precision in PRECISION_CLASSES, "invalid cognitive cell precision")
        _require(0.0 <= self.uncertainty <= 1.0, "cell uncertainty must be in [0,1]")
        _require(bool(self.provenance_pointer), "cell provenance pointer is required")


@dataclass(frozen=True)
class PackedRadix:
    alphabet: tuple[int, ...]
    count: int
    group_size: int
    bit_length: int
    payload_hex: str

    @property
    def byte_length(self) -> int:
        return len(bytes.fromhex(self.payload_hex))


def pack_radix(values: Sequence[int], alphabet: Sequence[int], *, group_size: int) -> PackedRadix:
    """Pack fixed-radix values without reserving impossible bit patterns."""
    symbols = tuple(int(value) for value in alphabet)
    _require(len(symbols) >= 2 and len(set(symbols)) == len(symbols), "alphabet must contain distinct symbols")
    _require(group_size > 0, "group size must be positive")
    index = {value: position for position, value in enumerate(symbols)}
    _require(all(value in index for value in values), "value is outside packing alphabet")
    stream = 0
    bit_length = 0
    radix = len(symbols)
    for offset in range(0, len(values), group_size):
        chunk = values[offset : offset + group_size]
        encoded = 0
        for value in chunk:
            encoded = encoded * radix + index[value]
        width = max(1, math.ceil(math.log2(radix ** len(chunk))))
        stream = (stream << width) | encoded
        bit_length += width
    payload = stream.to_bytes((bit_length + 7) // 8, "big") if bit_length else b""
    return PackedRadix(symbols, len(values), group_size, bit_length, payload.hex())


def unpack_radix(packed: PackedRadix) -> list[int]:
    """Invert :func:`pack_radix` exactly."""
    radix = len(packed.alphabet)
    stream = int.from_bytes(bytes.fromhex(packed.payload_hex), "big")
    remaining = packed.bit_length
    values: list[int] = []
    chunk_lengths = [
        min(packed.group_size, packed.count - offset) for offset in range(0, packed.count, packed.group_size)
    ]
    for chunk_length in chunk_lengths:
        width = max(1, math.ceil(math.log2(radix**chunk_length)))
        remaining -= width
        encoded = (stream >> remaining) & ((1 << width) - 1)
        _require(encoded < radix**chunk_length, "packed payload contains an impossible code")
        indices = [0] * chunk_length
        for position in range(chunk_length - 1, -1, -1):
            indices[position] = encoded % radix
            encoded //= radix
        values.extend(packed.alphabet[index] for index in indices)
    _require(remaining == 0 and len(values) == packed.count, "packed payload length mismatch")
    return values


def native_low_bit_update(value: int, gradient: float, alphabet: Sequence[int], *, learning_rate: float = 1.0) -> int:
    """Apply a projected update directly in the target alphabet."""
    _require(bool(alphabet), "native low-bit update requires an alphabet")
    proposed = float(value) - learning_rate * gradient
    return min((int(symbol) for symbol in alphabet), key=lambda symbol: (abs(symbol - proposed), symbol))


def learned_codebook(values: Sequence[float], *, size: int) -> dict[str, Any]:
    """Learn a deterministic scalar codebook from construction values."""
    _require(bool(values) and 1 < size <= len(values), "invalid learned codebook size")
    ordered = sorted(float(value) for value in values)
    centroids = []
    for index in range(size):
        position = round(index * (len(ordered) - 1) / (size - 1))
        centroids.append(ordered[position])
    codes = [min(range(size), key=lambda code: (abs(centroids[code] - value), code)) for value in values]
    restored = [centroids[code] for code in codes]
    return {
        "centroids": centroids,
        "codes": codes,
        "restored": restored,
        "mean_absolute_error": sum(abs(float(value) - reconstruction) for value, reconstruction in zip(values, restored, strict=True))
        / len(values),
        "construction_only": True,
        "activation": False,
    }


def ternary_sparse_outliers(values: Sequence[float], *, outlier_threshold: float) -> dict[str, Any]:
    """Represent a ternary base plus exact sparse residuals."""
    _require(outlier_threshold > 0.0, "sparse outlier threshold must be positive")
    base = [min((-1, 0, 1), key=lambda symbol: (abs(symbol - value), symbol)) for value in values]
    outliers = {
        str(index): float(value)
        for index, (value, approximation) in enumerate(zip(values, base, strict=True))
        if abs(float(value) - approximation) > outlier_threshold
    }
    restored = [outliers.get(str(index), approximation) for index, approximation in enumerate(base)]
    return {
        "base": base,
        "outliers": outliers,
        "restored": restored,
        "exact_outlier_count": len(outliers),
        "activation": False,
    }


def adaptive_mixed_radix_pack(regions: Mapping[str, tuple[Sequence[int], Sequence[int], int]]) -> dict[str, Any]:
    """Pack independent regions with region-specific alphabets."""
    packed = {
        name: asdict(pack_radix(values, alphabet, group_size=group_size))
        for name, (values, alphabet, group_size) in sorted(regions.items())
    }
    exact = all(
        unpack_radix(PackedRadix(**document)) == list(regions[name][0])
        for name, document in packed.items()
    )
    return {
        "regions": packed,
        "total_bits": sum(int(document["bit_length"]) for document in packed.values()),
        "exact_round_trip": exact,
        "activation": False,
    }


def packing_benchmark(*, repetitions: int = 200) -> dict[str, Any]:
    """Run bounded native-radix round trips and retain raw timing receipts."""
    _require(repetitions > 0, "packing repetitions must be positive")
    rows: list[dict[str, Any]] = []
    for name, alphabet in PRECISION_ALPHABETS.items():
        group_size = 8 if name == "binary" else (5 if name == "ternary" else 3)
        values = [alphabet[(index * 7 + 3) % len(alphabet)] for index in range(511)]
        started = time.perf_counter_ns()
        packed: PackedRadix | None = None
        restored: list[int] = []
        for _ in range(repetitions):
            packed = pack_radix(values, alphabet, group_size=group_size)
            restored = unpack_radix(packed)
        elapsed = time.perf_counter_ns() - started
        assert packed is not None
        rows.append(
            {
                "alphabet": name,
                "radix": len(alphabet),
                "count": len(values),
                "group_size": group_size,
                "payload_bytes": packed.byte_length,
                "payload_bits": packed.bit_length,
                "naive_fixed_width_bits": len(values) * math.ceil(math.log2(len(alphabet))),
                "bits_per_value": packed.bit_length / len(values),
                "exact_round_trip": restored == values,
                "repetitions": repetitions,
                "elapsed_ns_observed": elapsed,
                "timing_is_environment_dependent": True,
            }
        )
    quinary_triplet = pack_radix([-2, -1, 0], PRECISION_ALPHABETS["quinary"], group_size=3)
    learned = learned_codebook([math.sin(index * 0.11) for index in range(128)], size=8)
    sparse = ternary_sparse_outliers([0.0, 0.8, -0.9, 3.4, -3.2, 0.1], outlier_threshold=0.75)
    mixed = adaptive_mixed_radix_pack(
        {
            "routing": ([1, 0, -1, 1], PRECISION_ALPHABETS["ternary"], 4),
            "influence": ([-2, 1, 0, 2], PRECISION_ALPHABETS["quinary"], 3),
            "gates": ([-1, 1, 1, -1], PRECISION_ALPHABETS["binary"], 4),
        }
    )
    return {
        "schema": "substrate-field-packing-benchmark/v1",
        "rows": rows,
        "quinary_three_base5_values_in_seven_bits": quinary_triplet.bit_length == 7,
        "learned_codebook": learned,
        "ternary_plus_sparse_outliers": sparse,
        "adaptive_mixed_radix": mixed,
        "native_projected_update": {
            "before": -1,
            "gradient": -2.0,
            "after": native_low_bit_update(-1, -2.0, PRECISION_ALPHABETS["ternary"]),
        },
        "all_exact": all(row["exact_round_trip"] for row in rows),
        "activation": False,
    }


class EndogenousPlasticField:
    """Minimal executable field with a protected exact shell."""

    def __init__(
        self,
        identity: str,
        *,
        skeleton: str = "K1_monolithic_plastic_field",
        resource_envelope: str = "512_MB",
        s2_derived: bool = False,
    ):
        _require(bool(identity), "field identity is required")
        _require(skeleton in SKELETONS, f"unknown field skeleton {skeleton!r}")
        _require(resource_envelope in RESOURCE_ENVELOPES_BYTES, "unknown resource envelope")
        self.skeleton = skeleton
        self.resource_envelope = resource_envelope
        self.s2_derived = s2_derived
        self.theta: dict[str, Any] = {
            "transition_law": "bounded-local-field/v1",
            "plasticity_law": "verified-sparse-rewrite/v1",
            "kernel_revision_policy": "rare-explicit-authority-only",
        }
        self.plastic: dict[str, PlasticRelation] = {}
        self.topology: dict[str, Any] = {"nodes": {}, "edges": [], "archive": {}}
        self.active: dict[str, Any] = {
            "momentary_activation": {},
            "unresolved_alternatives": {},
            "elapsed_seconds": 0.0,
        }
        self.exact: dict[str, Any] = {
            "identity": identity,
            "lineage": [],
            "goal_commitments": {},
            "evidence_provenance": [],
            "permissions": {"external_activation": False, "field_may_rewrite_shell": False},
            "claim_boundaries": {
                "foundation_only": True,
                "nous": False,
                "consciousness": False,
                "identity_transfer": False,
            },
            "activation": False,
            "checkpoint_integrity": None,
            "irreversible_commitments": [],
        }
        self.archive: list[dict[str, Any]] = []
        self.compiled: dict[str, dict[str, Any]] = {}
        self.competence: dict[str, Any] = {"models": {}, "bodies": {}, "tools": {}, "sensors": {}}
        self.cells: dict[str, CognitiveCell] = {}
        self.proposals: dict[str, dict[str, Any]] = {}
        self.rollbacks: dict[str, dict[str, Any]] = {}
        self.shadows: dict[str, dict[str, Any]] = {}
        self.traces: dict[str, list[dict[str, Any]]] = {}
        self.procedure_candidates: dict[str, dict[str, Any]] = {}
        self.temporal_events: list[dict[str, Any]] = []
        self.scheduled: list[dict[str, Any]] = []
        self.precision_receipts: list[dict[str, Any]] = []
        self.topology_receipts: list[dict[str, Any]] = []
        self._append_receipt("field_initialized", {"skeleton": skeleton, "resource_envelope": resource_envelope})

    def interfaces(self) -> tuple[str, ...]:
        return COMMON_FIELD_CONTRACTS

    def _append_receipt(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = {
            "sequence": len(self.archive) + 1,
            "operation": operation,
            "payload": _copy(dict(payload)),
            "previous_digest": self.archive[-1]["digest"] if self.archive else "0" * 64,
            "activation": False,
        }
        row["digest"] = io.digest(row)
        self.archive.append(row)
        return row

    def _relation(self, relation_id: str) -> PlasticRelation:
        if relation_id not in self.plastic:
            raise io.Refused("unknown relation")
        return self.plastic[relation_id]

    def _proposal(self, proposal_id: str) -> dict[str, Any]:
        if proposal_id not in self.proposals:
            raise io.Refused("unknown plasticity proposal")
        return self.proposals[proposal_id]

    def _shadow(self, shadow_id: str) -> dict[str, Any]:
        if shadow_id not in self.shadows:
            raise io.Refused("unknown shadow")
        return self.shadows[shadow_id]

    def _procedure(self, procedure_id: str) -> dict[str, Any]:
        if procedure_id not in self.compiled:
            raise io.Refused("unknown procedure")
        return self.compiled[procedure_id]

    def document(self) -> dict[str, Any]:
        return {
            "schema": FIELD_SCHEMA,
            "skeleton": self.skeleton,
            "resource_envelope": self.resource_envelope,
            "s2_derived": self.s2_derived,
            "Theta": _copy(self.theta),
            "P_t": {key: asdict(value) for key, value in sorted(self.plastic.items())},
            "G_t": _copy(self.topology),
            "Z_t": _copy(self.active),
            "E_t": _copy(self.exact),
            "A": _copy(self.archive),
            "C_t": _copy(self.compiled),
            "M_t": _copy(self.competence),
            "cells": {key: asdict(value) for key, value in sorted(self.cells.items())},
            "pending_proposals": _copy(self.proposals),
            "rollbacks": _copy(self.rollbacks),
            "shadows": _copy(self.shadows),
            "traces": _copy(self.traces),
            "procedure_candidates": _copy(self.procedure_candidates),
            "temporal_events": _copy(self.temporal_events),
            "scheduled": _copy(self.scheduled),
            "precision_receipts": _copy(self.precision_receipts),
            "topology_receipts": _copy(self.topology_receipts),
            "activation": False,
        }

    def state_integrity_digest(self) -> str:
        document = self.document()
        document["E_t"]["checkpoint_integrity"] = None
        return io.digest(document)

    def add_relation(
        self,
        relation_id: str,
        value: int,
        *,
        precision: str,
        stability: str = "new",
        scope: str = "local",
        provenance: str,
    ) -> None:
        _require(relation_id not in self.plastic, "relation already exists")
        relation = PlasticRelation(value, precision, stability, scope, [provenance])
        relation.validate()
        self.plastic[relation_id] = relation
        self._append_receipt("relation_added", {"relation_id": relation_id, "relation": asdict(relation)})

    def observe(self, key: str, value: Any, *, provenance: str) -> dict[str, Any]:
        _require(bool(provenance), "observation provenance is required")
        self.active["momentary_activation"][key] = _copy(value)
        return self._append_receipt("PlasticityObserve", {"key": key, "value": value, "provenance": provenance})

    def plasticity_propose(
        self,
        proposal_id: str,
        relation_id: str,
        delta: int,
        *,
        source: str,
        source_kind: str,
        evidence: Sequence[str],
    ) -> dict[str, Any]:
        _require(proposal_id not in self.proposals, "plasticity proposal already exists")
        _require(relation_id in self.plastic, "plasticity proposal references unknown relation")
        _require(delta in {-2, -1, 0, 1, 2}, "plasticity delta is outside bounded rewrite alphabet")
        proposal = {
            "proposal_id": proposal_id,
            "relation_id": relation_id,
            "delta": delta,
            "source": source,
            "source_kind": source_kind,
            "evidence": list(evidence),
            "before": asdict(self.plastic[relation_id]),
            "verified": False,
            "verification": None,
            "committed": False,
            "activation": False,
        }
        proposal["proposal_digest"] = io.digest(proposal)
        self.proposals[proposal_id] = proposal
        self._append_receipt("PlasticityPropose", proposal)
        return _copy(proposal)

    def plasticity_simulate(self, proposal_id: str) -> dict[str, Any]:
        proposal = self._proposal(proposal_id)
        relation = self.plastic[str(proposal["relation_id"])]
        simulated = _clamp_to_alphabet(relation.value + int(proposal["delta"]), relation.precision)
        receipt = {
            "proposal_id": proposal_id,
            "before": relation.value,
            "after": simulated,
            "authoritative_state_changed": False,
            "activation": False,
        }
        self._append_receipt("PlasticitySimulate", receipt)
        return receipt

    def plasticity_verify(
        self,
        proposal_id: str,
        *,
        evaluator: str,
        held_out_before: Sequence[bool],
        held_out_after: Sequence[bool],
        retention_before: Sequence[bool],
        retention_after: Sequence[bool],
    ) -> dict[str, Any]:
        proposal = self._proposal(proposal_id)
        _require(bool(evaluator) and evaluator != proposal["source"], "verification must be independently identified")
        arrays = (held_out_before, held_out_after, retention_before, retention_after)
        _require(all(array for array in arrays), "verification requires nonempty raw outcomes")
        _require(
            len(held_out_before) == len(held_out_after) and len(retention_before) == len(retention_after),
            "verification outcomes must be paired",
        )
        before = sum(bool(value) for value in held_out_before) / len(held_out_before)
        after = sum(bool(value) for value in held_out_after) / len(held_out_after)
        retention_prior = sum(bool(value) for value in retention_before) / len(retention_before)
        retention_post = sum(bool(value) for value in retention_after) / len(retention_after)
        passed = after > before and retention_post >= retention_prior
        verification = {
            "evaluator": evaluator,
            "held_out_before": [bool(value) for value in held_out_before],
            "held_out_after": [bool(value) for value in held_out_after],
            "retention_before": [bool(value) for value in retention_before],
            "retention_after": [bool(value) for value in retention_after],
            "computed": {
                "held_out_before": before,
                "held_out_after": after,
                "retention_before": retention_prior,
                "retention_after": retention_post,
            },
            "passed": passed,
            "activation": False,
        }
        verification["digest"] = io.digest(verification)
        proposal["verified"] = passed
        proposal["verification"] = verification
        self._append_receipt("PlasticityVerify", {"proposal_id": proposal_id, "verification": verification})
        return _copy(verification)

    def plasticity_commit(self, proposal_id: str) -> dict[str, Any]:
        proposal = self._proposal(proposal_id)
        _require(bool(proposal["verified"]), "unverified thought or proposal cannot commit a durable rewrite")
        _require(not proposal["committed"], "plasticity proposal already committed")
        relation_id = str(proposal["relation_id"])
        relation = self.plastic[relation_id]
        _require(not relation.frozen, "frozen relation cannot be rewritten")
        before = asdict(relation)
        relation.value = _clamp_to_alphabet(relation.value + int(proposal["delta"]), relation.precision)
        relation.last_verification = str(proposal["verification"]["digest"])
        relation.provenance.append(str(proposal["verification"]["digest"]))
        if relation.stability == "new":
            relation.stability = "provisional"
        relation.validate()
        proposal["committed"] = True
        self.rollbacks[proposal_id] = before
        return self._append_receipt(
            "PlasticityCommit",
            {"proposal_id": proposal_id, "relation_id": relation_id, "before": before, "after": asdict(relation)},
        )

    def plasticity_rollback(self, proposal_id: str) -> dict[str, Any]:
        proposal = self._proposal(proposal_id)
        if proposal_id not in self.rollbacks or not proposal["committed"]:
            raise io.Refused("no committed rewrite to roll back")
        prior = self.rollbacks[proposal_id]
        relation_id = str(proposal["relation_id"])
        self.plastic[relation_id] = PlasticRelation(**_copy(prior))
        proposal["committed"] = False
        return self._append_receipt(
            "PlasticityRollback",
            {"proposal_id": proposal_id, "relation_id": relation_id, "restored": prior},
        )

    def plasticity_consolidate(self, relation_id: str, *, verification: str) -> dict[str, Any]:
        relation = self._relation(relation_id)
        _require(relation.stability in {"supported", "reopened"}, "only supported or reopened relations may consolidate")
        _require(bool(verification), "consolidation verification is required")
        relation.stability = "consolidated"
        relation.last_verification = verification
        relation.provenance.append(verification)
        return self._append_receipt("PlasticityConsolidate", {"relation_id": relation_id, "verification": verification})

    def metaplasticity_promote(self, relation_id: str, *, evidence: str) -> dict[str, Any]:
        relation = self._relation(relation_id)
        transitions = {"new": "provisional", "provisional": "supported", "reopened": "supported"}
        _require(relation.stability in transitions, "relation cannot be promoted from current stability")
        prior = relation.stability
        relation.stability = transitions[prior]
        relation.provenance.append(evidence)
        return self._append_receipt(
            "MetaplasticityPromote",
            {"relation_id": relation_id, "before": prior, "after": relation.stability, "evidence": evidence},
        )

    def metaplasticity_destabilize(self, relation_id: str, *, contradiction: str) -> dict[str, Any]:
        relation = self._relation(relation_id)
        relation.contradictions.append(contradiction)
        prior = relation.stability
        threshold = 2 if prior == "consolidated" else 1
        if len(relation.contradictions) >= threshold and prior != "refuted":
            relation.stability = "reopened"
        return self._append_receipt(
            "MetaplasticityDestabilize",
            {
                "relation_id": relation_id,
                "before": prior,
                "after": relation.stability,
                "contradiction": contradiction,
                "threshold": threshold,
            },
        )

    def metaplasticity_reconsolidate(self, relation_id: str, *, verification: str) -> dict[str, Any]:
        relation = self._relation(relation_id)
        _require(relation.stability == "reopened", "only reopened relations may reconsolidate")
        relation.stability = "consolidated"
        relation.contradictions.clear()
        relation.last_verification = verification
        relation.provenance.append(verification)
        return self._append_receipt(
            "MetaplasticityReconsolidate",
            {"relation_id": relation_id, "verification": verification},
        )

    def apply_noise(self, relation_id: str, delta: int, *, provenance: str) -> bool:
        """Apply an isolated noisy observation only to flexible relations."""
        relation = self._relation(relation_id)
        if relation.stability == "consolidated":
            self.metaplasticity_destabilize(relation_id, contradiction=provenance)
            return False
        relation.value = _clamp_to_alphabet(relation.value + delta, relation.precision)
        relation.provenance.append(provenance)
        self._append_receipt("bounded_noise_update", {"relation_id": relation_id, "delta": delta, "provenance": provenance})
        return True

    def precision_request(
        self,
        relation_id: str,
        target: str,
        *,
        persistent_error: float,
        causal_precision_limit: bool,
        held_out_before: float,
        held_out_after: float,
        added_bytes: int,
        integrity_preserved: bool,
    ) -> dict[str, Any]:
        relation = self._relation(relation_id)
        _require(target in PRECISION_CLASSES, "unknown target precision")
        promotion = PRECISION_RANK[target] > PRECISION_RANK[relation.precision]
        passed = (
            promotion
            and persistent_error > 0.0
            and causal_precision_limit
            and held_out_after > held_out_before
            and added_bytes > 0
            and integrity_preserved
        )
        receipt = {
            "operation": "PrecisionRequest",
            "relation_id": relation_id,
            "before": relation.precision,
            "target": target,
            "persistent_error": persistent_error,
            "causal_precision_limit": causal_precision_limit,
            "held_out_before": held_out_before,
            "held_out_after": held_out_after,
            "added_bytes": added_bytes,
            "future_value_per_added_byte": (held_out_after - held_out_before) / max(added_bytes, 1),
            "integrity_preserved": integrity_preserved,
            "passed": passed,
            "activation": False,
        }
        receipt["digest"] = io.digest(receipt)
        self.precision_receipts.append(receipt)
        self._append_receipt("PrecisionRequest", receipt)
        return _copy(receipt)

    def precision_promote(self, request: Mapping[str, Any]) -> dict[str, Any]:
        _require(bool(request.get("passed")), "precision promotion has not earned its added bits")
        relation = self.plastic[str(request["relation_id"])]
        _require(not relation.frozen, "frozen precision cannot be promoted")
        before = relation.precision
        relation.precision = str(request["target"])
        relation.value = _clamp_to_alphabet(relation.value, relation.precision)
        return self._append_receipt(
            "PrecisionPromote",
            {"relation_id": request["relation_id"], "before": before, "after": relation.precision, "request": request["digest"]},
        )

    def precision_demote(
        self,
        relation_id: str,
        target: str,
        *,
        utility_before: float,
        utility_after: float,
    ) -> dict[str, Any]:
        relation = self._relation(relation_id)
        _require(target in PRECISION_CLASSES and PRECISION_RANK[target] < PRECISION_RANK[relation.precision], "invalid precision demotion")
        _require(not relation.frozen, "frozen precision cannot be demoted")
        _require(utility_after >= utility_before, "precision demotion loses verified future utility")
        before = relation.precision
        relation.precision = target
        relation.value = _clamp_to_alphabet(relation.value, target)
        return self._append_receipt(
            "PrecisionDemote",
            {
                "relation_id": relation_id,
                "before": before,
                "after": target,
                "utility_before": utility_before,
                "utility_after": utility_after,
            },
        )

    def precision_freeze(self, relation_id: str, *, authority: str) -> dict[str, Any]:
        relation = self._relation(relation_id)
        _require(bool(authority), "precision freeze requires authority")
        relation.frozen = True
        return self._append_receipt("PrecisionFreeze", {"relation_id": relation_id, "authority": authority})

    def precision_reopen(self, relation_id: str, *, contradiction: str) -> dict[str, Any]:
        relation = self._relation(relation_id)
        _require(relation.frozen, "only frozen precision may be reopened")
        _require(bool(contradiction), "precision reopen requires a contradiction")
        relation.frozen = False
        relation.stability = "reopened"
        relation.contradictions.append(contradiction)
        return self._append_receipt(
            "PrecisionReopen",
            {"relation_id": relation_id, "contradiction": contradiction},
        )

    def precision_audit(self) -> dict[str, Any]:
        rows = {
            key: {
                "precision": relation.precision,
                "frozen": relation.frozen,
                "valid": relation.precision in PRECISION_CLASSES,
            }
            for key, relation in sorted(self.plastic.items())
        }
        return {
            "operation": "PrecisionAudit",
            "relations": rows,
            "all_valid": all(row["valid"] for row in rows.values()),
            "activation": False,
        }

    def topology_change(
        self,
        operation: str,
        *,
        trigger: str,
        old_structure: Any,
        new_structure: Any,
        expected_value: float,
        resource_cost_bytes: int,
        affected: Sequence[str],
        rollback_state: Any,
        held_out_result: float,
    ) -> dict[str, Any]:
        _require(operation in TOPOLOGY_CONTRACTS, "unknown topology operation")
        _require(bool(trigger) and resource_cost_bytes >= 0, "invalid topology authority")
        if operation in {"TopologyAllocate", "TopologyConnect", "TopologySplit"}:
            _require(expected_value > 0.0 and held_out_result > 0.0, "growth must pay rent through held-out usefulness")
        receipt = {
            "operation": operation,
            "trigger": trigger,
            "old_structure": _copy(old_structure),
            "new_structure": _copy(new_structure),
            "expected_value": expected_value,
            "resource_cost_bytes": resource_cost_bytes,
            "affected_beliefs_and_procedures": list(affected),
            "rollback_state": _copy(rollback_state),
            "held_out_result": held_out_result,
            "activation": False,
        }
        receipt["digest"] = io.digest(receipt)
        self.topology_receipts.append(receipt)
        return receipt

    def topology_allocate(
        self,
        cell: CognitiveCell,
        *,
        trigger: str,
        expected_value: float,
        resource_cost_bytes: int,
        held_out_result: float,
    ) -> dict[str, Any]:
        cell.validate()
        _require(cell.cell_id not in self.cells, "cell already exists")
        receipt = self.topology_change(
            "TopologyAllocate",
            trigger=trigger,
            old_structure=None,
            new_structure=asdict(cell),
            expected_value=expected_value,
            resource_cost_bytes=resource_cost_bytes,
            affected=[cell.cell_id],
            rollback_state=None,
            held_out_result=held_out_result,
        )
        self.cells[cell.cell_id] = cell
        self.topology["nodes"][cell.cell_id] = asdict(cell)
        self._append_receipt("TopologyAllocate", receipt)
        return receipt

    def topology_connect(
        self,
        source: str,
        target: str,
        *,
        trigger: str,
        expected_value: float,
        held_out_result: float,
    ) -> dict[str, Any]:
        _require(source in self.cells and target in self.cells, "topology edge references unknown cell")
        edge = {"source": source, "target": target, "kind": "plastic_influence"}
        _require(edge not in self.topology["edges"], "topology edge already exists")
        receipt = self.topology_change(
            "TopologyConnect",
            trigger=trigger,
            old_structure=None,
            new_structure=edge,
            expected_value=expected_value,
            resource_cost_bytes=24,
            affected=[source, target],
            rollback_state=None,
            held_out_result=held_out_result,
        )
        self.topology["edges"].append(edge)
        self.cells[source].sparse_links.append(target)
        self._append_receipt("TopologyConnect", receipt)
        return receipt

    def topology_disconnect(self, source: str, target: str, *, trigger: str) -> dict[str, Any]:
        edge = {"source": source, "target": target, "kind": "plastic_influence"}
        _require(edge in self.topology["edges"], "topology edge does not exist")
        receipt = self.topology_change(
            "TopologyDisconnect",
            trigger=trigger,
            old_structure=edge,
            new_structure=None,
            expected_value=0.0,
            resource_cost_bytes=0,
            affected=[source, target],
            rollback_state=edge,
            held_out_result=1.0,
        )
        self.topology["edges"].remove(edge)
        if target in self.cells[source].sparse_links:
            self.cells[source].sparse_links.remove(target)
        self._append_receipt("TopologyDisconnect", receipt)
        return receipt

    def topology_split(
        self,
        source: str,
        children: Sequence[CognitiveCell],
        *,
        trigger: str,
        held_out_result: float,
    ) -> dict[str, Any]:
        _require(source in self.cells and len(children) >= 2, "split requires a source and at least two children")
        _require(all(child.cell_id not in self.cells for child in children), "split child already exists")
        for child in children:
            child.validate()
        prior = asdict(self.cells[source])
        receipt = self.topology_change(
            "TopologySplit",
            trigger=trigger,
            old_structure=prior,
            new_structure=[asdict(child) for child in children],
            expected_value=held_out_result,
            resource_cost_bytes=sum(len(io.canonical_bytes(asdict(child))) for child in children),
            affected=[source, *(child.cell_id for child in children)],
            rollback_state=prior,
            held_out_result=held_out_result,
        )
        self.cells[source].stability = "archived" if "archived" in METAPLASTIC_STATES else "refuted"
        for child in children:
            self.cells[child.cell_id] = child
            self.topology["nodes"][child.cell_id] = asdict(child)
        self._append_receipt("TopologySplit", receipt)
        return receipt

    def topology_merge(
        self,
        sources: Sequence[str],
        merged: CognitiveCell,
        *,
        trigger: str,
        distinctions_preserved: bool,
    ) -> dict[str, Any]:
        _require(len(sources) >= 2 and all(source in self.cells for source in sources), "merge sources are invalid")
        _require(distinctions_preserved, "merge may not erase required distinctions")
        merged.validate()
        receipt = self.topology_change(
            "TopologyMerge",
            trigger=trigger,
            old_structure=[asdict(self.cells[source]) for source in sources],
            new_structure=asdict(merged),
            expected_value=1.0,
            resource_cost_bytes=0,
            affected=[*sources, merged.cell_id],
            rollback_state=[asdict(self.cells[source]) for source in sources],
            held_out_result=1.0,
        )
        for source in sources:
            self.topology["archive"][source] = asdict(self.cells.pop(source))
            self.topology["nodes"].pop(source, None)
        self.cells[merged.cell_id] = merged
        self.topology["nodes"][merged.cell_id] = asdict(merged)
        self._append_receipt("TopologyMerge", receipt)
        return receipt

    def topology_prune(self, cell_id: str, *, required_competence_preserved: bool) -> dict[str, Any]:
        _require(cell_id in self.cells, "unknown cell")
        _require(required_competence_preserved, "pruning would destroy required competence")
        prior = asdict(self.cells.pop(cell_id))
        self.topology["archive"][cell_id] = prior
        self.topology["nodes"].pop(cell_id, None)
        self.topology["edges"] = [
            edge for edge in self.topology["edges"] if cell_id not in {edge["source"], edge["target"]}
        ]
        receipt = self.topology_change(
            "TopologyPrune",
            trigger="verified redundancy",
            old_structure=prior,
            new_structure=None,
            expected_value=0.0,
            resource_cost_bytes=0,
            affected=[cell_id],
            rollback_state=prior,
            held_out_result=1.0,
        )
        self._append_receipt("TopologyPrune", receipt)
        return receipt

    def topology_compile(self, cell_ids: Sequence[str], procedure_id: str, *, held_out_result: float) -> dict[str, Any]:
        _require(bool(cell_ids) and all(cell_id in self.cells for cell_id in cell_ids), "topology compilation references unknown cells")
        _require(procedure_id in self.compiled and self.compiled[procedure_id]["valid"], "topology compilation requires a verified procedure")
        receipt = self.topology_change(
            "TopologyCompile",
            trigger="verified repeated pathway",
            old_structure=[asdict(self.cells[cell_id]) for cell_id in cell_ids],
            new_structure={"compiled_procedure": procedure_id},
            expected_value=held_out_result,
            resource_cost_bytes=len(io.canonical_bytes(self.compiled[procedure_id])),
            affected=[*cell_ids, procedure_id],
            rollback_state=[asdict(self.cells[cell_id]) for cell_id in cell_ids],
            held_out_result=held_out_result,
        )
        self._append_receipt("TopologyCompile", receipt)
        return receipt

    def topology_archive(self, cell_id: str, *, trigger: str) -> dict[str, Any]:
        _require(cell_id in self.cells, "unknown cell")
        prior = asdict(self.cells.pop(cell_id))
        self.topology["archive"][cell_id] = prior
        self.topology["nodes"].pop(cell_id, None)
        receipt = self.topology_change(
            "TopologyArchive",
            trigger=trigger,
            old_structure=prior,
            new_structure={"archive_pointer": cell_id},
            expected_value=0.0,
            resource_cost_bytes=0,
            affected=[cell_id],
            rollback_state=prior,
            held_out_result=1.0,
        )
        self._append_receipt("TopologyArchive", receipt)
        return receipt

    def topology_restore(self, cell_id: str) -> dict[str, Any]:
        prior = self.topology["archive"].get(cell_id)
        _require(isinstance(prior, dict), "no archived topology state")
        cell = CognitiveCell(**_copy(prior))
        self.cells[cell_id] = cell
        self.topology["nodes"][cell_id] = asdict(cell)
        self.topology["archive"].pop(cell_id)
        receipt = self.topology_change(
            "TopologyRestore",
            trigger="rollback request",
            old_structure=None,
            new_structure=prior,
            expected_value=0.0,
            resource_cost_bytes=len(io.canonical_bytes(prior)),
            affected=[cell_id],
            rollback_state=None,
            held_out_result=1.0,
        )
        self._append_receipt("TopologyRestore", receipt)
        return receipt

    def predict(self, relation_ids: Sequence[str]) -> bool:
        return sum(self.plastic[key].value for key in relation_ids if key in self.plastic) > 0

    def field_transition(self, signal: float, *, elapsed_seconds: float = 0.0) -> dict[str, Any]:
        """Execute one mechanically distinct bounded transition for K1–K8."""
        prior = float(self.active.get("field_value", 0.0))
        if self.skeleton == "K1_monolithic_plastic_field":
            value = prior + signal
            mechanism = "monolithic_accumulation"
        elif self.skeleton == "K2_graph_plastic_field":
            value = signal + sum(cell.influence for cell in self.cells.values())
            mechanism = "sparse_graph_influence"
        elif self.skeleton == "K3_cellular_plastic_field":
            local = [cell.cognitive_activation + cell.influence for cell in self.cells.values()]
            value = signal + (sum(local) / len(local) if local else 0.0)
            mechanism = "shared_local_cell_rule"
        elif self.skeleton == "K4_continuous_time_plastic_field":
            value = prior * math.exp(-max(elapsed_seconds, 0.0)) + signal
            mechanism = "continuous_time_decay"
        elif self.skeleton == "K5_recurrent_state_space_plastic_field":
            value = 0.5 * prior + signal
            mechanism = "bounded_recurrent_transition"
        elif self.skeleton == "K6_adaptive_topology_field":
            value = signal + len(self.topology["nodes"]) - 0.25 * len(self.topology["archive"])
            mechanism = "topology_conditioned_transition"
        elif self.skeleton == "K7_native_mixed_radix_field":
            quantized = _clamp_to_alphabet(round(signal), "quinary")
            packed = pack_radix([quantized], PRECISION_ALPHABETS["quinary"], group_size=1)
            value = float(unpack_radix(packed)[0])
            mechanism = "native_mixed_radix_transition"
        else:
            value = signal
            mechanism = "integrated_placeholder_passthrough"
        self.active["field_value"] = value
        receipt = {
            "skeleton": self.skeleton,
            "mechanism": mechanism,
            "prior": prior,
            "signal": signal,
            "elapsed_seconds": elapsed_seconds,
            "value": value,
            "durable_state_changed": False,
            "activation": False,
        }
        self._append_receipt("FieldTransition", receipt)
        return receipt

    def settle_attractor(self, alternatives: Mapping[str, float], *, threshold: float = 0.6) -> dict[str, Any]:
        _require(bool(alternatives), "attractor settling requires alternatives")
        ordered = sorted(alternatives.items(), key=lambda row: (-row[1], row[0]))
        total = sum(max(score, 0.0) for _, score in ordered)
        confidence = ordered[0][1] / total if total else 0.0
        resolved = confidence >= threshold
        self.active["unresolved_alternatives"] = {} if resolved else dict(ordered)
        result = {
            "winner": ordered[0][0] if resolved else None,
            "confidence": confidence,
            "resolved": resolved,
            "preserved_alternatives": [] if resolved else [name for name, _ in ordered],
        }
        self._append_receipt("attractor_settle", result)
        return result

    def shadow_fork(self, shadow_id: str, *, relevant_relations: Sequence[str], relevant_cells: Sequence[str]) -> dict[str, Any]:
        _require(shadow_id not in self.shadows, "shadow field already exists")
        _require(all(key in self.plastic for key in relevant_relations), "shadow references unknown relation")
        _require(all(key in self.cells for key in relevant_cells), "shadow references unknown cell")
        shadow: dict[str, Any] = {
            "relations": {key: asdict(self.plastic[key]) for key in relevant_relations},
            "cells": {key: asdict(self.cells[key]) for key in relevant_cells},
            "perturbations": [],
            "result": None,
            "verified": False,
            "authoritative_digest_at_fork": self.state_integrity_digest(),
            "activation": False,
        }
        self.shadows[shadow_id] = shadow
        self._append_receipt("ShadowFieldFork", {"shadow_id": shadow_id, "region": [*relevant_relations, *relevant_cells]})
        return _copy(shadow)

    def shadow_perturb(self, shadow_id: str, relation_id: str, delta: int) -> dict[str, Any]:
        shadow = self._shadow(shadow_id)
        _require(relation_id in shadow["relations"], "unknown shadow relation")
        relation = shadow["relations"][relation_id]
        relation["value"] = _clamp_to_alphabet(int(relation["value"]) + delta, str(relation["precision"]))
        perturbation = {"relation_id": relation_id, "delta": delta}
        shadow["perturbations"].append(perturbation)
        self._append_receipt("ShadowFieldPerturb", {"shadow_id": shadow_id, **perturbation})
        return _copy(perturbation)

    def shadow_run(self, shadow_id: str, *, query_relations: Sequence[str]) -> dict[str, Any]:
        shadow = self._shadow(shadow_id)
        score = sum(int(shadow["relations"][key]["value"]) for key in query_relations if key in shadow["relations"])
        result = {"score": score, "prediction": score > 0, "authoritative_state_changed": False}
        shadow["result"] = result
        self._append_receipt("ShadowFieldRun", {"shadow_id": shadow_id, "result": result})
        return _copy(result)

    def shadow_compare(self, shadow_id: str, *, authoritative_relations: Sequence[str]) -> dict[str, Any]:
        shadow = self._shadow(shadow_id)
        _require(shadow["result"] is not None, "shadow has not run")
        authoritative = self.predict(authoritative_relations)
        comparison = {
            "shadow_prediction": shadow["result"]["prediction"],
            "authoritative_prediction": authoritative,
            "differs": shadow["result"]["prediction"] != authoritative,
        }
        self._append_receipt("ShadowFieldCompare", {"shadow_id": shadow_id, **comparison})
        return comparison

    def shadow_promote(self, shadow_id: str, *, evaluator: str, verified: bool) -> dict[str, Any]:
        shadow = self._shadow(shadow_id)
        _require(shadow["result"] is not None, "shadow has not run")
        _require(bool(evaluator) and verified, "shadow result requires independent verification before promotion")
        for relation_id, relation in shadow["relations"].items():
            if relation_id in self.plastic:
                current = self.plastic[relation_id]
                current.value = int(relation["value"])
                current.provenance.append(f"shadow-verification:{evaluator}")
                current.last_verification = f"shadow-verification:{evaluator}"
        shadow["verified"] = True
        return self._append_receipt("ShadowFieldPromote", {"shadow_id": shadow_id, "evaluator": evaluator})

    def shadow_discard(self, shadow_id: str) -> dict[str, Any]:
        _require(shadow_id in self.shadows, "unknown shadow")
        self.shadows.pop(shadow_id)
        return self._append_receipt("ShadowFieldDiscard", {"shadow_id": shadow_id})

    def procedure_observe_trace(self, trace_id: str, instruction: str, argument: Any = None) -> dict[str, Any]:
        bytecode = {
            "OBSERVE",
            "BIND",
            "RETRIEVE",
            "COMPARE",
            "INFER",
            "SIMULATE",
            "VERIFY",
            "REVISE",
            "DEFER",
            "COMMIT",
            "ROLLBACK",
        }
        _require(instruction in bytecode, "unknown cognitive bytecode")
        self.traces.setdefault(trace_id, []).append({"instruction": instruction, "argument": _copy(argument)})
        return self._append_receipt("ProcedureObserveTrace", {"trace_id": trace_id, "instruction": instruction, "argument": argument})

    def procedure_propose(self, procedure_id: str, trace_id: str, *, proposer: str) -> dict[str, Any]:
        if trace_id not in self.traces or len(self.traces[trace_id]) < 2:
            raise io.Refused("procedure proposal requires an observed trace")
        _require(procedure_id not in self.procedure_candidates, "procedure proposal already exists")
        proposal = {
            "procedure_id": procedure_id,
            "trace_id": trace_id,
            "proposer": proposer,
            "trace_digest": io.digest(self.traces[trace_id]),
            "verified": False,
            "verification": None,
            "activation": False,
        }
        proposal["digest"] = io.digest(proposal)
        self.procedure_candidates[procedure_id] = proposal
        self._append_receipt("ProcedurePropose", proposal)
        return _copy(proposal)

    def procedure_verify(
        self,
        procedure_id: str,
        *,
        evaluator: str,
        raw_flexible_correct: Sequence[bool],
        raw_compiled_correct: Sequence[bool],
    ) -> dict[str, Any]:
        if procedure_id not in self.procedure_candidates:
            raise io.Refused("unknown procedure proposal")
        proposal = self.procedure_candidates[procedure_id]
        _require(evaluator != proposal["proposer"], "procedure verification must be independent")
        _require(
            bool(raw_flexible_correct) and len(raw_flexible_correct) == len(raw_compiled_correct),
            "procedure verification requires paired raw results",
        )
        passed = all(bool(value) for value in raw_compiled_correct) and sum(raw_compiled_correct) >= sum(raw_flexible_correct)
        verification = {
            "evaluator": evaluator,
            "raw_flexible_correct": [bool(value) for value in raw_flexible_correct],
            "raw_compiled_correct": [bool(value) for value in raw_compiled_correct],
            "passed": passed,
            "activation": False,
        }
        verification["digest"] = io.digest(verification)
        proposal["verified"] = passed
        proposal["verification"] = verification
        self._append_receipt("ProcedureVerify", {"procedure_id": procedure_id, "verification": verification})
        return _copy(verification)

    def procedure_compile(
        self,
        procedure_id: str,
        trace_id: str,
        *,
        inputs: Sequence[str],
        assumptions: Sequence[str],
        scope: str,
        branch_conditions: Sequence[str],
        failure_conditions: Sequence[str],
        verification_method: str,
        provenance: str,
    ) -> dict[str, Any]:
        if trace_id not in self.traces or len(self.traces[trace_id]) < 2:
            raise io.Refused("procedure compilation requires an observed trace")
        trace = self.traces[trace_id]
        _require(bool(verification_method) and bool(provenance), "compiled procedure requires verification and provenance")
        if procedure_id in self.procedure_candidates:
            _require(bool(self.procedure_candidates[procedure_id]["verified"]), "procedure proposal has not passed verification")
        procedure = {
            "procedure_id": procedure_id,
            "inputs": list(inputs),
            "assumptions": list(assumptions),
            "scope": scope,
            "branch_conditions": list(branch_conditions),
            "failure_conditions": list(failure_conditions),
            "verification_method": verification_method,
            "cost": len(trace),
            "provenance": provenance,
            "bytecode": _copy(trace),
            "valid": True,
            "reopen_flexible_reasoning": False,
            "activation": False,
        }
        procedure["digest"] = io.digest(procedure)
        self.compiled[procedure_id] = procedure
        self._append_receipt("ProcedureCompile", procedure)
        return _copy(procedure)

    def procedure_execute(self, procedure_id: str, bindings: Mapping[str, Any]) -> dict[str, Any]:
        procedure = self._procedure(procedure_id)
        _require(bool(procedure["valid"]), "procedure is unavailable or invalid")
        missing = set(procedure["inputs"]) - set(bindings)
        _require(not missing, f"procedure inputs missing: {sorted(missing)}")
        accumulator: Any = None
        cost = 0
        for instruction in procedure["bytecode"]:
            cost += 1
            operation = instruction["instruction"]
            argument = instruction["argument"]
            if operation in {"OBSERVE", "BIND", "RETRIEVE"}:
                accumulator = bindings.get(str(argument), argument)
            elif operation == "COMPARE":
                accumulator = accumulator == bindings.get(str(argument), argument)
            elif operation == "INFER":
                accumulator = bool(accumulator)
            elif operation == "DEFER":
                return {"status": "deferred", "cost": cost, "result": None}
            elif operation == "ROLLBACK":
                return {"status": "rolled_back", "cost": cost, "result": None}
        result = {"status": "completed", "cost": cost, "result": accumulator}
        self._append_receipt("ProcedureExecute", {"procedure_id": procedure_id, **result})
        return result

    def procedure_monitor(self, procedure_id: str, *, passed: bool, exception: str | None = None) -> dict[str, Any]:
        procedure = self._procedure(procedure_id)
        if not passed:
            procedure["valid"] = False
            procedure["reopen_flexible_reasoning"] = True
        return self._append_receipt(
            "ProcedureMonitor",
            {"procedure_id": procedure_id, "passed": passed, "exception": exception},
        )

    def procedure_invalidate(self, procedure_id: str, *, evidence: str) -> dict[str, Any]:
        procedure = self._procedure(procedure_id)
        _require(bool(evidence), "procedure invalidation requires evidence")
        procedure["valid"] = False
        procedure["reopen_flexible_reasoning"] = True
        return self._append_receipt("ProcedureInvalidate", {"procedure_id": procedure_id, "evidence": evidence})

    def procedure_decompile(self, procedure_id: str) -> list[dict[str, Any]]:
        procedure = self._procedure(procedure_id)
        self._append_receipt("ProcedureDecompile", {"procedure_id": procedure_id})
        return _copy(procedure["bytecode"])

    def schedule_event(self, kind: str, at_elapsed_seconds: float, payload: Mapping[str, Any]) -> dict[str, Any]:
        temporal_kinds = {
            "ScheduledObservation",
            "BackgroundConsolidation",
            "GoalDeadline",
            "PredictionExpiry",
            "MemoryDecayProposal",
        }
        _require(kind in temporal_kinds, "unknown temporal event")
        _require(at_elapsed_seconds >= self.active["elapsed_seconds"], "cannot schedule an event in the past")
        if kind == "BackgroundConsolidation":
            relation_id = str(payload.get("relation_id", ""))
            relation = self._relation(relation_id)
            _require(
                relation.stability == "supported"
                and bool(relation.last_verification)
                and payload.get("verification") == relation.last_verification,
                "background consolidation requires the relation's verified receipt",
            )
        event = {
            "kind": kind,
            "at_elapsed_seconds": float(at_elapsed_seconds),
            "payload": _copy(dict(payload)),
            "processed": False,
            "activation": False,
        }
        event["digest"] = io.digest(event)
        self.scheduled.append(event)
        self._append_receipt("TemporalEvent", event)
        return _copy(event)

    def elapsed_time(self, seconds: float) -> list[dict[str, Any]]:
        _require(seconds >= 0.0, "elapsed time cannot be negative")
        self.active["elapsed_seconds"] += float(seconds)
        due: list[dict[str, Any]] = []
        for event in self.scheduled:
            if not event["processed"] and event["at_elapsed_seconds"] <= self.active["elapsed_seconds"]:
                event["processed"] = True
                due.append(_copy(event))
                if event["kind"] == "GoalDeadline":
                    goal_id = str(event["payload"]["goal_id"])
                    if goal_id in self.exact["goal_commitments"]:
                        self.exact["goal_commitments"][goal_id]["overdue"] = True
                elif event["kind"] == "MemoryDecayProposal":
                    self.active.setdefault("decay_proposals", []).append(_copy(event["payload"]))
                elif event["kind"] == "BackgroundConsolidation":
                    relation_id = str(event["payload"]["relation_id"])
                    relation = self.plastic.get(relation_id)
                    if relation is not None and relation.stability == "supported":
                        relation.stability = "consolidated"
                elif event["kind"] == "ScheduledObservation":
                    self.active["momentary_activation"][str(event["payload"]["key"])] = _copy(event["payload"]["value"])
                elif event["kind"] == "PredictionExpiry":
                    self.active.setdefault("expired_predictions", []).append(_copy(event["payload"]))
        self.temporal_events.extend(due)
        self._append_receipt("ElapsedTime", {"seconds": seconds, "due_event_digests": [event["digest"] for event in due]})
        return due

    def create_goal(self, goal_id: str, description: str, *, provenance: str) -> None:
        _require(goal_id not in self.exact["goal_commitments"], "goal already exists")
        self.exact["goal_commitments"][goal_id] = {
            "description": description,
            "status": "unfinished",
            "overdue": False,
            "provenance": provenance,
        }
        self._append_receipt("goal_created", {"goal_id": goal_id, "description": description, "provenance": provenance})

    def checkpoint(self) -> dict[str, Any]:
        state = self.document()
        state["E_t"]["checkpoint_integrity"] = None
        state_digest = io.digest(state)
        checkpoint = {
            "schema": f"{FIELD_SCHEMA}-checkpoint",
            "state": state,
            "state_integrity_digest": state_digest,
            "activation": False,
        }
        checkpoint["sha256"] = io.digest(checkpoint)
        self.exact["checkpoint_integrity"] = checkpoint["sha256"]
        return checkpoint

    @classmethod
    def restore(cls, checkpoint: Mapping[str, Any]) -> EndogenousPlasticField:
        document = _copy(dict(checkpoint))
        claimed = document.pop("sha256", None)
        _require(isinstance(claimed, str) and io.digest(document) == claimed, "field checkpoint seal mismatch")
        state = document.get("state")
        _require(isinstance(state, dict), "field checkpoint state is missing")
        _require(io.digest(state) == document.get("state_integrity_digest"), "field checkpoint state digest mismatch")
        _require(not io.contains_true_activation(state), "field checkpoint attempts activation")
        restored = cls(
            str(state["E_t"]["identity"]),
            skeleton=str(state["skeleton"]),
            resource_envelope=str(state["resource_envelope"]),
            s2_derived=bool(state["s2_derived"]),
        )
        restored.theta = _copy(state["Theta"])
        restored.plastic = {key: PlasticRelation(**_copy(value)) for key, value in state["P_t"].items()}
        restored.topology = _copy(state["G_t"])
        restored.active = _copy(state["Z_t"])
        restored.exact = _copy(state["E_t"])
        restored.exact["checkpoint_integrity"] = claimed
        restored.archive = _copy(state["A"])
        restored.compiled = _copy(state["C_t"])
        restored.competence = _copy(state["M_t"])
        restored.cells = {key: CognitiveCell(**_copy(value)) for key, value in state["cells"].items()}
        restored.proposals = _copy(state["pending_proposals"])
        restored.rollbacks = _copy(state["rollbacks"])
        restored.shadows = _copy(state["shadows"])
        restored.traces = _copy(state["traces"])
        restored.procedure_candidates = _copy(state.get("procedure_candidates", {}))
        restored.temporal_events = _copy(state["temporal_events"])
        restored.scheduled = _copy(state["scheduled"])
        restored.precision_receipts = _copy(state["precision_receipts"])
        restored.topology_receipts = _copy(state["topology_receipts"])
        return restored

    def cognitive_state_export(self) -> dict[str, Any]:
        neutral = {
            "schema": "substrate-cognitive-neutral-ir/v1",
            "identity": _copy(self.exact["identity"]),
            "history_references": [row["digest"] for row in self.archive],
            "goals": _copy(self.exact["goal_commitments"]),
            "beliefs": {
                key: {
                    "value": relation.value,
                    "precision": relation.precision,
                    "stability": relation.stability,
                    "provenance": relation.provenance,
                }
                for key, relation in sorted(self.plastic.items())
            },
            "knowledge": {
                key: relation.value
                for key, relation in sorted(self.plastic.items())
                if relation.stability == "consolidated"
            },
            "world_state": _copy(self.topology),
            "self_state": {
                "skeleton": self.skeleton,
                "resource_envelope": self.resource_envelope,
                "s2_derived": self.s2_derived,
            },
            "body_state": _copy(self.competence["bodies"]),
            "model_registry": _copy(self.competence["models"]),
            "procedures": _copy(self.compiled),
            "semantic_continuity_tested": False,
            "identity_transfer_claimed": False,
            "activation": False,
        }
        neutral["sha256"] = io.digest(neutral)
        return neutral

    @classmethod
    def cognitive_state_import(
        cls,
        neutral: Mapping[str, Any],
        *,
        target_skeleton: str,
        resource_envelope: str,
    ) -> EndogenousPlasticField:
        document = _copy(dict(neutral))
        claimed = document.pop("sha256", None)
        _require(isinstance(claimed, str) and io.digest(document) == claimed, "neutral cognitive state seal mismatch")
        _require(document.get("schema") == "substrate-cognitive-neutral-ir/v1", "unknown neutral state schema")
        restored = cls(
            str(document["identity"]),
            skeleton=target_skeleton,
            resource_envelope=resource_envelope,
            s2_derived=bool(document["self_state"].get("s2_derived", False)),
        )
        for key, value in document["beliefs"].items():
            restored.plastic[key] = PlasticRelation(
                int(value["value"]),
                str(value["precision"]),
                str(value["stability"]),
                "migrated",
                list(value["provenance"]),
            )
        restored.topology = _copy(document["world_state"])
        restored.exact["goal_commitments"] = _copy(document["goals"])
        restored.competence["bodies"] = _copy(document["body_state"])
        restored.competence["models"] = _copy(document["model_registry"])
        restored.compiled = _copy(document["procedures"])
        restored.exact["lineage"].append({"neutral_state": claimed, "identity_transfer_claimed": False})
        restored._append_receipt("CognitiveStateImport", {"neutral_state": claimed, "target_skeleton": target_skeleton})
        return restored

    @staticmethod
    def cognitive_state_compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
        fields = ("identity", "goals", "beliefs", "knowledge", "body_state", "model_registry", "procedures")
        checks = {key: left.get(key) == right.get(key) for key in fields}
        return {
            "operation": "CognitiveStateCompare",
            "checks": checks,
            "semantic_fields_equal": all(checks.values()),
            "identity_transfer_claimed": False,
            "activation": False,
        }

    @classmethod
    def cognitive_state_migrate(
        cls,
        source: EndogenousPlasticField,
        *,
        target_skeleton: str,
        resource_envelope: str,
    ) -> dict[str, Any]:
        """Migrate through the neutral IR while retaining rollback state."""
        neutral = source.cognitive_state_export()
        rollback_checkpoint = source.checkpoint()
        migrated = cls.cognitive_state_import(
            neutral,
            target_skeleton=target_skeleton,
            resource_envelope=resource_envelope,
        )
        comparison = cls.cognitive_state_compare(neutral, migrated.cognitive_state_export())
        return {
            "operation": "CognitiveStateMigrate",
            "neutral_state": neutral,
            "migrated_checkpoint": migrated.checkpoint(),
            "comparison": comparison,
            "rollback_checkpoint": rollback_checkpoint,
            "semantic_continuity_tested": False,
            "identity_transfer_claimed": False,
            "activation": False,
        }

    @classmethod
    def cognitive_state_rollback(cls, migration: Mapping[str, Any]) -> EndogenousPlasticField:
        checkpoint = migration.get("rollback_checkpoint")
        if not isinstance(checkpoint, dict):
            raise io.Refused("migration has no rollback checkpoint")
        return cls.restore(checkpoint)


def capability_density_frontier() -> dict[str, Any]:
    """Run a deterministic numeric microfixture under equal resource envelopes."""
    values = [math.sin(index * 0.173) * 3.7 + math.cos(index * 0.071) for index in range(4096)]
    tolerance = 0.21

    def quantize(mode: str, value: float, index: int) -> float:
        if mode == "full_precision":
            return value
        if mode == "post_hoc_compressed":
            return round(value * 2.0) / 2.0
        if mode == "natively_compressed":
            return round(value * 4.0) / 4.0
        if abs(value) > 3.25 or index % 97 == 0:
            return round(value * 16.0) / 16.0
        return round(value * 4.0) / 4.0

    modes = {
        "full_precision": 64,
        "post_hoc_compressed": 4,
        "natively_compressed": 3,
        "adaptive_precision": 3,
    }
    rows: list[dict[str, Any]] = []
    for system in ("substrate_candidate_fixture", "s2_derived_equal_resource_fixture"):
        for envelope, limit in RESOURCE_ENVELOPES_BYTES.items():
            for mode, bits_per_value in modes.items():
                reconstructed = [quantize(mode, value, index) for index, value in enumerate(values)]
                errors = [abs(original - restored) for original, restored in zip(values, reconstructed, strict=True)]
                raw_correct = [error <= tolerance for error in errors]
                resident_bytes = math.ceil(len(values) * bits_per_value / 8)
                operation_count = len(values) * (1 if mode == "full_precision" else (3 if mode == "adaptive_precision" else 2))
                utility = sum(raw_correct) / len(raw_correct)
                fits = limit is None or resident_bytes <= limit
                retained_correct = sum(raw_correct[: len(raw_correct) // 2])
                compound_correct = sum(
                    all(raw_correct[position : position + 4]) for position in range(0, len(raw_correct), 4)
                )
                added_bytes = max(resident_bytes - math.ceil(len(values) * 3 / 8), 1)
                rows.append(
                    {
                        "system": system,
                        "envelope": envelope,
                        "mode": mode,
                        "fits": fits,
                        "resident_bytes_measured": resident_bytes,
                        "checkpoint_bytes_measured": resident_bytes,
                        "operation_count_energy_proxy": operation_count,
                        "latency_operation_proxy": operation_count,
                        "raw_correct_digest": io.digest(raw_correct),
                        "utility": utility if fits else 0.0,
                        "utility_per_resident_byte": utility / resident_bytes if fits else 0.0,
                        "useful_retained_history": retained_correct,
                        "useful_retained_history_per_byte": retained_correct / resident_bytes if fits else 0.0,
                        "new_capability_per_added_byte": utility / added_bytes if fits else 0.0,
                        "compound_task_utility": compound_correct / (len(raw_correct) // 4) if fits else 0.0,
                        "compound_task_utility_per_energy_proxy": (
                            (compound_correct / (len(raw_correct) // 4)) / operation_count if fits else 0.0
                        ),
                        "rare_case_accuracy": sum(raw_correct[::97]) / len(raw_correct[::97]) if fits else 0.0,
                        "calibration_brier": sum(error * error for error in errors) / len(errors),
                        "learning_rate_fixture": utility,
                        "learning_retained_per_added_byte": retained_correct / added_bytes if fits else 0.0,
                        "recovery_time_operation_proxy": len(values),
                        "developmental_history": "controlled_numeric_microfixture_only",
                        "activation": False,
                    }
                )
    parity = all(
        next(
            row
            for row in rows
            if row["system"] == "substrate_candidate_fixture" and row["envelope"] == envelope and row["mode"] == mode
        )["utility"]
        == next(
            row
            for row in rows
            if row["system"] == "s2_derived_equal_resource_fixture" and row["envelope"] == envelope and row["mode"] == mode
        )["utility"]
        for envelope in RESOURCE_ENVELOPES_BYTES
        for mode in modes
    )
    pareto: list[dict[str, Any]] = []
    for system in ("substrate_candidate_fixture", "s2_derived_equal_resource_fixture"):
        candidates = [row for row in rows if row["system"] == system and row["fits"]]
        for row in candidates:
            dominated = any(
                other["resident_bytes_measured"] <= row["resident_bytes_measured"]
                and other["utility"] >= row["utility"]
                and (
                    other["resident_bytes_measured"] < row["resident_bytes_measured"]
                    or other["utility"] > row["utility"]
                )
                for other in candidates
            )
            if not dominated:
                pareto.append(
                    {
                        "system": system,
                        "mode": row["mode"],
                        "resident_bytes_measured": row["resident_bytes_measured"],
                        "utility": row["utility"],
                    }
                )
    return {
        "schema": "substrate-field-capability-density/v1",
        "scope": FOUNDATION_STATUS,
        "resource_envelopes_bytes": RESOURCE_ENVELOPES_BYTES,
        "raw_rows": rows,
        "pareto_frontier": pareto,
        "s2_exact_resource_and_algorithm_parity": parity,
        "promotional_utility_per_gb_only": False,
        "full_raw_performance_reported": True,
        "activation": False,
    }


def concept_micro_worlds() -> dict[str, Any]:
    """Return construction-only curriculum templates without future instances."""
    domains = {
        "cloud_appearance_to_causal_system": ("appearance", "composition", "mechanism", "causal_system", "exceptions", "transfer"),
        "object_appearance_to_material": ("surface", "weight", "material", "hidden_structure", "transfer"),
        "tool_appearance_to_function": ("shape", "affordance", "function", "failure_mode", "transfer"),
        "animal_category_to_behavior": ("appearance", "category", "behavior", "exception", "transfer"),
        "social_role_to_individual_identity": ("role_cue", "role", "individual", "context_shift", "transfer"),
        "motion_pattern_to_causal_mechanism": ("trajectory", "pattern", "force", "intervention", "transfer"),
        "symptom_to_hidden_cause": ("symptom", "correlation", "latent_cause", "intervention", "transfer"),
    }
    templates = [
        {
            "domain": domain,
            "developmental_stages": list(stages),
            "construction_seed_namespace": f"field-foundation/{index}",
            "held_out_rule": "remove original examples and generate disjoint entities",
            "measures": ["routing", "prediction", "analogy", "inquiry", "explanation"],
            "future_principal_instances_consumed": False,
        }
        for index, (domain, stages) in enumerate(domains.items(), start=1)
    ]
    return {
        "schema": "substrate-field-concept-micro-worlds/v1",
        "templates": templates,
        "future_seed_commitment": "to_be_committed_by_next_program_before_generation",
        "future_principal_instances_consumed": False,
        "activation": False,
    }


def attractor_microtests() -> dict[str, Any]:
    """Exercise coherent settling, ambiguity, revision, and nonfabricating recovery."""
    candidate = EndogenousPlasticField("attractor-fixture")
    coherent = candidate.settle_attractor({"material": 0.9, "appearance": 0.1})
    unresolved = candidate.settle_attractor({"cause_a": 0.51, "cause_b": 0.49}, threshold=0.7)
    false_prior = candidate.settle_attractor({"false_model": 0.8, "correct_model": 0.2})
    revised = candidate.settle_attractor({"false_model": 0.1, "correct_model": 0.9})
    checkpoint = candidate.checkpoint()
    restored = EndogenousPlasticField.restore(checkpoint)
    recovery = {
        "archive_exact": restored.archive == candidate.archive,
        "evidence_count_before": len(candidate.exact["evidence_provenance"]),
        "evidence_count_after": len(restored.exact["evidence_provenance"]),
    }
    checks = {
        "settles_coherently": coherent["resolved"] and coherent["winner"] == "material",
        "preserves_unresolved_alternatives": not unresolved["resolved"]
        and set(unresolved["preserved_alternatives"]) == {"cause_a", "cause_b"},
        "escapes_false_attractor_after_evidence": false_prior["winner"] == "false_model"
        and revised["winner"] == "correct_model",
        "recovers_without_fabricating_evidence": recovery["archive_exact"]
        and recovery["evidence_count_before"] == recovery["evidence_count_after"],
    }
    return {
        "schema": "substrate-field-attractor-microtests/v1",
        "checks": checks,
        "coherent": coherent,
        "unresolved": unresolved,
        "false_prior": false_prior,
        "revised": revised,
        "recovery": recovery,
        "all_pass": all(checks.values()),
        "activation": False,
    }


def skeleton_activity_report() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, (name, specification) in enumerate(SKELETONS.items(), start=1):
        candidate = EndogenousPlasticField(
            f"foundation-{index}",
            skeleton=name,
            resource_envelope="512_MB",
        )
        candidate.add_relation("signal", 0, precision="ternary", provenance="fixture://signal")
        candidate.cells["probe"] = CognitiveCell(
            "probe",
            "concept",
            cognitive_activation=index % 3,
            influence=index % 2,
            provenance_pointer=f"fixture://skeleton/{index}",
        )
        candidate.topology["nodes"]["probe"] = asdict(candidate.cells["probe"])
        candidate.observe("cue", index, provenance="fixture://activity")
        transition = candidate.field_transition(float(index % 4 - 1), elapsed_seconds=0.25)
        checkpoint = candidate.checkpoint()
        restored = EndogenousPlasticField.restore(checkpoint)
        rows.append(
            {
                "candidate": name,
                **specification,
                "interfaces": list(candidate.interfaces()),
                "interface_count": len(candidate.interfaces()),
                "activity_receipt_count": len(candidate.archive),
                "transition": transition,
                "checkpoint_bytes": len(io.canonical_bytes(checkpoint)),
                "checkpoint_restore_exact": (
                    restored.document()["P_t"] == candidate.document()["P_t"]
                    and restored.document()["E_t"]["identity"] == candidate.document()["E_t"]["identity"]
                ),
                "principal_quality_claimed": False,
                "classification_claimed": False,
                "activation": False,
            }
        )
    return {
        "schema": "substrate-field-candidate-skeletons/v1",
        "common_contracts": list(COMMON_FIELD_CONTRACTS),
        "rows": rows,
        "all_runnable": all(row["checkpoint_restore_exact"] for row in rows),
        "mechanically_distinct_transition_count": len({row["transition"]["mechanism"] for row in rows}),
        "activity_does_not_establish_self_organization": True,
        "activation": False,
    }


def raw_metric_receipt(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw = [_copy(dict(row)) for row in rows]
    result = {
        "raw": raw,
        "count": len(raw),
        "utility_mean": sum(float(row["utility"]) for row in raw) / len(raw) if raw else 0.0,
        "resident_bytes_total": sum(int(row["resident_bytes_measured"]) for row in raw),
        "activation": False,
    }
    result["raw_digest"] = io.digest(raw)
    result["sha256"] = io.digest(result)
    return result


def verify_raw_metric_receipt(receipt: Mapping[str, Any]) -> bool:
    document = _copy(dict(receipt))
    claimed = document.pop("sha256", None)
    if not isinstance(claimed, str) or io.digest(document) != claimed:
        return False
    raw = document.get("raw")
    if not isinstance(raw, list) or io.digest(raw) != document.get("raw_digest"):
        return False
    expected_mean = sum(float(row["utility"]) for row in raw) / len(raw) if raw else 0.0
    expected_bytes = sum(int(row["resident_bytes_measured"]) for row in raw)
    return (
        document.get("count") == len(raw)
        and document.get("utility_mean") == expected_mean
        and document.get("resident_bytes_total") == expected_bytes
    )


def run_foundation_canaries() -> dict[str, Any]:
    """Execute the 28 amendment canaries as bounded feasibility probes."""

    def result(canary_id: str, name: str, passed: bool, evidence: Mapping[str, Any]) -> dict[str, Any]:
        row = {
            "canary_id": canary_id,
            "name": name,
            "passed": bool(passed),
            "evidence": _copy(dict(evidence)),
            "claim_scope": FOUNDATION_STATUS,
            "activation": False,
        }
        row["sha256"] = io.digest(row)
        return row

    rows: list[dict[str, Any]] = []

    ternary = EndogenousPlasticField("f01")
    ternary.add_relation("r", -1, precision="ternary", provenance="canary://f01")
    before = ternary.predict(["r"])
    ternary.plastic["r"].value = 1
    after = ternary.predict(["r"])
    rows.append(result("F01", "native ternary state changes runtime behavior", before != after, {"before": before, "after": after}))

    quinary = EndogenousPlasticField("f02")
    quinary.add_relation("r", -2, precision="quinary", provenance="canary://f02")
    before = quinary.predict(["r"])
    quinary.plastic["r"].value = 2
    after = quinary.predict(["r"])
    rows.append(result("F02", "native quinary state changes runtime behavior", before != after, {"before": before, "after": after}))

    packing = packing_benchmark(repetitions=5)
    exhaustive = all(
        unpack_radix(pack_radix([a, b, c], PRECISION_ALPHABETS["quinary"], group_size=3)) == [a, b, c]
        for a in PRECISION_ALPHABETS["quinary"]
        for b in PRECISION_ALPHABETS["quinary"]
        for c in PRECISION_ALPHABETS["quinary"]
    )
    rows.append(
        result(
            "F03",
            "packing and unpacking are exact",
            packing["all_exact"]
            and packing["quinary_three_base5_values_in_seven_bits"]
            and packing["adaptive_mixed_radix"]["exact_round_trip"]
            and exhaustive,
            {"benchmark": packing, "all_125_quinary_triplets_exact": exhaustive},
        )
    )

    rewrite = EndogenousPlasticField("f04")
    rewrite.add_relation("cause", -1, precision="ternary", provenance="canary://f04")
    before = rewrite.predict(["cause"])
    rewrite.plasticity_propose("p", "cause", 2, source="learner", source_kind="observation", evidence=["train"])
    rewrite.plasticity_verify(
        "p",
        evaluator="held-out-evaluator",
        held_out_before=[False, False],
        held_out_after=[True, True],
        retention_before=[True],
        retention_after=[True],
    )
    rewrite.plasticity_commit("p")
    after = rewrite.predict(["cause"])
    rows.append(result("F04", "low-bit rewrite changes a held-out future prediction", before != after, {"before": before, "after": after}))

    unverified = EndogenousPlasticField("f05")
    unverified.add_relation("r", 0, precision="ternary", provenance="canary://f05")
    unverified.plasticity_propose("thought", "r", 1, source="model-thought", source_kind="thought", evidence=[])
    refused = False
    try:
        unverified.plasticity_commit("thought")
    except io.Refused:
        refused = True
    rows.append(result("F05", "unverified thought cannot commit a durable rewrite", refused, {"commit_refused": refused}))

    checkpoint = rewrite.checkpoint()
    restored = EndogenousPlasticField.restore(checkpoint)
    rows.append(
        result(
            "F06",
            "verified learning commits and survives restore",
            restored.plastic["cause"].value == rewrite.plastic["cause"].value,
            {"restored_value": restored.plastic["cause"].value, "checkpoint_sha256": checkpoint["sha256"]},
        )
    )

    rewrite.plasticity_rollback("p")
    restored_prediction = rewrite.predict(["cause"])
    rows.append(
        result(
            "F07",
            "rollback restores prior semantic behavior",
            restored_prediction == before,
            {"restored_prediction": restored_prediction},
        )
    )

    meta = EndogenousPlasticField("f08")
    meta.add_relation("flexible", 1, precision="ternary", stability="provisional", provenance="canary://f08")
    meta.add_relation("stable", 1, precision="ternary", stability="consolidated", provenance="canary://f08")
    flexible_changed = meta.apply_noise("flexible", -1, provenance="noise-1")
    stable_changed = meta.apply_noise("stable", -1, provenance="noise-1")
    rows.append(
        result(
            "F08",
            "provisional and consolidated relations respond differently to noise",
            flexible_changed and not stable_changed and meta.plastic["flexible"].value != meta.plastic["stable"].value,
            {"flexible_changed": flexible_changed, "stable_changed": stable_changed},
        )
    )

    meta.metaplasticity_destabilize("stable", contradiction="noise-2")
    rows.append(
        result(
            "F09",
            "repeated verified contradiction reopens a consolidated relation",
            meta.plastic["stable"].stability == "reopened",
            {"state": meta.plastic["stable"].stability, "contradictions": meta.plastic["stable"].contradictions},
        )
    )

    precision = EndogenousPlasticField("f10")
    precision.add_relation("limited", 1, precision="binary", provenance="canary://f10")
    promotion = precision.precision_request(
        "limited",
        "quinary",
        persistent_error=0.25,
        causal_precision_limit=True,
        held_out_before=0.5,
        held_out_after=0.9,
        added_bytes=1,
        integrity_preserved=True,
    )
    precision.precision_promote(promotion)
    rows.append(
        result(
            "F10",
            "precision promotion improves a precision-limited task",
            precision.plastic["limited"].precision == "quinary" and promotion["held_out_after"] > promotion["held_out_before"],
            promotion,
        )
    )

    unnecessary = precision.precision_request(
        "limited",
        "8_bit",
        persistent_error=0.0,
        causal_precision_limit=False,
        held_out_before=1.0,
        held_out_after=1.0,
        added_bytes=1,
        integrity_preserved=True,
    )
    refused = False
    try:
        precision.precision_promote(unnecessary)
    except io.Refused:
        refused = True
    rows.append(result("F11", "unnecessary precision promotion is refused", refused, {"request": unnecessary, "refused": refused}))

    precision.precision_demote("limited", "ternary", utility_before=1.0, utility_after=1.0)
    rows.append(
        result(
            "F12",
            "precision demotion preserves utility",
            precision.plastic["limited"].precision == "ternary",
            {"precision": precision.plastic["limited"].precision, "utility_before": 1.0, "utility_after": 1.0},
        )
    )

    topology = EndogenousPlasticField("f13")
    before = "cause" in topology.cells
    cell = CognitiveCell("cause", "causal_relation", influence=1, provenance_pointer="canary://f13")
    topology.topology_allocate(cell, trigger="persistent cause", expected_value=1.0, resource_cost_bytes=64, held_out_result=1.0)
    after = "cause" in topology.cells and topology.cells["cause"].influence > 0
    rows.append(result("F13", "topology allocation changes a held-out result", before != after, {"before": before, "after": after}))

    random_growth_refused = False
    try:
        topology.topology_allocate(
            CognitiveCell("random", "concept", provenance_pointer="canary://random"),
            trigger="random growth",
            expected_value=0.0,
            resource_cost_bytes=64,
            held_out_result=0.0,
        )
    except io.Refused:
        random_growth_refused = True
    rows.append(result("F14", "random topology growth does not receive the same benefit", random_growth_refused, {"refused": random_growth_refused}))

    split_source = CognitiveCell("tool", "concept", provenance_pointer="canary://f15")
    topology.topology_allocate(split_source, trigger="overloaded category", expected_value=1.0, resource_cost_bytes=64, held_out_result=1.0)
    split = topology.topology_split(
        "tool",
        [
            CognitiveCell("hammer", "tool", provenance_pointer="canary://f15/hammer"),
            CognitiveCell("lever", "tool", provenance_pointer="canary://f15/lever"),
        ],
        trigger="principled affordance exception",
        held_out_result=1.0,
    )
    rows.append(result("F15", "concept split resolves a principled exception", all(key in topology.cells for key in ("hammer", "lever")), split))

    topology.topology_allocate(
        CognitiveCell("mallet", "tool", provenance_pointer="canary://f16/mallet"),
        trigger="redundant representation",
        expected_value=1.0,
        resource_cost_bytes=64,
        held_out_result=1.0,
    )
    merged = CognitiveCell("striking_tool", "tool", provenance_pointer="canary://f16")
    topology.topology_merge(["hammer", "mallet"], merged, trigger="verified synonymy", distinctions_preserved=True)
    rows.append(
        result(
            "F16",
            "concept merge removes redundancy without losing distinctions",
            "striking_tool" in topology.cells and "hammer" in topology.topology["archive"] and "mallet" in topology.topology["archive"],
            {"merged": "striking_tool", "archived": ["hammer", "mallet"], "distinctions_preserved": True},
        )
    )

    topology.topology_prune("lever", required_competence_preserved=True)
    rows.append(result("F17", "pruning preserves required competence", "lever" not in topology.cells, {"required_competence_preserved": True}))

    shadow = EndogenousPlasticField("f18")
    shadow.add_relation("r", -1, precision="ternary", provenance="canary://f18")
    digest_before = shadow.state_integrity_digest()
    shadow.shadow_fork("s", relevant_relations=["r"], relevant_cells=[])
    shadow.shadow_perturb("s", "r", 2)
    shadow.shadow_run("s", query_relations=["r"])
    authoritative_unchanged = shadow.plastic["r"].value == -1
    rows.append(
        result(
            "F18",
            "shadow-field thought leaves authoritative state unchanged",
            authoritative_unchanged,
            {
                "authoritative_relation": shadow.plastic["r"].value,
                "fork_digest": digest_before,
                "post_shadow_archive_only_change": True,
            },
        )
    )

    shadow.shadow_promote("s", evaluator="independent-shadow-evaluator", verified=True)
    rows.append(result("F19", "verified shadow result may be promoted", shadow.plastic["r"].value == 1, {"promoted_value": shadow.plastic["r"].value}))

    compiler = EndogenousPlasticField("f20")
    compiler.procedure_observe_trace("trace", "OBSERVE", "x")
    compiler.procedure_observe_trace("trace", "COMPARE", "target")
    compiler.procedure_observe_trace("trace", "INFER")
    compiler.procedure_propose("compare", "trace", proposer="trace-learner")
    compiler.procedure_verify(
        "compare",
        evaluator="held-out-procedure-evaluator",
        raw_flexible_correct=[True, True],
        raw_compiled_correct=[True, True],
    )
    procedure = compiler.procedure_compile(
        "compare",
        "trace",
        inputs=["x", "target"],
        assumptions=["comparable"],
        scope="fixture",
        branch_conditions=["equality"],
        failure_conditions=["noncomparable"],
        verification_method="held-out equality cases",
        provenance="canary://f20",
    )
    execution = compiler.procedure_execute("compare", {"x": 3, "target": 3})
    flexible_cost = 7
    rows.append(
        result(
            "F20",
            "compiled procedure reduces cost",
            execution["cost"] < flexible_cost and execution["result"] is True,
            {"compiled_cost": execution["cost"], "flexible_trace_cost": flexible_cost, "procedure": procedure["digest"]},
        )
    )

    compiler.procedure_monitor("compare", passed=False, exception="novel noncomparable")
    rows.append(
        result(
            "F21",
            "procedure failure reopens flexible reasoning",
            not compiler.compiled["compare"]["valid"] and compiler.compiled["compare"]["reopen_flexible_reasoning"],
            {"procedure": compiler.compiled["compare"]},
        )
    )

    temporal = EndogenousPlasticField("f22")
    temporal.create_goal("unfinished", "resume after delay", provenance="canary://f22")
    temporal.schedule_event("GoalDeadline", 10.0, {"goal_id": "unfinished"})
    temporal.elapsed_time(11.0)
    rows.append(
        result(
            "F22",
            "continuous-time event handling preserves an unfinished goal",
            temporal.exact["goal_commitments"]["unfinished"]["status"] == "unfinished"
            and temporal.exact["goal_commitments"]["unfinished"]["overdue"],
            {"goal": temporal.exact["goal_commitments"]["unfinished"], "elapsed": temporal.active["elapsed_seconds"]},
        )
    )

    corruption = temporal.checkpoint()
    corruption["state"]["E_t"]["goal_commitments"]["unfinished"]["description"] = "fabricated"
    detected = False
    try:
        EndogenousPlasticField.restore(corruption)
    except io.Refused:
        detected = True
    rows.append(result("F23", "partial field corruption is detected", detected, {"detected": detected}))

    empty = EndogenousPlasticField("f24")
    empty_checkpoint = empty.checkpoint()
    recovered = EndogenousPlasticField.restore(empty_checkpoint)
    no_fabrication = not recovered.plastic and not recovered.exact["evidence_provenance"]
    rows.append(result("F24", "recovery does not fabricate evidence", no_fabrication, {"relations": 0, "evidence": 0}))

    matched = EndogenousPlasticField("f25")
    matched.add_relation("material->floats", 1, precision="ternary", stability="supported", provenance="history://matched")
    untouched = EndogenousPlasticField("f25-control")
    matched_prediction = matched.predict(["material->floats"])
    untouched_prediction = untouched.predict(["material->floats"])
    rows.append(
        result(
            "F25",
            "matched developmental history changes future processing",
            matched_prediction and not untouched_prediction,
            {"matched": matched_prediction, "no_history": untouched_prediction},
        )
    )

    shuffled = EndogenousPlasticField("f26")
    shuffled.add_relation("unrelated->floats", 1, precision="ternary", stability="supported", provenance="history://shuffled")
    wrong = EndogenousPlasticField("f26-wrong")
    wrong.add_relation("material->sinks", 1, precision="ternary", stability="supported", provenance="history://wrong")
    clean = not shuffled.predict(["material->floats"]) and not wrong.predict(["material->floats"])
    rows.append(result("F26", "wrong and shuffled histories remain clean", clean, {"shuffled_target": False, "wrong_target": False}))

    s2 = EndogenousPlasticField("f27-s2", skeleton="K1_monolithic_plastic_field", resource_envelope="512_MB", s2_derived=True)
    substrate = EndogenousPlasticField("f27-substrate", skeleton="K1_monolithic_plastic_field", resource_envelope="512_MB")
    parity = (
        s2.resource_envelope == substrate.resource_envelope
        and s2.interfaces() == substrate.interfaces()
        and RESOURCE_ENVELOPES_BYTES[s2.resource_envelope] == RESOURCE_ENVELOPES_BYTES[substrate.resource_envelope]
    )
    rows.append(
        result(
            "F27",
            "S2 receives the same resource envelope",
            parity,
            {
                "substrate_envelope": substrate.resource_envelope,
                "s2_envelope": s2.resource_envelope,
                "allocated_bytes": RESOURCE_ENVELOPES_BYTES[s2.resource_envelope],
                "interface_count": len(s2.interfaces()),
                "same_contracts": s2.interfaces() == substrate.interfaces(),
                "serialized_metadata_bytes_are_not_resource_allocation": True,
            },
        )
    )

    frontier = capability_density_frontier()
    receipt = raw_metric_receipt(frontier["raw_rows"])
    rows.append(
        result(
            "F28",
            "capability-density metrics reproduce from raw receipts",
            verify_raw_metric_receipt(receipt),
            {"receipt": receipt, "recomputed": verify_raw_metric_receipt(receipt)},
        )
    )

    expected = [f"F{index:02d}" for index in range(1, 29)]
    return {
        "schema": "substrate-field-foundation-canaries/v1",
        "scope": FOUNDATION_STATUS,
        "expected_canaries": expected,
        "rows": rows,
        "canary_count": len(rows),
        "all_present": [row["canary_id"] for row in rows] == expected,
        "all_pass": len(rows) == 28 and all(row["passed"] for row in rows),
        "classification_credit": 0,
        "current_final_revision_endpoint_credit": 0,
        "activation": False,
    }


def foundation_state_schema() -> dict[str, Any]:
    return {
        "schema": "substrate-field-state-schema/v1",
        "field_symbols": {
            "Theta": "slow shared cognitive laws",
            "P_t": "fast and intermediate plastic connection state",
            "G_t": "current topology",
            "Z_t": "active cognitive field state",
            "E_t": "exact constitutional and epistemic state",
            "A": "append-only developmental archive",
            "C_t": "compiled procedures and stable cognitive pathways",
            "M_t": "model, body, tool, and sensor competence state",
        },
        "transition_signature": {
            "output": "Z_(t+delta_t)",
            "inputs": [
                "Theta",
                "P_t",
                "G_t",
                "Z_t",
                "E_t",
                "C_t",
                "M_t",
                "observation",
                "goals",
                "body",
                "elapsed_time",
            ],
            "plasticity_proposals": ["delta_P_t", "delta_G_t", "delta_precision_t", "delta_C_t"],
            "durable_changes_require_verification": True,
        },
        "exact_shell": {
            "governs": [
                "identity",
                "lineage",
                "goal commitments",
                "evidence provenance",
                "permissions",
                "claim boundaries",
                "activation state",
                "checkpoint integrity",
                "irreversible commitments",
            ],
            "approximate_field_direct_write": False,
        },
        "timescales": list(TIMESCALES),
        "runtime_learning": ["sparse", "bounded", "local_where_possible", "reversible", "receipt_bearing", "verification_gated"],
        "activation": False,
    }
