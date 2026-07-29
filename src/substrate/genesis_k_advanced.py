"""Advanced genesis cognitive materials: K9, K10 and K11.

Each material is a separate class with its own durable-change law. Observation and
answer touch active state only; durable writes happen only after a positive
verdict on a proposal. No material reads held-out labels.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from substrate import genesis_config as C
from substrate.final_revision_field import (
    PRECISION_ALPHABETS,
    PackedRadix,
    multiplication_light_dot,
    native_low_bit_update,
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

TERNARY = PRECISION_ALPHABETS["ternary"]
QUINARY = PRECISION_ALPHABETS["quinary"]
SEVEN = PRECISION_ALPHABETS["seven_state_powers_of_two"]
GROUP_TERNARY = 5
GROUP_QUINARY = 3

# Cap on proposals emitted per consolidation cycle. Explicit and matched across advanced materials.
PROPOSALS_PER_CYCLE = 32

STABILITY_RANK = ("new", "provisional", "supported", "consolidated", "reopened", "refuted")
PAYLOAD_ALPHABETS: dict[str, tuple[int, ...]] = {
    "quinary": QUINARY,
    "seven_state_powers_of_two": SEVEN,
    "4_bit": PRECISION_ALPHABETS["4_bit"],
}


def _clamp_to_alphabet(value: int | float, alphabet: Sequence[int]) -> int:
    return min((int(symbol) for symbol in alphabet), key=lambda symbol: (abs(symbol - float(value)), symbol))


def _encode_ternary(payload: Sequence[int], dim: int) -> list[int]:
    out = [0] * dim
    if not payload:
        return out
    for index, raw in enumerate(payload):
        out[index % dim] = _clamp_to_alphabet(int(raw), TERNARY)
    # Sparse: zero every third coordinate to keep keys addressable.
    for index in range(0, dim, 3):
        if index < dim:
            out[index] = 0
    return out


def _pack_vector(values: Sequence[int], alphabet: Sequence[int], group_size: int) -> dict[str, Any]:
    packed = pack_radix(list(values), alphabet, group_size=group_size)
    return {
        "alphabet": list(packed.alphabet),
        "count": packed.count,
        "group_size": packed.group_size,
        "bit_length": packed.bit_length,
        "payload_hex": packed.payload_hex,
    }


def _unpack_vector(document: Mapping[str, Any]) -> list[int]:
    packed = PackedRadix(
        alphabet=tuple(int(v) for v in document["alphabet"]),
        count=int(document["count"]),
        group_size=int(document["group_size"]),
        bit_length=int(document["bit_length"]),
        payload_hex=str(document["payload_hex"]),
    )
    return unpack_radix(packed)


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())


def _shell_identity(name: str) -> dict[str, Any]:
    return {
        "identity": name,
        "lineage": (name,),
        "evidence_provenance": (),
        "goal_commitments": (),
        "permissions": ("observe", "answer", "propose", "apply", "checkpoint"),
        "activation_state": False,
        "claim_boundary": dict(C.CLAIM_BOUNDARY),
        "checkpoint_integrity": "sha256_durable_active",
        "activation": False,
    }


# ---------------------------------------------------------------------------
# K9 — prediction error gates every durable write and precision request
# ---------------------------------------------------------------------------


@dataclass
class K9_predictive_plastic_field(MaterialBase):
    """Durable rewrites fire only when a non-zero prediction error is present."""

    dim: int = 12
    error_threshold: int = 1
    _predictors: list[int] = field(default_factory=list, repr=False)
    _plastic: list[int] = field(default_factory=list, repr=False)
    _precision_map: dict[str, str] = field(default_factory=dict, repr=False)
    _topology: dict[str, Any] = field(default_factory=dict, repr=False)
    _compiled: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _archive: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _shell: dict[str, Any] = field(default_factory=dict, repr=False)
    _last_observation: list[int] = field(default_factory=list, repr=False)
    _last_prediction: list[int] = field(default_factory=list, repr=False)
    _prediction_error: list[int] = field(default_factory=list, repr=False)
    _error_energy: int = 0
    _proposal_seq: int = 0
    _undo: dict[str, Any] = field(default_factory=dict, repr=False)
    _utility_window: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self._predictors:
            self._predictors = [0] * self.dim
        if not self._plastic:
            self._plastic = [0] * self.dim
        if not self._precision_map:
            self._precision_map = {"predictors": "ternary", "plastic": "quinary"}
        if not self._topology:
            self._topology = {"nodes": ["predictor_bank", "plastic_bank"], "edges": [], "archive": []}
        if not self._shell:
            self._shell = _shell_identity(self.name)
        if not self._last_observation:
            self._last_observation = [0] * self.dim
        if not self._last_prediction:
            self._last_prediction = [0] * self.dim
        if not self._prediction_error:
            self._prediction_error = [0] * self.dim
        self._resize()

    def _resize(self) -> None:
        self._opportunity.ledger.resize(self._resident_bytes())

    def _resident_bytes(self) -> int:
        return _json_bytes(self._durable_state())

    def _predict(self, observation: Sequence[int]) -> list[int]:
        self._opportunity.ledger.spend(1, precision_bits=self.dim * 2)
        predicted: list[int] = []
        for index in range(self.dim):
            score = multiplication_light_dot(
                observation,
                [self._predictors[(index + offset) % self.dim] for offset in range(self.dim)],
            )["value"]
            predicted.append(_clamp_to_alphabet(score + self._plastic[index], TERNARY))
        return predicted

    def _transition(self, observation: Observation) -> None:
        encoded = _encode_ternary(observation.payload, self.dim)
        prediction = self._predict(self._last_observation if any(self._last_observation) else encoded)
        error = [
            _clamp_to_alphabet(int(encoded[i]) - int(prediction[i]), QUINARY)
            for i in range(self.dim)
        ]
        self._last_observation = list(encoded)
        self._last_prediction = list(prediction)
        self._prediction_error = list(error)
        self._error_energy = sum(abs(value) for value in error)
        self._opportunity.ledger.spend(1)

    def _answer(self, probe: Probe) -> Answer:
        encoded = _encode_ternary(probe.probe, self.dim)
        prediction = self._predict(encoded)
        mixed = [
            _clamp_to_alphabet(prediction[i] + self._plastic[i], QUINARY)
            for i in range(min(probe.arity, self.dim))
        ]
        while len(mixed) < probe.arity:
            mixed.append(0)
        confidence = max(0, 8 - self._error_energy)
        return Answer(probe_index=probe.index, value=tuple(mixed[: probe.arity]), confidence=confidence, abstained=False)

    def _gate_open(self) -> bool:
        if "prediction_error_gate_on_every_durable_write" in self.frozen_mechanisms:
            return False
        return self._error_energy >= self.error_threshold

    def proposals_per_cycle(self) -> int:
        return PROPOSALS_PER_CYCLE

    def _propose(self) -> Iterable[Proposal]:
        if not self._gate_open():
            return ()
        self._opportunity.ledger.spend(self.dim * 2)
        error = list(self._prediction_error)
        if all(value == 0 for value in error):
            return ()

        precision_request = None
        if self._error_energy >= 2 * self.error_threshold and self._precision_map.get("plastic") == "quinary":
            self._utility_window.append(float(self._error_energy))
            if len(self._utility_window) > C.PRECISION_AUDIT_WINDOW:
                self._utility_window = self._utility_window[-C.PRECISION_AUDIT_WINDOW :]
            mean_utility = sum(self._utility_window) / max(1, len(self._utility_window))
            if mean_utility >= C.MINIMUM_UTILITY_PER_ADDED_BYTE * 100:
                precision_request = "seven_state_powers_of_two"

        candidates: list[tuple[float, tuple[int, ...], str, str | None]] = []

        # Full residual correction (historical single proposal).
        full = tuple(_clamp_to_alphabet(-error[i], QUINARY) for i in range(self.dim))
        if not all(value == 0 for value in full):
            candidates.append((float(self._error_energy), full, "full", precision_request))
            # Scaled full residual.
            for scale, tag in ((2, "scale2"), (-1, "invert")):
                scaled = tuple(_clamp_to_alphabet(-error[i] * scale, QUINARY) for i in range(self.dim))
                if all(v == 0 for v in scaled) or scaled == full:
                    continue
                candidates.append((float(self._error_energy) * abs(scale) * 0.5, scaled, tag, None))

        # Per-site residual rewrites ordered by absolute residual (largest prediction error first).
        sites = sorted(range(self.dim), key=lambda i: (-abs(error[i]), i))
        for site in sites:
            if error[site] == 0:
                continue
            for scale, tag in ((1, "site"), (2, "site2"), (-1, "site_inv")):
                step = _clamp_to_alphabet(-error[site] * scale, QUINARY)
                if step == 0:
                    continue
                delta = tuple(step if i == site else 0 for i in range(self.dim))
                candidates.append((float(abs(error[site]) * abs(scale)) + 0.01 * (self.dim - site), delta, f"{tag}:{site}", None))

        # Top-k residual coalitions: correct the k largest residual sites together.
        nonzero = [i for i in sites if error[i] != 0]
        for k in range(2, min(6, len(nonzero)) + 1):
            coalition = nonzero[:k]
            delta = tuple(
                _clamp_to_alphabet(-error[i], QUINARY) if i in coalition else 0 for i in range(self.dim)
            )
            if all(v == 0 for v in delta):
                continue
            score = float(sum(abs(error[i]) for i in coalition))
            candidates.append((score, delta, f"top{k}", precision_request if k == len(nonzero) else None))

        # Deduplicate and emit best-first under the cycle cap.
        seen: set[tuple[int, ...]] = set()
        emitted: list[Proposal] = []
        for score, delta, tag, prec in sorted(candidates, key=lambda row: (-row[0], row[2], row[1])):
            if delta in seen or all(v == 0 for v in delta):
                continue
            # Skip no-ops relative to current predictor/plastic banks.
            would_change = False
            alphabet_plastic = PAYLOAD_ALPHABETS.get(self._precision_map.get("plastic", "quinary"), QUINARY)
            for index, step in enumerate(delta):
                if step == 0:
                    continue
                new_pred = native_low_bit_update(self._predictors[index], -float(step), TERNARY, learning_rate=1.0)
                new_plastic = _clamp_to_alphabet(self._plastic[index] + int(step), alphabet_plastic)
                if new_pred != self._predictors[index] or new_plastic != self._plastic[index]:
                    would_change = True
                    break
            if not would_change:
                continue
            seen.add(delta)
            self._proposal_seq += 1
            emitted.append(
                Proposal(
                    proposal_id=f"k9-pe-{self._proposal_seq}:{tag}",
                    kind="PlasticityPropose",
                    target="predictor_plastic_bank",
                    delta=delta,
                    precision_request=prec,
                    trigger="prediction_error",
                    expected_value=score,
                    cost_bytes=max(1, sum(abs(v) for v in delta)),
                )
            )
            if len(emitted) >= PROPOSALS_PER_CYCLE:
                break
        return tuple(emitted)

    def _snapshot_durable(self) -> dict[str, Any]:
        return copy.deepcopy(self._durable_state())

    def _commit(self, proposal: Proposal) -> None:
        self._undo[proposal.proposal_id] = self._snapshot_durable()
        alphabet_plastic = PAYLOAD_ALPHABETS.get(self._precision_map.get("plastic", "quinary"), QUINARY)
        for index, step in enumerate(proposal.delta):
            if index >= self.dim:
                break
            self._predictors[index] = native_low_bit_update(
                self._predictors[index],
                -float(step),
                TERNARY,
                learning_rate=1.0,
            )
            self._plastic[index] = _clamp_to_alphabet(self._plastic[index] + int(step), alphabet_plastic)
        if proposal.precision_request and self._gate_open():
            self._precision_map["plastic"] = proposal.precision_request
            self._topology["nodes"] = list(dict.fromkeys([*self._topology["nodes"], f"precision:{proposal.precision_request}"]))
        self._archive.append(
            {
                "proposal_id": proposal.proposal_id,
                "error_energy": self._error_energy,
                "activation": False,
            }
        )
        self._compiled.append(
            {
                "kind": "prediction_error_rewrite",
                "proposal_id": proposal.proposal_id,
                "activation": False,
            }
        )
        self._error_energy = 0
        self._prediction_error = [0] * self.dim
        self._opportunity.ledger.spend(self.dim)
        self._resize()

    def _rollback(self, receipt: Receipt) -> None:
        snapshot = self._undo.pop(receipt.proposal_id, None)
        if snapshot is None:
            return
        self._restore_durable(snapshot)
        self._resize()

    def _durable_state(self) -> Any:
        return {
            "predictors": _pack_vector(self._predictors, TERNARY, GROUP_TERNARY),
            "plastic": _pack_vector(
                self._plastic,
                PAYLOAD_ALPHABETS.get(self._precision_map.get("plastic", "quinary"), QUINARY),
                GROUP_QUINARY,
            ),
            "precision_map": dict(self._precision_map),
            "topology": copy.deepcopy(self._topology),
            "compiled_procedures": copy.deepcopy(self._compiled),
            "archive": copy.deepcopy(self._archive),
            "shell": copy.deepcopy(self._shell),
            "dim": self.dim,
            "activation": False,
        }

    def _active_state(self) -> Any:
        return {
            "last_observation": list(self._last_observation),
            "last_prediction": list(self._last_prediction),
            "prediction_error": list(self._prediction_error),
            "error_energy": self._error_energy,
            "proposal_seq": self._proposal_seq,
            "utility_window": list(self._utility_window),
            "activation": False,
        }

    def _restore_durable(self, state: Any) -> None:
        document = dict(state)
        self.dim = int(document.get("dim", self.dim))
        self._predictors = _unpack_vector(document["predictors"])
        self._plastic = _unpack_vector(document["plastic"])
        self._precision_map = dict(document["precision_map"])
        self._topology = copy.deepcopy(document["topology"])
        self._compiled = copy.deepcopy(document["compiled_procedures"])
        self._archive = copy.deepcopy(document["archive"])
        self._shell = copy.deepcopy(document["shell"])
        self._resize()

    def _restore_active(self, state: Any) -> None:
        document = dict(state)
        self._last_observation = list(document.get("last_observation", [0] * self.dim))
        self._last_prediction = list(document.get("last_prediction", [0] * self.dim))
        self._prediction_error = list(document.get("prediction_error", [0] * self.dim))
        self._error_energy = int(document.get("error_energy", 0))
        self._proposal_seq = int(document.get("proposal_seq", 0))
        self._utility_window = list(document.get("utility_window", []))


def _factory_k9(opportunity: Opportunity, **options: Any) -> K9_predictive_plastic_field:
    return K9_predictive_plastic_field(
        name="K9_predictive_plastic_field",
        mechanism="prediction_error_gated_durable_rewrite",
        _opportunity=opportunity,
        dim=int(options.get("dim", 12)),
        error_threshold=int(options.get("error_threshold", 1)),
    )


register("K9_predictive_plastic_field", _factory_k9)


# ---------------------------------------------------------------------------
# K11 — interference-gated sparse fiber rebind (Grok-original material)
# ---------------------------------------------------------------------------


@dataclass
class _Fiber:
    key: list[int]
    payload: list[int]
    stability: str
    precision: str
    interference_ema: int
    provenance: str
    utility: int
    occupied: bool

    def document(self) -> dict[str, Any]:
        alphabet = PAYLOAD_ALPHABETS.get(self.precision, QUINARY)
        return {
            "key": _pack_vector(self.key, TERNARY, GROUP_TERNARY),
            "payload": _pack_vector(self.payload, alphabet, GROUP_QUINARY),
            "stability": self.stability,
            "precision": self.precision,
            "interference_ema": self.interference_ema,
            "provenance": self.provenance,
            "utility": self.utility,
            "occupied": self.occupied,
            "activation": False,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> _Fiber:
        return cls(
            key=_unpack_vector(document["key"]),
            payload=_unpack_vector(document["payload"]),
            stability=str(document["stability"]),
            precision=str(document["precision"]),
            interference_ema=int(document["interference_ema"]),
            provenance=str(document["provenance"]),
            utility=int(document.get("utility", 0)),
            occupied=bool(document.get("occupied", True)),
        )


@dataclass
class K11_interference_gated_sparse_fiber_field(MaterialBase):
    """Fixed-capacity sparse fibers; durable rebind only under interference pressure."""

    capacity: int = 8
    key_dim: int = 12
    payload_dim: int = 4
    top_k: int = 3
    tau: int = 3
    hamming_budget: int = 2
    _fibers: list[_Fiber] = field(default_factory=list, repr=False)
    _precision_map: dict[str, str] = field(default_factory=dict, repr=False)
    _topology: dict[str, Any] = field(default_factory=dict, repr=False)
    _compiled: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _shell: dict[str, Any] = field(default_factory=dict, repr=False)
    _probe_q: list[int] = field(default_factory=list, repr=False)
    _active_ids: list[int] = field(default_factory=list, repr=False)
    _active_weights: list[int] = field(default_factory=list, repr=False)
    _reconstructed: list[int] = field(default_factory=list, repr=False)
    _interference_residual: int = 0
    _last_payload_target: list[int] = field(default_factory=list, repr=False)
    _active_interference_ema: dict[int, int] = field(default_factory=dict, repr=False)
    _proposal_seq: int = 0
    _undo: dict[str, Any] = field(default_factory=dict, repr=False)
    _audit_scores: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self._fibers:
            self._fibers = [self._empty_fiber(index) for index in range(self.capacity)]
            # Seed overlapping keys with disagreeing payloads so interference can rise.
            for index in range(min(4, self.capacity)):
                fiber = self._fibers[index]
                fiber.occupied = True
                fiber.key = [0] * self.key_dim
                fiber.key[0] = 1
                fiber.key[1 + (index % 3)] = 1 if index % 2 == 0 else -1
                fiber.key[(index * 2 + 3) % self.key_dim] = -1
                fiber.payload = [0] * self.payload_dim
                fiber.payload[0] = (-2, -1, 1, 2)[index % 4]
                fiber.payload[1] = (2, -2, 1, -1)[index % 4]
                fiber.provenance = f"seed:{index}"
                fiber.stability = "provisional"
        if not self._precision_map:
            self._precision_map = {"keys": "ternary", "payloads": "quinary"}
        if not self._topology:
            self._topology = {
                "bank_capacity": self.capacity,
                "free_slots": [i for i, f in enumerate(self._fibers) if not f.occupied],
                "archive": [],
                "splits": 0,
                "fuses": 0,
            }
        if not self._shell:
            self._shell = _shell_identity(self.name)
        if not self._probe_q:
            self._probe_q = [0] * self.key_dim
        if not self._reconstructed:
            self._reconstructed = [0] * self.payload_dim
        if not self._last_payload_target:
            self._last_payload_target = [0] * self.payload_dim
        self._resize()

    def _empty_fiber(self, index: int) -> _Fiber:
        return _Fiber(
            key=[0] * self.key_dim,
            payload=[0] * self.payload_dim,
            stability="new",
            precision="quinary",
            interference_ema=0,
            provenance=f"slot:{index}",
            utility=0,
            occupied=False,
        )

    def _resize(self) -> None:
        self._opportunity.ledger.resize(self._resident_bytes())

    def _resident_bytes(self) -> int:
        return _json_bytes(self._durable_state())

    def _overlap(self, probe: Sequence[int], key: Sequence[int]) -> int:
        score = 0
        for left, right in zip(probe, key, strict=True):
            if left == 0 or right == 0:
                continue
            if left == right:
                score += 1
            else:
                score -= 1
        return score

    def _soft_activate(self, probe: Sequence[int]) -> tuple[list[int], list[int]]:
        scored: list[tuple[int, int]] = []
        for index, fiber in enumerate(self._fibers):
            if not fiber.occupied:
                continue
            scored.append((self._overlap(probe, fiber.key), index))
        scored.sort(key=lambda row: (-row[0], row[1]))
        chosen = scored[: self.top_k]
        ids = [index for _, index in chosen]
        # Soft weights in quinary, shifted to non-negative ranks.
        weights = [_clamp_to_alphabet(score, QUINARY) for score, _ in chosen]
        return ids, weights

    def _reconstruct(self, ids: Sequence[int], weights: Sequence[int]) -> list[int]:
        if not ids:
            return [0] * self.payload_dim
        totals = [0] * self.payload_dim
        mass = 0
        for fiber_id, weight in zip(ids, weights, strict=True):
            fiber = self._fibers[fiber_id]
            w = abs(int(weight)) + 1
            mass += w
            for axis, value in enumerate(fiber.payload):
                totals[axis] += int(value) * w
        alphabet = PAYLOAD_ALPHABETS.get(self._precision_map.get("payloads", "quinary"), QUINARY)
        if mass <= 0:
            return [0] * self.payload_dim
        return [_clamp_to_alphabet(total / mass, alphabet) for total in totals]

    def _interference(self, ids: Sequence[int], weights: Sequence[int]) -> int:
        if len(ids) < 2:
            return 0
        residual = 0
        # Collision energy among co-active bindings that disagree on payload.
        for left in range(len(ids)):
            for right in range(left + 1, len(ids)):
                a = self._fibers[ids[left]].payload
                b = self._fibers[ids[right]].payload
                disagreement = sum(1 for x, y in zip(a, b, strict=True) if x != y)
                residual += disagreement * (abs(weights[left]) + abs(weights[right]) + 1)
        # Over-subscription: many actives on one fiber is impossible here; proxy via high EMA.
        for fiber_id in ids:
            residual += max(0, self._fibers[fiber_id].interference_ema - 2)
        return int(residual)

    def _step(self, cue: Sequence[int], target_payload: Sequence[int] | None = None) -> None:
        probe = _encode_ternary(cue, self.key_dim)
        self._opportunity.ledger.spend(max(1, self.capacity // 2), retrievals=1, precision_bits=self.key_dim)
        ids, weights = self._soft_activate(probe)
        reconstructed = self._reconstruct(ids, weights)
        residual = self._interference(ids, weights)
        for fiber_id in ids:
            prior = self._active_interference_ema.get(fiber_id, self._fibers[fiber_id].interference_ema)
            self._active_interference_ema[fiber_id] = min(8, (prior * 3 + residual) // 4)
        self._probe_q = list(probe)
        self._active_ids = list(ids)
        self._active_weights = list(weights)
        self._reconstructed = list(reconstructed)
        self._interference_residual = residual
        if target_payload is not None:
            self._last_payload_target = [
                _clamp_to_alphabet(int(target_payload[i % len(target_payload)]), QUINARY)
                for i in range(self.payload_dim)
            ]
        else:
            self._last_payload_target = list(reconstructed)

    def _transition(self, observation: Observation) -> None:
        self._step(observation.payload, target_payload=observation.payload)

    def _answer(self, probe: Probe) -> Answer:
        self._step(probe.probe)
        value = tuple(self._reconstructed[: probe.arity])
        while len(value) < probe.arity:
            value = (*value, 0)
        confidence = max(0, 8 - self._interference_residual)
        return Answer(probe_index=probe.index, value=value[: probe.arity], confidence=confidence, abstained=False)

    def _mechanism_open(self) -> bool:
        return "interference_gated_rebind_split_fuse" not in self.frozen_mechanisms

    def proposals_per_cycle(self) -> int:
        return PROPOSALS_PER_CYCLE

    def _fiber_rebind_parts(self, target_id: int) -> tuple[list[int], list[int], int] | None:
        fiber = self._fibers[target_id]
        if not fiber.occupied and target_id not in self._active_ids:
            return None
        key_delta = [0] * self.key_dim
        flips = 0
        for index in range(self.key_dim):
            if flips >= self.hamming_budget:
                break
            if self._probe_q[index] != 0 and fiber.key[index] != self._probe_q[index]:
                key_delta[index] = self._probe_q[index]
                flips += 1
        payload_delta = [
            _clamp_to_alphabet(int(self._last_payload_target[i]) - int(fiber.payload[i]), QUINARY)
            for i in range(self.payload_dim)
        ]
        if flips == 0 and all(v == 0 for v in payload_delta):
            # Force a minimal payload step so a licensed rebind still changes durable state.
            payload_delta[0] = 1 if fiber.payload[0] < 2 else -1
        return key_delta, payload_delta, flips

    def _propose(self) -> Iterable[Proposal]:
        if not self._mechanism_open():
            return ()
        if self._interference_residual < self.tau:
            return ()
        if not self._active_ids:
            return ()
        self._opportunity.ledger.spend(self.capacity + self.key_dim)
        free = [i for i, f in enumerate(self._fibers) if not f.occupied]
        self._audit_scores.append(float(self._interference_residual))
        if len(self._audit_scores) > C.PRECISION_AUDIT_WINDOW:
            self._audit_scores = self._audit_scores[-C.PRECISION_AUDIT_WINDOW :]
        mean_utility = sum(self._audit_scores) / max(1, len(self._audit_scores))

        # Fibers ranked by interference pressure among the active set (and high-EMA bank members).
        ranked_ids: list[tuple[float, int]] = []
        seen_ids: set[int] = set()
        for fiber_id in self._active_ids:
            ema = self._active_interference_ema.get(fiber_id, self._fibers[fiber_id].interference_ema)
            ranked_ids.append((float(self._interference_residual + ema), fiber_id))
            seen_ids.add(fiber_id)
        for fiber_id, fiber in enumerate(self._fibers):
            if not fiber.occupied or fiber_id in seen_ids:
                continue
            if fiber.interference_ema >= self.tau:
                ranked_ids.append((float(fiber.interference_ema), fiber_id))
        ranked_ids.sort(key=lambda row: (-row[0], row[1]))

        candidates: list[tuple[float, Proposal]] = []
        for score, target_id in ranked_ids:
            parts = self._fiber_rebind_parts(target_id)
            if parts is None:
                continue
            key_delta, payload_delta, flips = parts
            fiber = self._fibers[target_id]
            ema = self._active_interference_ema.get(target_id, fiber.interference_ema)
            precision_request = None
            if mean_utility >= C.MINIMUM_UTILITY_PER_ADDED_BYTE * 100 and fiber.precision == "quinary":
                precision_request = "seven_state_powers_of_two"

            # Plain rebind for this fiber.
            delta = (target_id, *key_delta, *payload_delta)
            self._proposal_seq += 1
            candidates.append(
                (
                    score,
                    Proposal(
                        proposal_id=f"k11-rebind-{self._proposal_seq}:f{target_id}",
                        kind="PlasticityPropose",
                        target=f"fiber:{target_id}",
                        delta=delta,
                        precision_request=precision_request,
                        topology_operation=None,
                        trigger="interference_residual",
                        expected_value=score,
                        cost_bytes=max(1, flips + sum(abs(v) for v in payload_delta)),
                    ),
                )
            )
            # Licensed split when EMA pressure is high and capacity remains.
            if ema >= 4 and free:
                self._proposal_seq += 1
                candidates.append(
                    (
                        score + 1.0,
                        Proposal(
                            proposal_id=f"k11-split-{self._proposal_seq}:f{target_id}",
                            kind="PlasticityPropose",
                            target=f"fiber:{target_id}",
                            delta=delta,
                            precision_request=precision_request,
                            topology_operation="FiberSplit",
                            trigger="interference_residual",
                            expected_value=score + 1.0,
                            cost_bytes=max(1, flips + sum(abs(v) for v in payload_delta) + 4),
                        ),
                    )
                )
            # Payload-only and key-only variants remain interference-gated rebinds.
            payload_only = (target_id, *([0] * self.key_dim), *payload_delta)
            if any(payload_delta) and payload_only != delta:
                self._proposal_seq += 1
                candidates.append(
                    (
                        score * 0.75,
                        Proposal(
                            proposal_id=f"k11-payload-{self._proposal_seq}:f{target_id}",
                            kind="PlasticityPropose",
                            target=f"fiber:{target_id}",
                            delta=payload_only,
                            trigger="interference_residual",
                            expected_value=score * 0.75,
                            cost_bytes=max(1, sum(abs(v) for v in payload_delta)),
                        ),
                    )
                )
            key_only = (target_id, *key_delta, *([0] * self.payload_dim))
            if flips > 0 and key_only != delta:
                self._proposal_seq += 1
                candidates.append(
                    (
                        score * 0.7,
                        Proposal(
                            proposal_id=f"k11-key-{self._proposal_seq}:f{target_id}",
                            kind="PlasticityPropose",
                            target=f"fiber:{target_id}",
                            delta=key_only,
                            trigger="interference_residual",
                            expected_value=score * 0.7,
                            cost_bytes=max(1, flips),
                        ),
                    )
                )

        # Fuse operations over co-active pairs when interference residual licenses topology repair.
        if len(self._active_ids) >= 2:
            for left, right in zip(self._active_ids, self._active_ids[1:], strict=False):
                if left == right:
                    continue
                parts = self._fiber_rebind_parts(left)
                if parts is None:
                    continue
                key_delta, payload_delta, flips = parts
                delta = (left, *key_delta, *payload_delta)
                self._proposal_seq += 1
                candidates.append(
                    (
                        float(self._interference_residual) * 0.6,
                        Proposal(
                            proposal_id=f"k11-fuse-{self._proposal_seq}:f{left}+{right}",
                            kind="PlasticityPropose",
                            target=f"fiber:{left}",
                            delta=delta,
                            topology_operation="FiberFuse",
                            trigger="interference_residual",
                            expected_value=float(self._interference_residual) * 0.6,
                            cost_bytes=max(1, flips + sum(abs(v) for v in payload_delta)),
                        ),
                    )
                )

        candidates.sort(key=lambda row: (-row[0], row[1].proposal_id))
        return tuple(proposal for _score, proposal in candidates[:PROPOSALS_PER_CYCLE])

    def _snapshot_durable(self) -> dict[str, Any]:
        return copy.deepcopy(self._durable_state())

    def _commit(self, proposal: Proposal) -> None:
        self._undo[proposal.proposal_id] = self._snapshot_durable()
        if not self._mechanism_open():
            return
        parts = list(proposal.delta)
        if not parts:
            return
        target_id = int(parts[0])
        key_delta = [int(v) for v in parts[1 : 1 + self.key_dim]]
        payload_delta = [int(v) for v in parts[1 + self.key_dim : 1 + self.key_dim + self.payload_dim]]
        fiber = self._fibers[target_id]
        if not fiber.occupied:
            fiber.occupied = True
            fiber.provenance = f"rebind:{proposal.proposal_id}"
        # Key rebind under Hamming budget.
        flips = 0
        for index, step in enumerate(key_delta):
            if step == 0:
                continue
            if flips >= self.hamming_budget:
                break
            fiber.key[index] = _clamp_to_alphabet(step, TERNARY)
            flips += 1
        alphabet = PAYLOAD_ALPHABETS.get(fiber.precision, QUINARY)
        for index, step in enumerate(payload_delta):
            if index >= self.payload_dim:
                break
            fiber.payload[index] = _clamp_to_alphabet(fiber.payload[index] + int(step), alphabet)
        fiber.utility += 1
        fiber.interference_ema = self._active_interference_ema.get(target_id, fiber.interference_ema)
        if fiber.stability == "new":
            fiber.stability = "provisional"
        elif fiber.stability == "provisional":
            fiber.stability = "supported"
        if proposal.precision_request:
            fiber.precision = proposal.precision_request
            self._precision_map["payloads"] = proposal.precision_request
            # A demotion narrows the alphabet. Values written while the fiber
            # was wider would then sit outside it and refuse to pack, so the
            # payload is re-clamped at the moment the precision changes rather
            # than at the moment it is serialized.
            narrowed = PAYLOAD_ALPHABETS.get(fiber.precision, QUINARY)
            fiber.payload = [_clamp_to_alphabet(int(value), narrowed) for value in fiber.payload]
        if proposal.topology_operation == "FiberSplit":
            self._fiber_split(target_id)
        elif proposal.topology_operation == "FiberFuse" and len(self._active_ids) >= 2:
            self._fiber_fuse(self._active_ids[0], self._active_ids[1])
        # Envelope pressure: prune zero-utility freeable fibers when near full.
        occupied = sum(1 for f in self._fibers if f.occupied)
        if occupied >= self.capacity:
            self._prune_zero_utility()
        self._compiled.append(
            {
                "kind": "interference_gated_rebind",
                "proposal_id": proposal.proposal_id,
                "target": target_id,
                "activation": False,
            }
        )
        self._topology["free_slots"] = [i for i, f in enumerate(self._fibers) if not f.occupied]
        self._interference_residual = 0
        self._opportunity.ledger.spend(self.key_dim + self.payload_dim)
        self._resize()

    def _fiber_split(self, source_id: int) -> None:
        free = [i for i, f in enumerate(self._fibers) if not f.occupied]
        if not free:
            return
        child_id = free[0]
        source = self._fibers[source_id]
        child = self._fibers[child_id]
        child.occupied = True
        child.precision = source.precision
        child.stability = "new"
        child.provenance = f"split:{source_id}->{child_id}"
        child.payload = list(source.payload)
        child.key = list(source.key)
        mid = self.key_dim // 2
        for index in range(mid, self.key_dim):
            child.key[index] = source.key[index]
            source.key[index] = 0
        for index in range(0, mid):
            child.key[index] = 0
        self._topology["splits"] = int(self._topology.get("splits", 0)) + 1

    def _fiber_fuse(self, left_id: int, right_id: int) -> None:
        if left_id == right_id:
            return
        left = self._fibers[left_id]
        right = self._fibers[right_id]
        if not left.occupied or not right.occupied:
            return
        # Merge into left; free right. Competence check: keep higher-utility payload.
        if right.utility > left.utility:
            left.payload = list(right.payload)
            left.utility = right.utility
        for index in range(self.key_dim):
            if left.key[index] == 0:
                left.key[index] = right.key[index]
        left.interference_ema = min(left.interference_ema, right.interference_ema)
        left.stability = "supported"
        left.provenance = f"fuse:{left_id}+{right_id}"
        self._fibers[right_id] = self._empty_fiber(right_id)
        self._topology["fuses"] = int(self._topology.get("fuses", 0)) + 1
        self._topology.setdefault("archive", []).append({"fused": right_id, "into": left_id, "activation": False})

    def _prune_zero_utility(self) -> None:
        candidates = [
            (fiber.utility, fiber.interference_ema, index)
            for index, fiber in enumerate(self._fibers)
            if fiber.occupied and fiber.utility <= 0 and fiber.stability in {"new", "provisional", "refuted"}
        ]
        if not candidates:
            return
        candidates.sort()
        _, _, index = candidates[0]
        self._topology.setdefault("archive", []).append(self._fibers[index].document())
        self._fibers[index] = self._empty_fiber(index)

    def _rollback(self, receipt: Receipt) -> None:
        snapshot = self._undo.pop(receipt.proposal_id, None)
        if snapshot is None:
            return
        self._restore_durable(snapshot)
        self._resize()

    def _durable_state(self) -> Any:
        return {
            "fibers": [fiber.document() for fiber in self._fibers],
            "precision_map": dict(self._precision_map),
            "topology": copy.deepcopy(self._topology),
            "compiled_procedures": copy.deepcopy(self._compiled),
            "shell": copy.deepcopy(self._shell),
            "capacity": self.capacity,
            "key_dim": self.key_dim,
            "payload_dim": self.payload_dim,
            "tau": self.tau,
            "activation": False,
        }

    def _active_state(self) -> Any:
        return {
            "probe_q": list(self._probe_q),
            "active_ids": list(self._active_ids),
            "active_weights": list(self._active_weights),
            "reconstructed": list(self._reconstructed),
            "interference_residual": self._interference_residual,
            "last_payload_target": list(self._last_payload_target),
            "active_interference_ema": {str(k): int(v) for k, v in self._active_interference_ema.items()},
            "proposal_seq": self._proposal_seq,
            "audit_scores": list(self._audit_scores),
            "activation": False,
        }

    def _restore_durable(self, state: Any) -> None:
        document = dict(state)
        self.capacity = int(document.get("capacity", self.capacity))
        self.key_dim = int(document.get("key_dim", self.key_dim))
        self.payload_dim = int(document.get("payload_dim", self.payload_dim))
        self.tau = int(document.get("tau", self.tau))
        self._fibers = [_Fiber.from_document(row) for row in document["fibers"]]
        self._precision_map = dict(document["precision_map"])
        self._topology = copy.deepcopy(document["topology"])
        self._compiled = copy.deepcopy(document["compiled_procedures"])
        self._shell = copy.deepcopy(document["shell"])
        self._resize()

    def _restore_active(self, state: Any) -> None:
        document = dict(state)
        self._probe_q = list(document.get("probe_q", [0] * self.key_dim))
        self._active_ids = list(document.get("active_ids", []))
        self._active_weights = list(document.get("active_weights", []))
        self._reconstructed = list(document.get("reconstructed", [0] * self.payload_dim))
        self._interference_residual = int(document.get("interference_residual", 0))
        self._last_payload_target = list(document.get("last_payload_target", [0] * self.payload_dim))
        self._active_interference_ema = {
            int(k): int(v) for k, v in dict(document.get("active_interference_ema", {})).items()
        }
        self._proposal_seq = int(document.get("proposal_seq", 0))
        self._audit_scores = list(document.get("audit_scores", []))

    def force_interference(self, residual: int) -> None:
        """Test seam: set residual without durable change. Not a label channel."""
        self._interference_residual = int(residual)
        if not self._active_ids:
            occupied = [i for i, f in enumerate(self._fibers) if f.occupied]
            self._active_ids = occupied[: self.top_k] or [0]
            self._active_weights = [1] * len(self._active_ids)


def _factory_k11(opportunity: Opportunity, **options: Any) -> K11_interference_gated_sparse_fiber_field:
    return K11_interference_gated_sparse_fiber_field(
        name="K11_interference_gated_sparse_fiber_field",
        mechanism="interference_gated_sparse_fiber_rebind",
        _opportunity=opportunity,
        capacity=int(options.get("capacity", 8)),
        key_dim=int(options.get("key_dim", 12)),
        payload_dim=int(options.get("payload_dim", 4)),
        top_k=int(options.get("top_k", 3)),
        tau=int(options.get("tau", 3)),
        hamming_budget=int(options.get("hamming_budget", 2)),
    )


register("K11_interference_gated_sparse_fiber_field", _factory_k11)


# ---------------------------------------------------------------------------
# K10 — integrated composition of K1–K9 pathways under one exact shell
# ---------------------------------------------------------------------------

K10_COMPOSED_MECHANISMS = (
    "monolithic_dense_rewrite",
    "typed_per_edge_plastic_value_scope_and_precision",
    "bounded_radius_local_neighbourhood_rule",
    "elapsed_time_driven_decay_and_expiry",
    "input_dependent_bounded_recurrence",
    "unfrozen_allocate_split_merge_prune_under_rent",
    "per_region_radix_selection_under_rent",
    "append_only_projection_as_the_only_durable_path",
    "prediction_error_gate_on_every_durable_write",
)


@dataclass
class K10_integrated_plastic_field(MaterialBase):
    """Composes K1–K9 durable pathways; ablations freeze one composed mechanism each."""

    dim: int = 10
    _field: list[int] = field(default_factory=list, repr=False)
    _edges: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _cells: list[int] = field(default_factory=list, repr=False)
    _recurrent: list[int] = field(default_factory=list, repr=False)
    _event_log: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _predictors: list[int] = field(default_factory=list, repr=False)
    _precision_map: dict[str, str] = field(default_factory=dict, repr=False)
    _topology: dict[str, Any] = field(default_factory=dict, repr=False)
    _compiled: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _shell: dict[str, Any] = field(default_factory=dict, repr=False)
    _last_input: list[int] = field(default_factory=list, repr=False)
    _active_field: list[int] = field(default_factory=list, repr=False)
    _active_cells: list[int] = field(default_factory=list, repr=False)
    _active_recurrent: list[int] = field(default_factory=list, repr=False)
    _prediction_error: int = 0
    _elapsed_ms: int = 0
    _proposal_seq: int = 0
    _undo: dict[str, Any] = field(default_factory=dict, repr=False)
    _enabled: dict[str, bool] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._field:
            self._field = [0] * self.dim
        if not self._cells:
            self._cells = [0] * self.dim
        if not self._recurrent:
            self._recurrent = [0, 0]
        if not self._predictors:
            self._predictors = [0] * self.dim
        if not self._edges:
            self._edges = [
                {"source": i, "target": (i + 1) % self.dim, "weight": 0, "scope": "local", "precision": "quinary"}
                for i in range(min(3, self.dim))
            ]
        if not self._precision_map:
            self._precision_map = {
                "field": "quinary",
                "cells": "ternary",
                "edges": "quinary",
                "recurrent": "ternary",
            }
        if not self._topology:
            self._topology = {"nodes": list(range(self.dim)), "archive": [], "allocated": self.dim}
        if not self._shell:
            self._shell = _shell_identity(self.name)
        if not self._last_input:
            self._last_input = [0] * self.dim
        if not self._active_field:
            self._active_field = list(self._field)
        if not self._active_cells:
            self._active_cells = list(self._cells)
        if not self._active_recurrent:
            self._active_recurrent = list(self._recurrent)
        if not self._enabled:
            self._enabled = {name: True for name in K10_COMPOSED_MECHANISMS}
        self._resize()

    def freeze_mechanism(self, mechanism: str) -> None:
        super().freeze_mechanism(mechanism)
        if mechanism in self._enabled:
            self._enabled[mechanism] = False

    def _open(self, mechanism: str) -> bool:
        if mechanism in self.frozen_mechanisms:
            return False
        return bool(self._enabled.get(mechanism, True))

    def _resize(self) -> None:
        self._opportunity.ledger.resize(self._resident_bytes())

    def _resident_bytes(self) -> int:
        return _json_bytes(self._durable_state())

    def _transition(self, observation: Observation) -> None:
        encoded = _encode_ternary(observation.payload, self.dim)
        self._opportunity.ledger.spend(2, precision_bits=self.dim)
        self._elapsed_ms += max(0, observation.elapsed_ms)
        predicted = [
            _clamp_to_alphabet(self._predictors[i] + self._field[i], TERNARY)
            for i in range(self.dim)
        ]
        self._prediction_error = sum(abs(int(encoded[i]) - int(predicted[i])) for i in range(self.dim))
        # Active-only integration of all pathways; durable writes wait for propose/commit.
        active_field = list(self._field)
        active_cells = list(self._cells)
        active_recurrent = list(self._recurrent)
        if self._open("monolithic_dense_rewrite"):
            for i in range(self.dim):
                active_field[i] = _clamp_to_alphabet(active_field[i] + encoded[i], QUINARY)
        if self._open("typed_per_edge_plastic_value_scope_and_precision"):
            for edge in self._edges:
                src = int(edge["source"])
                dst = int(edge["target"])
                active_field[dst] = _clamp_to_alphabet(
                    active_field[dst] + int(edge["weight"]) * encoded[src % self.dim],
                    QUINARY,
                )
        if self._open("bounded_radius_local_neighbourhood_rule"):
            for i in range(self.dim):
                left = active_cells[(i - 1) % self.dim]
                right = active_cells[(i + 1) % self.dim]
                active_cells[i] = _clamp_to_alphabet((left + right + encoded[i]) // 3, TERNARY)
        if self._open("elapsed_time_driven_decay_and_expiry"):
            decay = 1 if self._elapsed_ms > 0 else 0
            active_field = [_clamp_to_alphabet(v - decay, QUINARY) for v in active_field]
        if self._open("input_dependent_bounded_recurrence"):
            s0, s1 = active_recurrent
            n0 = _clamp_to_alphabet((7 * s0 + 2 * s1 + encoded[0]) // 8, TERNARY)
            n1 = _clamp_to_alphabet((-1 * s0 + 8 * s1 + encoded[1 % self.dim]) // 8, TERNARY)
            active_recurrent = [n0, n1]
        self._active_field = active_field
        self._active_cells = active_cells
        self._active_recurrent = active_recurrent
        self._last_input = list(encoded)
        self._opportunity.ledger.spend(1)

    def _answer(self, probe: Probe) -> Answer:
        encoded = _encode_ternary(probe.probe, self.dim)
        field = self._active_field or self._field
        cells = self._active_cells or self._cells
        recurrent = self._active_recurrent or self._recurrent
        score = multiplication_light_dot(encoded, field)["value"]
        score += multiplication_light_dot(
            encoded[:2] + [0] * max(0, 2 - len(encoded)),
            recurrent + [0] * max(0, 2 - len(recurrent)),
        )["value"]
        values = [_clamp_to_alphabet(score + cells[i % self.dim], QUINARY) for i in range(probe.arity)]
        return Answer(probe_index=probe.index, value=tuple(values), confidence=max(0, 6 - self._prediction_error), abstained=False)

    def proposals_per_cycle(self) -> int:
        return PROPOSALS_PER_CYCLE

    def _propose(self) -> Iterable[Proposal]:
        # Integration candidate: at least one enabled pathway must justify a write.
        pe_gate = self._open("prediction_error_gate_on_every_durable_write")
        if pe_gate and self._prediction_error <= 0:
            # Prediction-error pathway is active and reports no error: no write.
            # Other pathways may still write only if PE gate is frozen/disabled.
            return ()
        if not pe_gate and self._prediction_error <= 0 and not any(
            self._open(name)
            for name in K10_COMPOSED_MECHANISMS
            if name != "prediction_error_gate_on_every_durable_write"
        ):
            return ()
        # When PE gate is frozen, allow monolithic/event pathways to fire on any observation.
        if not pe_gate and not any(self._last_input):
            return ()
        self._opportunity.ledger.spend(self.dim * 2)
        kinds = [name for name in K10_COMPOSED_MECHANISMS if self._open(name)]
        base = list(self._last_input)
        if all(v == 0 for v in base) and self._prediction_error <= 0:
            base = [1 if i == (self._proposal_seq % self.dim) else 0 for i in range(self.dim)]

        candidates: list[tuple[float, Proposal]] = []

        def emit(delta_vals: Sequence[int], tag: str, *, topology: str | None = None, precision: str | None = None, score: float) -> None:
            delta = tuple(_clamp_to_alphabet(v, QUINARY) for v in delta_vals)
            if all(v == 0 for v in delta) and topology is None and precision is None:
                return
            self._proposal_seq += 1
            candidates.append(
                (
                    score,
                    Proposal(
                        proposal_id=f"k10-int-{self._proposal_seq}:{tag}",
                        kind="PlasticityPropose",
                        target="integrated_shell",
                        delta=delta,
                        precision_request=precision,
                        topology_operation=topology,
                        trigger=("+".join(kinds) if kinds else "inert") + f"|{tag}",
                        expected_value=score,
                        cost_bytes=max(1, sum(abs(v) for v in delta) + len(kinds)),
                    ),
                )
            )

        # Full integrated write plus scaled/inverted variants of the last input.
        full = [_clamp_to_alphabet(v, QUINARY) for v in base]
        emit(full, "full", score=float(self._prediction_error + sum(abs(v) for v in full)))
        emit([_clamp_to_alphabet(v * 2, QUINARY) for v in base], "scale2", score=float(sum(abs(v) for v in base)) * 2)
        emit([_clamp_to_alphabet(-v, QUINARY) for v in base], "invert", score=float(sum(abs(v) for v in base)))

        # Pathway-tagged site rewrites: one per high-energy coordinate, reflecting composed pathways.
        axes = sorted(range(self.dim), key=lambda i: (-abs(base[i]), i))
        for axis in axes:
            if base[axis] == 0 and self._prediction_error <= 0:
                continue
            unit = [0] * self.dim
            unit[axis] = base[axis] if base[axis] != 0 else (1 if self._prediction_error > 0 else 0)
            if unit[axis] == 0:
                continue
            emit(unit, f"axis:{axis}", score=float(abs(unit[axis]) + self._prediction_error * 0.1))

        # Topology allocate variants when the unfrozen allocator pathway is open.
        if self._open("unfrozen_allocate_split_merge_prune_under_rent") and self._prediction_error >= 2:
            for tag in ("alloc_a", "alloc_b", "alloc_c"):
                emit(full, tag, topology="TopologyAllocate", score=float(self._prediction_error) + 2.0)

        # Precision promotion when the radix pathway is open under high residual.
        if self._open("per_region_radix_selection_under_rent") and self._prediction_error >= 3:
            emit(full, "promote", precision="seven_state_powers_of_two", score=float(self._prediction_error) + 3.0)

        # Event-log flavoured integrated writes under the append-only pathway.
        if self._open("append_only_projection_as_the_only_durable_path"):
            for scale in (1, 2):
                emit(
                    [_clamp_to_alphabet(v * scale, QUINARY) for v in base],
                    f"event:{scale}",
                    score=float(sum(abs(v) for v in base)) * scale + 0.5,
                )

        candidates.sort(key=lambda row: (-row[0], row[1].proposal_id))
        return tuple(proposal for _score, proposal in candidates[:PROPOSALS_PER_CYCLE])

    def _snapshot_durable(self) -> dict[str, Any]:
        return copy.deepcopy(self._durable_state())

    def _commit(self, proposal: Proposal) -> None:
        self._undo[proposal.proposal_id] = self._snapshot_durable()
        delta = list(proposal.delta)
        if self._open("monolithic_dense_rewrite"):
            for i, step in enumerate(delta):
                if i >= self.dim:
                    break
                self._field[i] = _clamp_to_alphabet(self._field[i] + int(step), QUINARY)
        if self._open("typed_per_edge_plastic_value_scope_and_precision"):
            for edge in self._edges:
                src = int(edge["source"]) % max(1, len(delta))
                edge["weight"] = _clamp_to_alphabet(int(edge["weight"]) + int(delta[src]), QUINARY)
        if self._open("bounded_radius_local_neighbourhood_rule"):
            for i, step in enumerate(delta):
                if i >= self.dim:
                    break
                self._cells[i] = _clamp_to_alphabet(self._cells[i] + int(step), TERNARY)
        if self._open("elapsed_time_driven_decay_and_expiry"):
            for i in range(self.dim):
                self._field[i] = _clamp_to_alphabet(self._field[i] - (1 if self._elapsed_ms > 100 else 0), QUINARY)
        if self._open("input_dependent_bounded_recurrence"):
            self._recurrent[0] = _clamp_to_alphabet(self._recurrent[0] + (delta[0] if delta else 0), TERNARY)
            self._recurrent[1] = _clamp_to_alphabet(self._recurrent[1] + (delta[1] if len(delta) > 1 else 0), TERNARY)
        if self._open("unfrozen_allocate_split_merge_prune_under_rent") and proposal.topology_operation == "TopologyAllocate":
            new_id = int(self._topology.get("allocated", self.dim))
            self._topology["allocated"] = new_id + 1
            self._topology["nodes"] = list(self._topology.get("nodes", [])) + [new_id]
            self._edges.append(
                {
                    "source": new_id % self.dim,
                    "target": (new_id + 1) % self.dim,
                    "weight": 1,
                    "scope": "allocated",
                    "precision": "quinary",
                }
            )
        if self._open("per_region_radix_selection_under_rent") and proposal.precision_request:
            self._precision_map["field"] = proposal.precision_request
        if self._open("append_only_projection_as_the_only_durable_path"):
            self._event_log.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "delta": list(delta),
                    "trigger": proposal.trigger,
                    "activation": False,
                }
            )
            # Projection from event log into field when this is the only durable path? Integrated uses all.
            # When ONLY this path is open (others frozen), project sum of events.
            if not any(
                self._open(name)
                for name in K10_COMPOSED_MECHANISMS
                if name not in {"append_only_projection_as_the_only_durable_path", "prediction_error_gate_on_every_durable_write"}
            ):
                projected = [0] * self.dim
                for event in self._event_log:
                    for i, step in enumerate(event["delta"]):
                        if i < self.dim:
                            projected[i] = _clamp_to_alphabet(projected[i] + int(step), QUINARY)
                self._field = projected
        if self._open("prediction_error_gate_on_every_durable_write"):
            for i, step in enumerate(delta):
                if i >= self.dim:
                    break
                self._predictors[i] = native_low_bit_update(self._predictors[i], -float(step), TERNARY)
        self._compiled.append(
            {
                "kind": "integrated_commit",
                "proposal_id": proposal.proposal_id,
                "enabled": {name: self._open(name) for name in K10_COMPOSED_MECHANISMS},
                "activation": False,
            }
        )
        self._opportunity.ledger.spend(self.dim + len(self._edges))
        self._resize()

    def _rollback(self, receipt: Receipt) -> None:
        snapshot = self._undo.pop(receipt.proposal_id, None)
        if snapshot is None:
            return
        self._restore_durable(snapshot)
        self._resize()

    def _durable_state(self) -> Any:
        field_alphabet = PAYLOAD_ALPHABETS.get(self._precision_map.get("field", "quinary"), QUINARY)
        return {
            "field": _pack_vector(self._field, field_alphabet, GROUP_QUINARY),
            "cells": _pack_vector(self._cells, TERNARY, GROUP_TERNARY),
            "recurrent": _pack_vector(self._recurrent, TERNARY, 2),
            "predictors": _pack_vector(self._predictors, TERNARY, GROUP_TERNARY),
            "edges": copy.deepcopy(self._edges),
            "event_log": copy.deepcopy(self._event_log),
            "precision_map": dict(self._precision_map),
            "topology": copy.deepcopy(self._topology),
            "compiled_procedures": copy.deepcopy(self._compiled),
            "shell": copy.deepcopy(self._shell),
            "enabled": dict(self._enabled),
            "dim": self.dim,
            "activation": False,
        }

    def _active_state(self) -> Any:
        return {
            "last_input": list(self._last_input),
            "active_field": list(self._active_field),
            "active_cells": list(self._active_cells),
            "active_recurrent": list(self._active_recurrent),
            "prediction_error": self._prediction_error,
            "elapsed_ms": self._elapsed_ms,
            "proposal_seq": self._proposal_seq,
            "activation": False,
        }

    def _restore_durable(self, state: Any) -> None:
        document = dict(state)
        self.dim = int(document.get("dim", self.dim))
        self._field = _unpack_vector(document["field"])
        self._cells = _unpack_vector(document["cells"])
        self._recurrent = _unpack_vector(document["recurrent"])
        self._predictors = _unpack_vector(document["predictors"])
        self._edges = copy.deepcopy(document["edges"])
        self._event_log = copy.deepcopy(document["event_log"])
        self._precision_map = dict(document["precision_map"])
        self._topology = copy.deepcopy(document["topology"])
        self._compiled = copy.deepcopy(document["compiled_procedures"])
        self._shell = copy.deepcopy(document["shell"])
        self._enabled = dict(document.get("enabled", {name: True for name in K10_COMPOSED_MECHANISMS}))
        self._resize()

    def _restore_active(self, state: Any) -> None:
        document = dict(state)
        self._last_input = list(document.get("last_input", [0] * self.dim))
        self._active_field = list(document.get("active_field", list(self._field)))
        self._active_cells = list(document.get("active_cells", list(self._cells)))
        self._active_recurrent = list(document.get("active_recurrent", list(self._recurrent)))
        self._prediction_error = int(document.get("prediction_error", 0))
        self._elapsed_ms = int(document.get("elapsed_ms", 0))
        self._proposal_seq = int(document.get("proposal_seq", 0))

    @classmethod
    def ablations(cls) -> dict[str, Callable[..., K10_integrated_plastic_field]]:
        """Return constructors for K10 with each composed mechanism frozen."""

        def _make(frozen: str) -> Callable[..., K10_integrated_plastic_field]:
            def factory(opportunity: Opportunity, **options: Any) -> K10_integrated_plastic_field:
                material = _factory_k10(opportunity, **options)
                # Owned exclusive mechanisms cannot be frozen via MaterialBase; these
                # composed names are K10-internal switches except true exclusives.
                if frozen in C.EXCLUSIVE_MECHANISMS.get(material.name, ()):
                    material._enabled[frozen] = False
                else:
                    try:
                        material.freeze_mechanism(frozen)
                    except ValueError:
                        material._enabled[frozen] = False
                material._enabled[frozen] = False
                return material

            return factory

        return {name: _make(name) for name in K10_COMPOSED_MECHANISMS}


def _factory_k10(opportunity: Opportunity, **options: Any) -> K10_integrated_plastic_field:
    return K10_integrated_plastic_field(
        name="K10_integrated_plastic_field",
        mechanism="integrated_k1_to_k9_composed_shell",
        _opportunity=opportunity,
        dim=int(options.get("dim", 10)),
    )


register("K10_integrated_plastic_field", _factory_k10)


__all__ = [
    "PROPOSALS_PER_CYCLE",
    "K9_predictive_plastic_field",
    "K10_integrated_plastic_field",
    "K11_interference_gated_sparse_fiber_field",
    "K10_COMPOSED_MECHANISMS",
]
