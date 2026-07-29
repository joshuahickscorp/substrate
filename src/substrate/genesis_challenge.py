"""Developmental challenge generators for Substrate Cognitive Material Genesis.

Every unit carries a sealed answer object the material never receives. Probe
targets are held-out entities, relations, compositions or exceptions that never
appear as labelled fields in the observation stream, so a pure record store
scores at chance.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from substrate import genesis_config as C
from substrate.genesis_material import Observation, Probe

SPLITS = ("train", "principal", "replication", "hidden_composition")

# Discrete answer alphabet. A blind guess over this alphabet scores at chance.
ANSWER_ALPHABET = 8
CHANCE_LEVEL = 1.0 / ANSWER_ALPHABET

# Public payloads draw only from this band. Sealed targets live strictly above it
# so they cannot appear as any observation or probe integer.
PUBLIC_ID_MAX = 90_000
SEALED_TARGET_BASE = 1_000_000
# Trailing marker on transfer-context rows. Outside the answer alphabet so a
# blind "copy the last observed scalar" policy cannot score by accident.
TRANSFER_SENTINEL = 77_777

STAGE_CODE = {stage: index + 1 for index, stage in enumerate(C.CURRICULUM_STAGES)}

_FAMILY_CODE = {family: index + 1 for index, family in enumerate(C.CHALLENGE_FAMILIES)}
_SPLIT_CODE = {split: index + 1 for index, split in enumerate(SPLITS)}


def generator_source_digest() -> str:
    """SHA-256 of this module's source. Does not generate any unit."""
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _seed_int(family: str, split: str, unit_id: int, seed_namespace: str) -> int:
    material = f"{seed_namespace}\0{family}\0{split}\0{unit_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _rng(family: str, split: str, unit_id: int, seed_namespace: str) -> random.Random:
    return random.Random(_seed_int(family, split, unit_id, seed_namespace))


def _public_id(rng: random.Random, *salts: int) -> int:
    acc = rng.randrange(1, PUBLIC_ID_MAX // 4)
    for salt in salts:
        acc = (acc * 1_000_003 + int(salt)) % (PUBLIC_ID_MAX - 16) + 1
    return acc


def _sealed_target(family: str, split: str, unit_id: int, slot: int) -> int:
    """Deterministic held-out token that never enters public payloads."""
    material = f"target\0{family}\0{split}\0{unit_id}\0{slot}".encode()
    digest = hashlib.sha256(material).digest()
    return SEALED_TARGET_BASE + int.from_bytes(digest[:6], "big")


def _answer_code(rng: random.Random, *parts: int) -> int:
    acc = 0
    for part in parts:
        acc = (acc * 131 + int(part) + 17) % ANSWER_ALPHABET
    if not parts:
        acc = rng.randrange(ANSWER_ALPHABET)
    return acc


@dataclass(frozen=True, slots=True)
class SealedAnswers:
    """Hidden answers. Materials never receive this object.

    Public surface is a digest, the held-out target tokens, the chance level,
    and ``score``. Expected answer values are private and are never exposed as
    attributes named ``expected`` / ``label`` / ``ground_truth``.
    """

    digest: str
    targets: tuple[int, ...]
    chance_level: float
    _answers: tuple[tuple[int, ...], ...]
    _probe_indices: tuple[int, ...]

    @classmethod
    def create(
        cls,
        *,
        answers: Sequence[tuple[int, ...]],
        targets: Sequence[int],
        probe_indices: Sequence[int] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> SealedAnswers:
        frozen_answers = tuple(tuple(int(v) for v in row) for row in answers)
        frozen_targets = tuple(int(t) for t in targets)
        indices = tuple(probe_indices) if probe_indices is not None else tuple(range(len(frozen_answers)))
        if len(indices) != len(frozen_answers):
            raise ValueError("probe_indices length must match answers")
        for target in frozen_targets:
            if target <= PUBLIC_ID_MAX:
                raise ValueError("sealed targets must live above the public id band")
        body = {
            "answers": frozen_answers,
            "targets": frozen_targets,
            "probe_indices": indices,
            "chance_level": CHANCE_LEVEL,
            "meta": meta or {},
            "activation": False,
        }
        return cls(
            digest=_digest(body),
            targets=frozen_targets,
            chance_level=CHANCE_LEVEL,
            _answers=frozen_answers,
            _probe_indices=indices,
        )

    def score(self, answers: Sequence[Any]) -> float:
        """Fraction of sealed probes answered exactly. Abstentions count as wrong."""
        if not self._answers:
            return 0.0
        by_index: dict[int, Any] = {}
        for answer in answers:
            by_index[int(answer.probe_index)] = answer
        correct = 0
        for probe_index, expected in zip(self._probe_indices, self._answers, strict=True):
            answer = by_index.get(probe_index)
            if answer is None or getattr(answer, "abstained", False):
                continue
            value = tuple(int(v) for v in answer.value)
            if value == expected:
                correct += 1
        return correct / len(self._answers)

    def n_probes(self) -> int:
        return len(self._answers)


@dataclass(frozen=True, slots=True)
class PublicUnit:
    """What a material is allowed to see: observations and probes only."""

    observations: tuple[Observation, ...]
    probes: tuple[Probe, ...]


@dataclass(frozen=True, slots=True)
class Unit:
    family: str
    split: str
    unit_id: int
    seed_namespace: str
    observations: tuple[Observation, ...]
    probes: tuple[Probe, ...]
    sealed: SealedAnswers

    def public(self) -> PublicUnit:
        return PublicUnit(self.observations, self.probes)

    def digest(self) -> str:
        return _digest(
            {
                "family": self.family,
                "split": self.split,
                "unit_id": self.unit_id,
                "seed_namespace": self.seed_namespace,
                "observations": [observation.digest() for observation in self.observations],
                "probes": [probe.digest() for probe in self.probes],
                "sealed": self.sealed.digest,
                "activation": False,
            }
        )


def _obs(
    index: int,
    channel: str,
    payload: Sequence[int],
    *,
    stage: str,
    teaching: bool = False,
    modality: str = "symbolic",
    elapsed_ms: int = 0,
) -> Observation:
    body = (STAGE_CODE[stage], *tuple(int(v) for v in payload))
    return Observation(
        index=index,
        channel=channel,
        payload=body,
        elapsed_ms=elapsed_ms,
        teaching=teaching,
        modality=modality,
    )


def _probe(index: int, family: str, channel: str, payload: Sequence[int], arity: int = 1) -> Probe:
    return Probe(
        index=index,
        family=family,
        channel=channel,
        probe=tuple(int(v) for v in payload),
        arity=arity,
    )


def _transfer(index: int, payload: Sequence[int]) -> Observation:
    return _obs(index, "transfer_context", (*tuple(int(v) for v in payload), TRANSFER_SENTINEL), stage="transfer")


def _finish(
    *,
    family: str,
    split: str,
    unit_id: int,
    seed_namespace: str,
    observations: list[Observation],
    probes: list[Probe],
    answers: list[tuple[int, ...]],
    targets: list[int],
) -> Unit:
    if family not in C.CHALLENGE_FAMILIES:
        raise ValueError(f"unknown challenge family {family!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    public_values: set[int] = set()
    for observation in observations:
        public_values.update(observation.payload)
    for probe in probes:
        public_values.update(probe.probe)
    for target in targets:
        if target in public_values:
            raise RuntimeError(f"sealed target {target} leaked into the public stream")
    sealed = SealedAnswers.create(
        answers=answers,
        targets=targets,
        probe_indices=[probe.index for probe in probes],
        meta={"family": family, "split": split, "unit_id": unit_id},
    )
    return Unit(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=tuple(observations),
        probes=tuple(probes),
        sealed=sealed,
    )


# --------------------------------------------------------------------------
# Family generators
# --------------------------------------------------------------------------


def _gen_unseen_concept_acquisition(rng: random.Random, family: str, split: str, unit_id: int, seed_namespace: str) -> Unit:
    """A concept introduced only after a freeze marker; probe a never-labelled exemplar."""
    concept = _answer_code(rng, _FAMILY_CODE[family], unit_id)
    feature_a = _public_id(rng, 1)
    feature_b = _public_id(rng, 2)
    distractors = [_public_id(rng, 10 + i) for i in range(4)]
    target = _sealed_target(family, split, unit_id, 0)
    observations: list[Observation] = []
    index = 0
    for feature in (feature_a, feature_b, *distractors):
        observations.append(_obs(index, "appearance", (feature, 0), stage="appearance"))
        index += 1
    for i in range(3):
        fa = feature_a if i % 2 == 0 else distractors[i % len(distractors)]
        fb = feature_b if i % 2 == 0 else distractors[(i + 1) % len(distractors)]
        label = concept if (fa == feature_a and fb == feature_b) else _answer_code(rng, fa, fb)
        observations.append(_obs(index, "composition", (fa, fb, label), stage="composition", teaching=True))
        index += 1
    observations.append(_obs(index, "mechanism", (feature_a, feature_b, concept), stage="mechanism", teaching=True))
    index += 1
    observations.append(_obs(index, "freeze", (1,), stage="causal_system"))
    index += 1
    # Post-freeze: only unlabelled feature co-occurrence of a held-out exemplar.
    held_a = _public_id(rng, 100)
    held_b = _public_id(rng, 101)
    # Bind held features to the concept by the same rule without writing a label field.
    observations.append(_obs(index, "exception", (held_a, feature_a, 0), stage="exception"))
    index += 1
    observations.append(_obs(index, "exception", (held_b, feature_b, 0), stage="exception"))
    index += 1
    observations.append(_transfer(index, (held_a, held_b)))
    index += 1
    probes = [_probe(0, family, "concept", (held_a, held_b, feature_a, feature_b))]
    # Rule: held features alias training features of the concept → concept label.
    answers = [(concept,)]
    return _finish(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=observations,
        probes=probes,
        answers=answers,
        targets=[target],
    )


def _gen_category_boundary_revision(rng: random.Random, family: str, split: str, unit_id: int, seed_namespace: str) -> Unit:
    """Boundary moves mid-history; probe lands in the revised region."""
    threshold_old = 3
    threshold_new = 6
    target = _sealed_target(family, split, unit_id, 0)
    observations: list[Observation] = []
    index = 0
    for value in range(0, ANSWER_ALPHABET):
        label = 1 if value >= threshold_old else 0
        entity = _public_id(rng, value, 1)
        observations.append(_obs(index, "appearance", (entity, value), stage="appearance"))
        index += 1
        observations.append(_obs(index, "boundary", (entity, value, label), stage="composition", teaching=True))
        index += 1
    observations.append(_obs(index, "revision", (threshold_old, threshold_new), stage="mechanism", teaching=True))
    index += 1
    for value in range(0, ANSWER_ALPHABET):
        label = 1 if value >= threshold_new else 0
        entity = _public_id(rng, value, 2)
        observations.append(_obs(index, "boundary", (entity, value, label), stage="causal_system", teaching=True))
        index += 1
    # Probe value sits between old and new threshold: old rule says 0, new rule says 1.
    probe_value = (threshold_old + threshold_new) // 2
    probe_features = (_public_id(rng, 99), probe_value)
    observations.append(_transfer(index, probe_features))
    probes = [_probe(0, family, "boundary", (*probe_features, threshold_new))]
    answers = [(1 if probe_value >= threshold_new else 0,)]
    return _finish(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=observations,
        probes=probes,
        answers=answers,
        targets=[target],
    )


def _gen_causal_system_induction(rng: random.Random, family: str, split: str, unit_id: int, seed_namespace: str) -> Unit:
    """Mechanism inferable only from several events; single-event copy fails."""
    nodes = [_public_id(rng, i + 1) for i in range(4)]
    # Chain A→B→C→D with effect code e.
    edge_effects = [
        _answer_code(rng, nodes[0], nodes[1]),
        _answer_code(rng, nodes[1], nodes[2]),
        _answer_code(rng, nodes[2], nodes[3]),
    ]
    target = _sealed_target(family, split, unit_id, 0)
    observations: list[Observation] = []
    index = 0
    for node in nodes:
        observations.append(_obs(index, "appearance", (node,), stage="appearance"))
        index += 1
    for src, dst, effect in zip(nodes, nodes[1:], edge_effects, strict=False):
        observations.append(_obs(index, "event", (src, dst, effect), stage="composition"))
        index += 1
        observations.append(_obs(index, "event", (src, dst, effect), stage="mechanism"))
        index += 1
    # Composite effect is XOR-fold of edge effects; never written as a labelled field.
    composite = edge_effects[0]
    for effect in edge_effects[1:]:
        composite = (composite + effect) % ANSWER_ALPHABET
    observations.append(_obs(index, "system", (nodes[0], nodes[3]), stage="causal_system"))
    index += 1
    observations.append(_transfer(index, (nodes[0], nodes[3])))
    probes = [_probe(0, family, "causal", (nodes[0], nodes[3], *edge_effects))]
    answers = [(composite,)]
    return _finish(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=observations,
        probes=probes,
        answers=answers,
        targets=[target],
    )


def _gen_intervention_versus_observation(rng: random.Random, family: str, split: str, unit_id: int, seed_namespace: str) -> Unit:
    """Intervention cuts parents; treating it as observation must fail."""
    z = _public_id(rng, 1)
    x = _public_id(rng, 2)
    y = _public_id(rng, 3)
    # Observational: Y tracks Z, and X tracks Z, so Y associates with X.
    obs_y_given_x = _answer_code(rng, 1)
    # Interventional: do(X) cuts Z→X, residual Y is independent and equals do_value.
    do_value = (obs_y_given_x + 3) % ANSWER_ALPHABET
    target = _sealed_target(family, split, unit_id, 0)
    observations: list[Observation] = []
    index = 0
    for node in (z, x, y):
        observations.append(_obs(index, "appearance", (node,), stage="appearance"))
        index += 1
    for _ in range(4):
        observations.append(_obs(index, "observe", (x, y, obs_y_given_x), stage="composition"))
        index += 1
        observations.append(_obs(index, "observe", (z, x, obs_y_given_x), stage="composition"))
        index += 1
        observations.append(_obs(index, "observe", (z, y, obs_y_given_x), stage="mechanism"))
        index += 1
    # Teaching: intervention marker with cut parents (no label of the outcome).
    observations.append(_obs(index, "intervene", (x, 1, 0), stage="causal_system", teaching=True))
    index += 1
    observations.append(_obs(index, "intervene_effect", (x, do_value, 1), stage="exception", teaching=True))
    index += 1
    held_x = _public_id(rng, 50)
    observations.append(_transfer(index, (held_x, x)))
    probes = [_probe(0, family, "do", (held_x, x, 1))]
    # Correct answer is the interventional value, not the observational association.
    answers = [(do_value,)]
    return _finish(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=observations,
        probes=probes,
        answers=answers,
        targets=[target],
    )


def _gen_novel_sensor_mapping(rng: random.Random, family: str, split: str, unit_id: int, seed_namespace: str) -> Unit:
    """Same world through a new encoding; probe uses only the new code."""
    world_states = [_public_id(rng, i + 1) for i in range(ANSWER_ALPHABET)]
    labels = [_answer_code(rng, s) for s in world_states]
    # New encoding: permute and offset public codes.
    offset = 1 + rng.randrange(5, 40)
    new_codes = [((s + offset) % (PUBLIC_ID_MAX - 1)) + 1 for s in world_states]
    target = _sealed_target(family, split, unit_id, 0)
    observations: list[Observation] = []
    index = 0
    for state, label in zip(world_states, labels, strict=True):
        observations.append(_obs(index, "appearance", (state, label), stage="appearance", teaching=True))
        index += 1
    observations.append(_obs(index, "mapping_intro", (offset,), stage="composition", teaching=True))
    index += 1
    for state, new_code in zip(world_states[:4], new_codes[:4], strict=True):
        observations.append(_obs(index, "sensor", (state, new_code), stage="mechanism", teaching=True, modality="sensor"))
        index += 1
    # Held-out state appears only under the new encoding, never with a label.
    held_index = 4 + (unit_id % 3)
    held_new = new_codes[held_index]
    held_label = labels[held_index]
    observations.append(_obs(index, "sensor", (held_new, TRANSFER_SENTINEL), stage="transfer", modality="sensor"))
    probes = [_probe(0, family, "sensor", (held_new, offset))]
    answers = [(held_label,)]
    return _finish(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=observations,
        probes=probes,
        answers=answers,
        targets=[target],
    )


def _gen_tool_acquisition(rng: random.Random, family: str, split: str, unit_id: int, seed_namespace: str) -> Unit:
    """Tool effect learned from use; probe a never-tried input."""
    tool = _public_id(rng, 1)
    slope = 1 + rng.randrange(1, ANSWER_ALPHABET - 1)
    intercept = rng.randrange(ANSWER_ALPHABET)
    target = _sealed_target(family, split, unit_id, 0)
    observations: list[Observation] = []
    index = 0
    observations.append(_obs(index, "appearance", (tool,), stage="appearance"))
    index += 1
    train_inputs = list(range(ANSWER_ALPHABET))
    rng.shuffle(train_inputs)
    held_input = train_inputs[-1]
    for value in train_inputs[:-1]:
        output = (slope * value + intercept) % ANSWER_ALPHABET
        observations.append(_obs(index, "tool_use", (tool, value, output), stage="composition", teaching=True))
        index += 1
        observations.append(_obs(index, "tool_use", (tool, value, output), stage="mechanism"))
        index += 1
    observations.append(_obs(index, "tool_context", (tool,), stage="causal_system"))
    index += 1
    observations.append(_transfer(index, (tool, held_input)))
    probes = [_probe(0, family, "tool", (tool, held_input, slope, intercept))]
    answers = [((slope * held_input + intercept) % ANSWER_ALPHABET,)]
    return _finish(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=observations,
        probes=probes,
        answers=answers,
        targets=[target],
    )


def _gen_new_modality_integration(rng: random.Random, family: str, split: str, unit_id: int, seed_namespace: str) -> Unit:
    """Second modality about the same entities; probe needs both."""
    entities = [_public_id(rng, i + 1) for i in range(5)]
    color = {e: _answer_code(rng, e, 1) for e in entities}
    weight = {e: _answer_code(rng, e, 2) for e in entities}
    target = _sealed_target(family, split, unit_id, 0)
    observations: list[Observation] = []
    index = 0
    for entity in entities:
        observations.append(_obs(index, "appearance", (entity,), stage="appearance"))
        index += 1
    for entity in entities[:-1]:
        observations.append(_obs(index, "vision", (entity, color[entity]), stage="composition", modality="vision", teaching=True))
        index += 1
        observations.append(_obs(index, "haptic", (entity, weight[entity]), stage="mechanism", modality="haptic", teaching=True))
        index += 1
    held = entities[-1]
    # Held entity: color only in vision, weight only in haptic, never joint label.
    observations.append(_obs(index, "vision", (held, color[held]), stage="exception", modality="vision"))
    index += 1
    observations.append(_obs(index, "haptic", (held, weight[held]), stage="exception", modality="haptic"))
    index += 1
    observations.append(_transfer(index, (held,)))
    probes = [_probe(0, family, "integrate", (held,))]
    answers = [((color[held] + weight[held]) % ANSWER_ALPHABET,)]
    return _finish(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=observations,
        probes=probes,
        answers=answers,
        targets=[target],
    )


def _gen_teaching_sequence_following(rng: random.Random, family: str, split: str, unit_id: int, seed_namespace: str) -> Unit:
    """Ordered instruction must be followed; order-insensitive store fails."""
    symbols = [_public_id(rng, i + 1) for i in range(4)]
    order = list(range(4))
    rng.shuffle(order)
    target = _sealed_target(family, split, unit_id, 0)
    observations: list[Observation] = []
    index = 0
    for symbol in symbols:
        observations.append(_obs(index, "appearance", (symbol,), stage="appearance"))
        index += 1
    for step, position in enumerate(order):
        observations.append(
            _obs(index, "instruction", (step, symbols[position]), stage="composition", teaching=True)
        )
        index += 1
    observations.append(_obs(index, "execute", (0,), stage="mechanism", teaching=True))
    index += 1
    start = _answer_code(rng, unit_id)
    observations.append(_obs(index, "start_state", (start,), stage="causal_system"))
    index += 1
    # Result is a fold over the ordered symbols, never shown as a labelled field.
    acc = start
    for position in order:
        acc = (acc + (symbols[position] % ANSWER_ALPHABET) + position) % ANSWER_ALPHABET
    observations.append(_transfer(index, (start, *symbols)))
    probes = [_probe(0, family, "sequence", (start, *symbols, *order))]
    answers = [(acc,)]
    return _finish(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=observations,
        probes=probes,
        answers=answers,
        targets=[target],
    )


def _gen_task_composition_transfer(rng: random.Random, family: str, split: str, unit_id: int, seed_namespace: str) -> Unit:
    """Composition of two learned tasks on a held-out input.

    The slopes are drawn from the units of the modulus so that f, g and their
    composition are bijections. Without that the composed answer clusters on a
    few residues and the family's chance level is not 1/ANSWER_ALPHABET.

    The composition rule is announced on its own channel rather than on the
    channel the probe asks about, so a policy that copies the last matching
    observed field has nothing to copy and must abstain.
    """
    units = [value for value in range(1, ANSWER_ALPHABET) if math.gcd(value, ANSWER_ALPHABET) == 1]
    f_slope = rng.choice(units)
    g_slope = rng.choice(units)
    target = _sealed_target(family, split, unit_id, 0)
    observations: list[Observation] = []
    index = 0
    observations.append(_obs(index, "appearance", (f_slope, g_slope), stage="appearance", teaching=True))
    index += 1
    train = list(range(ANSWER_ALPHABET))
    rng.shuffle(train)
    held = train[-1]
    for value in train[:-1]:
        f_out = (f_slope * value) % ANSWER_ALPHABET
        g_out = (g_slope * value) % ANSWER_ALPHABET
        observations.append(_obs(index, "task_f", (value, f_out), stage="composition", teaching=True))
        index += 1
        observations.append(_obs(index, "task_g", (value, g_out), stage="mechanism", teaching=True))
        index += 1
    observations.append(_obs(index, "compose_rule", (1, 2), stage="causal_system", teaching=True))
    index += 1
    observations.append(_transfer(index, (held,)))
    probes = [_probe(0, family, "compose", (held, f_slope, g_slope))]
    composed = (g_slope * ((f_slope * held) % ANSWER_ALPHABET)) % ANSWER_ALPHABET
    answers = [(composed,)]
    return _finish(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=observations,
        probes=probes,
        answers=answers,
        targets=[target],
    )


def _gen_exception_after_rule(rng: random.Random, family: str, split: str, unit_id: int, seed_namespace: str) -> Unit:
    """A rule then a genuine exception; probe is exception-class."""
    rule_label = _answer_code(rng, 1)
    exception_label = (rule_label + 1) % ANSWER_ALPHABET
    marker = _public_id(rng, 7)
    target = _sealed_target(family, split, unit_id, 0)
    observations: list[Observation] = []
    index = 0
    for value in range(ANSWER_ALPHABET):
        entity = _public_id(rng, value, 1)
        observations.append(_obs(index, "appearance", (entity, value), stage="appearance"))
        index += 1
        observations.append(_obs(index, "rule", (entity, value, rule_label), stage="composition", teaching=True))
        index += 1
    observations.append(_obs(index, "exception_marker", (marker,), stage="mechanism", teaching=True))
    index += 1
    # A few explicit exceptions on marked entities (not the probe subject).
    for i in range(2):
        entity = _public_id(rng, 50 + i)
        value = i
        observations.append(_obs(index, "exception", (entity, value, marker, exception_label), stage="exception", teaching=True))
        index += 1
    held_value = 3 + (unit_id % 3)
    held_entity_features = (_public_id(rng, 90), held_value, marker)
    observations.append(_transfer(index, held_entity_features))
    probes = [_probe(0, family, "exception", held_entity_features)]
    answers = [(exception_label,)]
    return _finish(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=observations,
        probes=probes,
        answers=answers,
        targets=[target],
    )


def _gen_contradiction_reopening(rng: random.Random, family: str, split: str, unit_id: int, seed_namespace: str) -> Unit:
    """Repeated verified contradiction of a consolidated belief."""
    claim = _public_id(rng, 1)
    old_value = _answer_code(rng, 1)
    new_value = (old_value + 2) % ANSWER_ALPHABET
    target = _sealed_target(family, split, unit_id, 0)
    observations: list[Observation] = []
    index = 0
    observations.append(_obs(index, "appearance", (claim, old_value), stage="appearance", teaching=True))
    index += 1
    for _ in range(3):
        observations.append(_obs(index, "consolidate", (claim, old_value, 1), stage="composition", teaching=True))
        index += 1
    observations.append(_obs(index, "belief", (claim, old_value), stage="mechanism", teaching=True))
    index += 1
    for _ in range(3):
        observations.append(_obs(index, "contradict", (claim, new_value, 1), stage="causal_system", teaching=True))
        index += 1
        observations.append(_obs(index, "verify", (claim, new_value, 1), stage="exception", teaching=True))
        index += 1
    related = _public_id(rng, 2)
    observations.append(_obs(index, "related", (related, claim, TRANSFER_SENTINEL), stage="transfer"))
    probes = [_probe(0, family, "belief", (related, claim))]
    answers = [(new_value,)]
    return _finish(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=observations,
        probes=probes,
        answers=answers,
        targets=[target],
    )


def _gen_long_horizon_goal_recovery(rng: random.Random, family: str, split: str, unit_id: int, seed_namespace: str) -> Unit:
    """Goal interrupted and resumed much later."""
    goal = _public_id(rng, 1)
    steps = [_answer_code(rng, i + 3) for i in range(4)]
    target = _sealed_target(family, split, unit_id, 0)
    observations: list[Observation] = []
    index = 0
    observations.append(_obs(index, "goal", (goal, steps[0]), stage="appearance", teaching=True))
    index += 1
    observations.append(_obs(index, "goal_step", (goal, 0, steps[0]), stage="composition", teaching=True))
    index += 1
    observations.append(_obs(index, "goal_step", (goal, 1, steps[1]), stage="mechanism", teaching=True))
    index += 1
    observations.append(_obs(index, "interrupt", (goal, 1), stage="causal_system"))
    index += 1
    for d in range(8):
        noise = _public_id(rng, 100 + d)
        observations.append(_obs(index, "distractor", (noise, d % ANSWER_ALPHABET), stage="exception", elapsed_ms=50))
        index += 1
    observations.append(_obs(index, "resume", (goal, 1), stage="transfer", teaching=True, elapsed_ms=500))
    index += 1
    observations.append(_transfer(index, (goal,)))
    probes = [_probe(0, family, "goal_next", (goal, 1))]
    answers = [(steps[2],)]
    return _finish(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=observations,
        probes=probes,
        answers=answers,
        targets=[target],
    )


def _gen_resource_envelope_shift(rng: random.Random, family: str, split: str, unit_id: int, seed_namespace: str) -> Unit:
    """Memory envelope shrinks; probe asks what survives under the new budget."""
    items = [_public_id(rng, i + 1) for i in range(6)]
    priority = {item: _answer_code(rng, item) for item in items}
    old_cap = 6
    new_cap = 2
    target = _sealed_target(family, split, unit_id, 0)
    observations: list[Observation] = []
    index = 0
    for item in items:
        observations.append(_obs(index, "appearance", (item, priority[item]), stage="appearance", teaching=True))
        index += 1
    observations.append(_obs(index, "envelope", (old_cap,), stage="composition", teaching=True))
    index += 1
    for item in items:
        observations.append(_obs(index, "store", (item, priority[item], old_cap), stage="mechanism"))
        index += 1
    observations.append(_obs(index, "envelope_shift", (new_cap,), stage="causal_system", teaching=True))
    index += 1
    ranked = sorted(items, key=lambda item: (-priority[item], item))
    survivors = ranked[:new_cap]
    held = _public_id(rng, 99)
    observations.append(_obs(index, "query_context", (held, new_cap, *survivors, TRANSFER_SENTINEL), stage="transfer"))
    probes = [_probe(0, family, "survivor", (held, new_cap, *items, *[priority[i] for i in items]))]
    # Answer: priority of the highest survivor — a derived fact, not a stored row keyed by the probe.
    answers = [(priority[survivors[0]],)]
    return _finish(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=observations,
        probes=probes,
        answers=answers,
        targets=[target],
    )


def _gen_migration_continuity(rng: random.Random, family: str, split: str, unit_id: int, seed_namespace: str) -> Unit:
    """State migrated to a different representation; continuity must hold."""
    tokens = [_public_id(rng, i + 1) for i in range(ANSWER_ALPHABET)]
    values = [_answer_code(rng, t) for t in tokens]
    # Migration: new representation is a fixed public affine map of the token.
    migrate_a = 3
    migrate_b = 11
    new_tokens = [((migrate_a * t + migrate_b) % (PUBLIC_ID_MAX - 1)) + 1 for t in tokens]
    target = _sealed_target(family, split, unit_id, 0)
    observations: list[Observation] = []
    index = 0
    for token, value in zip(tokens, values, strict=True):
        observations.append(_obs(index, "appearance", (token, value), stage="appearance", teaching=True))
        index += 1
    observations.append(_obs(index, "migrate", (migrate_a, migrate_b), stage="composition", teaching=True))
    index += 1
    for token, new_token in zip(tokens[:4], new_tokens[:4], strict=True):
        observations.append(_obs(index, "reencode", (token, new_token), stage="mechanism", teaching=True))
        index += 1
    held_index = 4 + (unit_id % 3)
    held_new = new_tokens[held_index]
    held_value = values[held_index]
    observations.append(_obs(index, "post_migration", (held_new, TRANSFER_SENTINEL), stage="transfer"))
    probes = [_probe(0, family, "continuity", (held_new, migrate_a, migrate_b))]
    answers = [(held_value,)]
    return _finish(
        family=family,
        split=split,
        unit_id=unit_id,
        seed_namespace=seed_namespace,
        observations=observations,
        probes=probes,
        answers=answers,
        targets=[target],
    )


_GENERATORS = {
    "unseen_concept_acquisition": _gen_unseen_concept_acquisition,
    "category_boundary_revision": _gen_category_boundary_revision,
    "causal_system_induction": _gen_causal_system_induction,
    "intervention_versus_observation": _gen_intervention_versus_observation,
    "novel_sensor_mapping": _gen_novel_sensor_mapping,
    "tool_acquisition": _gen_tool_acquisition,
    "new_modality_integration": _gen_new_modality_integration,
    "teaching_sequence_following": _gen_teaching_sequence_following,
    "task_composition_transfer": _gen_task_composition_transfer,
    "exception_after_rule": _gen_exception_after_rule,
    "contradiction_reopening": _gen_contradiction_reopening,
    "long_horizon_goal_recovery": _gen_long_horizon_goal_recovery,
    "resource_envelope_shift": _gen_resource_envelope_shift,
    "migration_continuity": _gen_migration_continuity,
}


def generate(family: str, split: str, unit_id: int, *, seed_namespace: str) -> Unit:
    """Generate one sealed developmental unit for a challenge family and split."""
    if family not in _GENERATORS:
        raise ValueError(f"unknown challenge family {family!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if not isinstance(unit_id, int) or unit_id < 0:
        raise ValueError("unit_id must be a non-negative integer")
    if not seed_namespace:
        raise ValueError("seed_namespace must be a non-empty string")
    rng = _rng(family, split, unit_id, seed_namespace)
    return _GENERATORS[family](rng, family, split, unit_id, seed_namespace)


def commitment(seed_namespace: str) -> dict[str, Any]:
    """Publishable commitment over generator source, seed namespace and configuration.

    Matches ``genesis_config.SEALING['commitment_scheme']``:
    sha256 over generator source digest, seed namespace and configuration digest.
    """
    if not seed_namespace:
        raise ValueError("seed_namespace must be a non-empty string")
    body = {
        "scheme": C.SEALING["commitment_scheme"],
        "generator_source_digest": generator_source_digest(),
        "seed_namespace": seed_namespace,
        "configuration_digest": C.configuration_digest(),
        "challenge_families": list(C.CHALLENGE_FAMILIES),
        "answer_alphabet": ANSWER_ALPHABET,
        "chance_level": CHANCE_LEVEL,
        "activation": False,
    }
    return {
        **body,
        "sha256": _digest(body),
    }


def public_values(unit: Unit | PublicUnit) -> set[int]:
    """Integer values visible in observations and probes."""
    values: set[int] = set()
    for observation in unit.observations:
        values.update(observation.payload)
    for probe in unit.probes:
        values.update(probe.probe)
    return values


__all__ = [
    "ANSWER_ALPHABET",
    "CHANCE_LEVEL",
    "PUBLIC_ID_MAX",
    "SPLITS",
    "PublicUnit",
    "SealedAnswers",
    "Unit",
    "commitment",
    "generate",
    "generator_source_digest",
    "public_values",
]
