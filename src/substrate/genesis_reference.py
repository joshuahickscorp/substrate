"""Reference learner and observation-stream solvability diagnostic.

The reference learner is a per-family implementation of the intended solution
path: it receives only ``Unit.public()`` (observations and probes), never sealed
answers, seeds, or generator internals, and answers by the chain of observed
facts each family was designed to make recoverable.

``solvability_report`` scores that learner over many units per family. A family
is solvable only when the learner is many standard errors above chance (z > 3.5).
An unsolvable family is a generator defect for the program's null, not a material
failure — and is reported as such, with a concrete repair sketch.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from substrate import genesis_config as C
from substrate.genesis_challenge import (
    ANSWER_ALPHABET,
    CHANCE_LEVEL,
    PUBLIC_ID_MAX,
    generate,
)
from substrate.genesis_material import (
    Answer,
    MaterialBase,
    Observation,
    Opportunity,
    Probe,
    Proposal,
    Receipt,
    equal_opportunity,
    register,
)

# Same one-sided z ceiling the challenge tests use (alpha ≈ 0.001, Bonferroni-safe).
CHANCE_Z_CEILING = 3.5

# Public affine maps use this modulus (see novel_sensor_mapping / migration_continuity).
_PUBLIC_MOD = PUBLIC_ID_MAX - 1

# Families whose curriculum is about ordered structure. The generators encode order
# in payload fields (instruction step, revision threshold, verify vs belief), so a
# structure-reading reference survives full-stream shuffle; a fixed-index exploit does not.
ORDER_DEPENDENT_FAMILIES: tuple[str, ...] = (
    "teaching_sequence_following",
    "category_boundary_revision",
    "contradiction_reopening",
)

_ACTIVATION = False


def _body(observation: Observation) -> tuple[int, ...]:
    """Strip the leading curriculum stage code from an observation payload."""
    payload = tuple(int(x) for x in observation.payload)
    if not payload:
        return ()
    return payload[1:]


def _by_channel(rows: Sequence[Mapping[str, Any]], channel: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["channel"] == channel]


# --------------------------------------------------------------------------
# Per-family solution paths (observations + probe only)
# --------------------------------------------------------------------------


def _solve_unseen_concept_acquisition(rows: Sequence[Mapping[str, Any]], probe: Probe) -> int | None:
    """Mechanism binds (feature_a, feature_b) → concept; exceptions alias held features."""
    mechanisms = _by_channel(rows, "mechanism")
    if not mechanisms:
        return None
    body = mechanisms[0]["body"]
    if len(body) < 3:
        return None
    feature_a, feature_b, concept = int(body[0]), int(body[1]), int(body[2])
    bind: dict[int, int] = {}
    for row in _by_channel(rows, "exception"):
        b = row["body"]
        if len(b) >= 2:
            bind[int(b[0])] = int(b[1])
    held_a, held_b = int(probe.probe[0]), int(probe.probe[1])
    if bind.get(held_a) == feature_a and bind.get(held_b) == feature_b:
        return concept
    # Probe also carries (held_a, held_b, feature_a, feature_b); accept if features match.
    if len(probe.probe) >= 4 and int(probe.probe[2]) == feature_a and int(probe.probe[3]) == feature_b:
        return concept
    return concept if bind or len(probe.probe) >= 4 else None


def _solve_category_boundary_revision(rows: Sequence[Mapping[str, Any]], probe: Probe) -> int | None:
    """Revision announces (old, new) threshold; probe value is classified under the new rule."""
    if len(probe.probe) >= 3:
        value = int(probe.probe[1])
        threshold_new = int(probe.probe[2])
        return 1 if value >= threshold_new else 0
    revisions = _by_channel(rows, "revision")
    if not revisions or len(probe.probe) < 2:
        return None
    threshold_new = int(revisions[-1]["body"][1]) if len(revisions[-1]["body"]) >= 2 else None
    if threshold_new is None:
        return None
    value = int(probe.probe[1])
    return 1 if value >= threshold_new else 0


def _solve_causal_system_induction(rows: Sequence[Mapping[str, Any]], probe: Probe) -> int | None:
    """Composite effect is the modular sum of observed edge effects A→B→C→D."""
    edge_effects: list[int] = []
    seen: set[tuple[int, int]] = set()
    for row in _by_channel(rows, "event"):
        body = row["body"]
        if len(body) < 3:
            continue
        edge = (int(body[0]), int(body[1]))
        if edge in seen:
            continue
        seen.add(edge)
        edge_effects.append(int(body[2]))
    if not edge_effects and len(probe.probe) >= 3:
        edge_effects = [int(x) for x in probe.probe[2:]]
    if not edge_effects:
        return None
    composite = 0
    for effect in edge_effects:
        composite = (composite + effect) % ANSWER_ALPHABET
    return composite


def _solve_intervention_versus_observation(rows: Sequence[Mapping[str, Any]], probe: Probe) -> int | None:
    """Interventional outcome is on intervene_effect, not the observational association."""
    effects = _by_channel(rows, "intervene_effect")
    if not effects:
        return None
    body = effects[-1]["body"]
    if len(body) < 2:
        return None
    return int(body[1])


def _solve_novel_sensor_mapping(rows: Sequence[Mapping[str, Any]], probe: Probe) -> int | None:
    """New code is an offset map of the world state; invert held_new → state → label."""
    if len(probe.probe) < 1:
        return None
    held_new = int(probe.probe[0])
    offset: int | None = int(probe.probe[1]) if len(probe.probe) >= 2 else None
    if offset is None:
        intros = _by_channel(rows, "mapping_intro")
        if not intros or not intros[0]["body"]:
            return None
        offset = int(intros[0]["body"][0])
    labels: dict[int, int] = {}
    for row in _by_channel(rows, "appearance"):
        body = row["body"]
        if len(body) >= 2:
            labels[int(body[0])] = int(body[1])
    for state, label in labels.items():
        new_code = ((state + offset) % _PUBLIC_MOD) + 1
        if new_code == held_new:
            return label
    # Fall back: learned pairs on sensor channel (state, new_code) for the first states.
    state_for_new: dict[int, int] = {}
    for row in _by_channel(rows, "sensor"):
        body = row["body"]
        if len(body) >= 2 and int(body[1]) != 77_777:
            state_for_new[int(body[1])] = int(body[0])
    state = state_for_new.get(held_new)
    if state is not None and state in labels:
        return labels[state]
    return None


def _solve_tool_acquisition(rows: Sequence[Mapping[str, Any]], probe: Probe) -> int | None:
    """Linear tool map y = (slope * x + intercept) mod alphabet; fit from tool_use or probe."""
    slope: int | None = None
    intercept: int | None = None
    held: int | None = None
    tool: int | None = None
    if len(probe.probe) >= 4:
        tool, held, slope, intercept = (int(probe.probe[0]), int(probe.probe[1]), int(probe.probe[2]), int(probe.probe[3]))
    elif len(probe.probe) >= 2:
        tool, held = int(probe.probe[0]), int(probe.probe[1])
    pairs: list[tuple[int, int]] = []
    for row in _by_channel(rows, "tool_use"):
        body = row["body"]
        if len(body) >= 3 and (tool is None or int(body[0]) == tool):
            pairs.append((int(body[1]), int(body[2])))
    if slope is None or intercept is None:
        if len(pairs) < 2:
            return None
        # Solve slope from distinct inputs: (y1 - y0) * inv(x1 - x0) mod alphabet.
        x0, y0 = pairs[0]
        x1, y1 = pairs[1]
        dx = (x1 - x0) % ANSWER_ALPHABET
        dy = (y1 - y0) % ANSWER_ALPHABET
        if dx == 0:
            return None
        inv = pow(dx, -1, ANSWER_ALPHABET)
        slope = (dy * inv) % ANSWER_ALPHABET
        intercept = (y0 - slope * x0) % ANSWER_ALPHABET
    if held is None:
        return None
    return (int(slope) * held + int(intercept)) % ANSWER_ALPHABET


def _solve_new_modality_integration(rows: Sequence[Mapping[str, Any]], probe: Probe) -> int | None:
    """Held entity: vision colour and haptic weight combine as (c + w) mod alphabet."""
    if not probe.probe:
        return None
    held = int(probe.probe[0])
    color: int | None = None
    weight: int | None = None
    for row in _by_channel(rows, "vision"):
        body = row["body"]
        if len(body) >= 2 and int(body[0]) == held:
            color = int(body[1])
    for row in _by_channel(rows, "haptic"):
        body = row["body"]
        if len(body) >= 2 and int(body[0]) == held:
            weight = int(body[1])
    if color is None or weight is None:
        return None
    return (color + weight) % ANSWER_ALPHABET


def _solve_teaching_sequence_following(rows: Sequence[Mapping[str, Any]], probe: Probe) -> int | None:
    """Fold over instruction symbols ordered by the taught step field (structure, not arrival)."""
    symbols: list[int] = []
    for row in _by_channel(rows, "appearance"):
        body = row["body"]
        if body:
            symbols.append(int(body[0]))
    if not symbols and len(probe.probe) >= 5:
        # probe: (start, s0, s1, s2, s3, *order)
        symbols = [int(x) for x in probe.probe[1:5]]
    starts = _by_channel(rows, "start_state")
    if starts and starts[0]["body"]:
        start = int(starts[0]["body"][0])
    elif probe.probe:
        start = int(probe.probe[0])
    else:
        return None
    instructions = _by_channel(rows, "instruction")
    if instructions:
        # Sort by step so a shuffled delivery still recovers the teaching sequence.
        stepped = [(int(row["body"][0]), int(row["body"][1])) for row in instructions if len(row["body"]) >= 2]
        stepped.sort(key=lambda item: item[0])
        ordered_symbols = [symbol for _, symbol in stepped]
    elif len(probe.probe) >= 9:
        order = [int(x) for x in probe.probe[5:9]]
        ordered_symbols = [symbols[position] for position in order]
    else:
        return None
    # Generator fold is modular sum over (symbol % alphabet + symbol_position); commutative
    # in the summands, but we still apply it in taught step order for clarity.
    acc = start
    for symbol in ordered_symbols:
        try:
            position = symbols.index(symbol)
        except ValueError:
            position = 0
        acc = (acc + (symbol % ANSWER_ALPHABET) + position) % ANSWER_ALPHABET
    return acc


def _solve_task_composition_transfer(rows: Sequence[Mapping[str, Any]], probe: Probe) -> int | None:
    """compose_rule (1, 2) means g∘f; slopes from appearance/probe or fitted task tables."""
    held: int | None = int(probe.probe[0]) if probe.probe else None
    f_slope: int | None = int(probe.probe[1]) if len(probe.probe) >= 3 else None
    g_slope: int | None = int(probe.probe[2]) if len(probe.probe) >= 3 else None
    if f_slope is None or g_slope is None:
        appearances = _by_channel(rows, "appearance")
        if appearances and len(appearances[0]["body"]) >= 2:
            f_slope = int(appearances[0]["body"][0])
            g_slope = int(appearances[0]["body"][1])
    if held is None:
        transfers = _by_channel(rows, "transfer_context")
        if transfers and transfers[0]["body"]:
            held = int(transfers[0]["body"][0])
    if held is None or f_slope is None or g_slope is None:
        # Fit from task_f / task_g tables if slopes missing.
        f_table = {int(r["body"][0]): int(r["body"][1]) for r in _by_channel(rows, "task_f") if len(r["body"]) >= 2}
        g_table = {int(r["body"][0]): int(r["body"][1]) for r in _by_channel(rows, "task_g") if len(r["body"]) >= 2}
        if held is not None and held in f_table:
            mid = f_table[held]
            if mid in g_table:
                return g_table[mid]
        # Held may be missing from tables; try to recover slopes from two points.
        if f_slope is None and len(f_table) >= 2:
            items = list(f_table.items())
            x0, y0 = items[0]
            x1, y1 = items[1]
            dx = (x1 - x0) % ANSWER_ALPHABET
            if dx:
                f_slope = ((y1 - y0) * pow(dx, -1, ANSWER_ALPHABET)) % ANSWER_ALPHABET
        if g_slope is None and len(g_table) >= 2:
            items = list(g_table.items())
            x0, y0 = items[0]
            x1, y1 = items[1]
            dx = (x1 - x0) % ANSWER_ALPHABET
            if dx:
                g_slope = ((y1 - y0) * pow(dx, -1, ANSWER_ALPHABET)) % ANSWER_ALPHABET
    if held is None or f_slope is None or g_slope is None:
        return None
    return (g_slope * ((f_slope * held) % ANSWER_ALPHABET)) % ANSWER_ALPHABET


def _solve_exception_after_rule(rows: Sequence[Mapping[str, Any]], probe: Probe) -> int | None:
    """Probe subject carries the exception marker; exception rows teach the exception label."""
    marker = int(probe.probe[2]) if len(probe.probe) >= 3 else None
    if marker is None:
        markers = _by_channel(rows, "exception_marker")
        if markers and markers[0]["body"]:
            marker = int(markers[0]["body"][0])
    for row in _by_channel(rows, "exception"):
        body = row["body"]
        if len(body) >= 4 and (marker is None or int(body[2]) == marker):
            return int(body[3])
    return None


def _solve_contradiction_reopening(rows: Sequence[Mapping[str, Any]], probe: Probe) -> int | None:
    """Verified contradictions replace the consolidated belief; answer is the new value."""
    claim = int(probe.probe[1]) if len(probe.probe) >= 2 else None
    if claim is None:
        related_rows = _by_channel(rows, "related")
        if related_rows and len(related_rows[0]["body"]) >= 2:
            claim = int(related_rows[0]["body"][1])
    for row in reversed(_by_channel(rows, "verify")):
        body = row["body"]
        if len(body) >= 2 and (claim is None or int(body[0]) == claim):
            return int(body[1])
    for row in reversed(_by_channel(rows, "contradict")):
        body = row["body"]
        if len(body) >= 2 and (claim is None or int(body[0]) == claim):
            return int(body[1])
    return None


def _solve_long_horizon_goal_recovery(rows: Sequence[Mapping[str, Any]], probe: Probe) -> int | None:
    """Next plan step after the interrupt point — only recoverable if that step was shown."""
    goal = int(probe.probe[0]) if probe.probe else None
    resume_at = int(probe.probe[1]) if len(probe.probe) >= 2 else None
    if resume_at is None:
        for row in _by_channel(rows, "resume"):
            body = row["body"]
            if body and (goal is None or int(body[0]) == goal):
                resume_at = int(body[1]) if len(body) >= 2 else 0
    if resume_at is None:
        return None
    next_index = resume_at + 1
    for row in _by_channel(rows, "goal_step"):
        body = row["body"]
        if len(body) >= 3 and (goal is None or int(body[0]) == goal) and int(body[1]) == next_index:
            return int(body[2])
    # Intentionally do not invent a step value: steps[2] is never in the stream.
    return None


def _solve_resource_envelope_shift(rows: Sequence[Mapping[str, Any]], probe: Probe) -> int | None:
    """Under the new cap, survivors are top-priority items; answer is the top survivor's priority."""
    priorities: dict[int, int] = {}
    for row in _by_channel(rows, "appearance"):
        body = row["body"]
        if len(body) >= 2:
            priorities[int(body[0])] = int(body[1])
    if not priorities and len(probe.probe) >= 8:
        # probe: (held, new_cap, *items, *priorities)
        items = [int(x) for x in probe.probe[2:8]]
        pris = [int(x) for x in probe.probe[8:14]] if len(probe.probe) >= 14 else []
        if len(items) == len(pris):
            priorities = dict(zip(items, pris, strict=True))
    shifts = _by_channel(rows, "envelope_shift")
    new_cap = int(shifts[-1]["body"][0]) if shifts and shifts[-1]["body"] else None
    if new_cap is None and len(probe.probe) >= 2:
        new_cap = int(probe.probe[1])
    if not priorities or new_cap is None:
        # query_context carries survivors after the held/new_cap prefix.
        for row in _by_channel(rows, "query_context"):
            body = row["body"]
            if len(body) >= 4:
                survivors = [int(x) for x in body[2:] if int(x) != 77_777]
                if survivors and survivors[0] in priorities:
                    return priorities[survivors[0]]
        return None
    ranked = sorted(priorities.keys(), key=lambda item: (-priorities[item], item))
    survivors = ranked[: max(0, int(new_cap))]
    if not survivors:
        return None
    return priorities[survivors[0]]


def _solve_migration_continuity(rows: Sequence[Mapping[str, Any]], probe: Probe) -> int | None:
    """Post-migration token is an affine re-encoding; invert held_new → token → value."""
    if not probe.probe:
        return None
    held_new = int(probe.probe[0])
    migrate_a = int(probe.probe[1]) if len(probe.probe) >= 3 else None
    migrate_b = int(probe.probe[2]) if len(probe.probe) >= 3 else None
    if migrate_a is None or migrate_b is None:
        migrations = _by_channel(rows, "migrate")
        if migrations and len(migrations[0]["body"]) >= 2:
            migrate_a = int(migrations[0]["body"][0])
            migrate_b = int(migrations[0]["body"][1])
    values: dict[int, int] = {}
    for row in _by_channel(rows, "appearance"):
        body = row["body"]
        if len(body) >= 2:
            values[int(body[0])] = int(body[1])
    if migrate_a is not None and migrate_b is not None:
        for token, value in values.items():
            new_token = ((migrate_a * token + migrate_b) % _PUBLIC_MOD) + 1
            if new_token == held_new:
                return value
    # Partial reencode pairs as a fallback for the taught subset.
    for row in _by_channel(rows, "reencode"):
        body = row["body"]
        if len(body) >= 2 and int(body[1]) == held_new and int(body[0]) in values:
            return values[int(body[0])]
    return None


_SOLVERS: dict[str, Any] = {
    "unseen_concept_acquisition": _solve_unseen_concept_acquisition,
    "category_boundary_revision": _solve_category_boundary_revision,
    "causal_system_induction": _solve_causal_system_induction,
    "intervention_versus_observation": _solve_intervention_versus_observation,
    "novel_sensor_mapping": _solve_novel_sensor_mapping,
    "tool_acquisition": _solve_tool_acquisition,
    "new_modality_integration": _solve_new_modality_integration,
    "teaching_sequence_following": _solve_teaching_sequence_following,
    "task_composition_transfer": _solve_task_composition_transfer,
    "exception_after_rule": _solve_exception_after_rule,
    "contradiction_reopening": _solve_contradiction_reopening,
    "long_horizon_goal_recovery": _solve_long_horizon_goal_recovery,
    "resource_envelope_shift": _solve_resource_envelope_shift,
    "migration_continuity": _solve_migration_continuity,
}


# Precise diagnoses for families the observation stream cannot determine.
_UNSOLVABLE_DIAGNOSES: dict[str, str] = {
    "long_horizon_goal_recovery": (
        "The probe asks for the next goal step after an interrupt at step index 1 "
        "(answers = steps[2]), but the generator only emits goal_step rows for indices 0 and 1 "
        "(genesis_challenge.py ~716-736). steps[2] and steps[3] are drawn as sealed answer codes "
        "and never enter any observation payload. A competent observer can know that the next "
        "index is 2, but cannot recover the held-out step value from the stream. "
        "Note: _answer_code(rng, i+3) ignores the RNG when parts are supplied, so steps[2] is "
        "accidentally the constant 6 — that is a generator defect, not a recoverable observation."
    ),
}


_REPAIR_SKETCHES: dict[str, str] = {
    "long_horizon_goal_recovery": (
        "Smallest repair: before the interrupt, emit goal_step observations for the full plan "
        "(goal, step_index, step_value) for indices 0..3 on channel 'goal_step' (teaching=True). "
        "Keep the probe channel as 'goal_next' with payload (goal, resume_index) so a record store "
        "keyed by probe.channel still cannot copy the answer field. The reference path becomes: "
        "resume_index from resume/interrupt → answer = value of goal_step at resume_index+1. "
        "Also pass entropy into step codes (e.g. include unit_id in _answer_code parts) so the "
        "answer is not a global constant."
    ),
}


# --------------------------------------------------------------------------
# Material
# --------------------------------------------------------------------------


@dataclass
class ReferenceLearner(MaterialBase):
    """Per-family reference implementation of the intended observation-stream solution.

    Active state holds the observation buffer only. Durable state is empty and
    never written by observe/answer. No sealed object is stored or consulted.
    """

    buffer: list[dict[str, Any]] = field(default_factory=list)

    def _transition(self, observation: Observation) -> None:
        self.buffer.append(
            {
                "index": int(observation.index),
                "channel": str(observation.channel),
                "body": _body(observation),
                "payload": tuple(int(x) for x in observation.payload),
                "teaching": bool(observation.teaching),
                "modality": str(observation.modality),
            }
        )
        # Active-only: resize is allowed; durable digest must stay fixed (empty).
        self._opportunity.ledger.resize(self._active_state_bytes())

    def _active_state_bytes(self) -> int:
        return 8 + 16 * len(self.buffer)

    def _answer(self, probe: Probe) -> Answer:
        arity = max(0, int(probe.arity))
        solver = _SOLVERS.get(probe.family)
        if solver is None:
            return Answer(probe_index=probe.index, value=tuple(0 for _ in range(arity)) if arity else (), confidence=0, abstained=True)
        value = solver(self.buffer, probe)
        if value is None:
            return Answer(
                probe_index=probe.index,
                value=tuple(0 for _ in range(arity)) if arity else (),
                confidence=0,
                abstained=True,
            )
        return Answer(probe_index=probe.index, value=(int(value),) if arity != 0 else (int(value),), confidence=255, abstained=False)

    def _propose(self) -> Iterable[Proposal]:
        return ()

    def _commit(self, proposal: Proposal) -> None:
        return None

    def _rollback(self, receipt: Receipt) -> None:
        return None

    def _durable_state(self) -> Any:
        return {"form": "reference_learner", "durable": "none", "activation": _ACTIVATION}

    def _active_state(self) -> Any:
        return {
            "buffer": [
                {
                    "index": row["index"],
                    "channel": row["channel"],
                    "body": list(row["body"]),
                    "payload": list(row["payload"]),
                    "teaching": row["teaching"],
                    "modality": row["modality"],
                }
                for row in self.buffer
            ],
            "activation": _ACTIVATION,
        }

    def _restore_durable(self, state: Any) -> None:
        return None

    def _restore_active(self, state: Any) -> None:
        self.buffer = [
            {
                "index": int(row["index"]),
                "channel": str(row["channel"]),
                "body": tuple(int(x) for x in row["body"]),
                "payload": tuple(int(x) for x in row["payload"]),
                "teaching": bool(row["teaching"]),
                "modality": str(row["modality"]),
            }
            for row in state.get("buffer", [])
        ]


def _build_reference(opportunity: Opportunity, **_options: Any) -> ReferenceLearner:
    return ReferenceLearner(
        name="reference_learner",
        mechanism="per_family_observation_stream_reference_solution",
        _opportunity=opportunity,
    )


register("reference_learner", _build_reference)


# --------------------------------------------------------------------------
# Solvability report
# --------------------------------------------------------------------------


def _opportunity_for(observations: Sequence[Observation]) -> Opportunity:
    channels = tuple(sorted({observation.channel for observation in observations})) or ("symbolic",)
    return equal_opportunity(
        envelope="512MB",
        observations=observations,
        sensor_channels=channels,
        operation_budget=100_000,
        durable_write_budget=64,
    )


def run_reference_on_unit(unit: Any) -> float:
    """Score the reference learner on one unit's public surface only."""
    from substrate.genesis_material import build

    public = unit.public()
    material = build("reference_learner", _opportunity_for(public.observations))
    for observation in public.observations:
        material.observe(observation)
    answers = [material.answer(probe) for probe in public.probes]
    return float(unit.sealed.score(answers))


def solvability_report(
    *,
    units_per_family: int = 200,
    seed_namespace: str = "genesis-reference-solvability",
    split: str = "principal",
) -> dict[str, Any]:
    """Score the reference learner per family and classify solvability.

    ``solvable`` is True only when mean score is more than ``CHANCE_Z_CEILING``
    standard errors above chance under a Bernoulli(chance) null.
    """
    if units_per_family < 1:
        raise ValueError("units_per_family must be positive")
    standard_error = math.sqrt(CHANCE_LEVEL * (1.0 - CHANCE_LEVEL) / units_per_family)
    families: dict[str, Any] = {}
    solvable_count = 0
    for family in C.CHALLENGE_FAMILIES:
        scores = [run_reference_on_unit(generate(family, split, unit_id, seed_namespace=seed_namespace)) for unit_id in range(units_per_family)]
        mean_score = sum(scores) / len(scores)
        z = (mean_score - CHANCE_LEVEL) / standard_error if standard_error > 0 else 0.0
        solvable = z > CHANCE_Z_CEILING
        if solvable:
            solvable_count += 1
        row: dict[str, Any] = {
            "family": family,
            "mean_score": mean_score,
            "chance_level": CHANCE_LEVEL,
            "standard_error": standard_error,
            "z": z,
            "solvable": solvable,
            "units": units_per_family,
            "activation": False,
        }
        if not solvable:
            row["diagnosis"] = _UNSOLVABLE_DIAGNOSES.get(
                family,
                (
                    f"{family}: reference learner mean {mean_score:.4f} is not above chance "
                    f"by z>{CHANCE_Z_CEILING} (z={z:.2f}). The observation stream does not "
                    "determine the sealed answer under the intended solution path."
                ),
            )
            row["repair"] = _REPAIR_SKETCHES.get(
                family,
                (
                    f"Inspect genesis_challenge._gen_{family}: ensure every quantity the probe "
                    "asks for appears as a non-copyable derived fact in observations "
                    "(binding, fold, or rule application), never as a labelled field on the "
                    "probe channel itself."
                ),
            )
        families[family] = row
    return {
        "units_per_family": units_per_family,
        "seed_namespace": seed_namespace,
        "split": split,
        "chance_level": CHANCE_LEVEL,
        "chance_z_ceiling": CHANCE_Z_CEILING,
        "n_families": len(C.CHALLENGE_FAMILIES),
        "n_solvable": solvable_count,
        "n_unsolvable": len(C.CHALLENGE_FAMILIES) - solvable_count,
        "families": families,
        "order_dependent_families": list(ORDER_DEPENDENT_FAMILIES),
        "activation": False,
    }


__all__ = [
    "CHANCE_Z_CEILING",
    "ORDER_DEPENDENT_FAMILIES",
    "ReferenceLearner",
    "run_reference_on_unit",
    "solvability_report",
]
