
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.studies import generation1_c3_d1_frozen_queue as d1
from mop.studies import generation1_consolidated_final_campaign as consolidated
from mop.studies import generation1_successor_horizon as horizon
from mop.studies import generation1_successor_mechanics_queue as mechanics

VERIFICATION_SCHEMA = "mop-generation1-successor-horizon-verification/v1"
EXPECTED_D1_VARIANT = {
    "variant_id": "centroid-h64-e60-lr03",
    "feature_set": "centroid",
    "hidden": 64,
    "epochs": 60,
    "lr": 0.003,
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _sealed(value: Mapping[str, Any], field: str) -> bool:
    return value.get(field) == canonical_sha256({key: item for key, item in value.items() if key != field})


def _repo_path(value: Any) -> Path:
    if not isinstance(value, str):
        raise ValueError("artifact path is not a string")
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError("artifact path is not repository-relative")
    resolved = (REPO_ROOT / raw).resolve()
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("artifact path escapes the repository")
    return resolved


class Moments:
    __slots__ = ("count", "mean", "m2")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def lo(self) -> float:
        if self.count < 2:
            return self.mean
        variance = self.m2 / (self.count - 1)
        return self.mean - 1.96 * math.sqrt(variance) / math.sqrt(self.count)


class PhaseGateAccumulator:
    __slots__ = ("favorable", "margins", "seed_count")

    def __init__(self) -> None:
        self.margins = {
            name: Moments()
            for name in (
                "global_static",
                "difficulty_static",
                "context_route_nonpromotable",
            )
        }
        self.favorable = {"global_static": 0, "difficulty_static": 0}
        self.seed_count = 0

    def add(self, row: Mapping[str, Any]) -> None:
        by_seed: dict[int, dict[str, list[float]]] = {}
        cells = row.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError("fresh D1 receipt has no cells")
        for cell in cells:
            learned = float(cell["variant_accuracy"][d1.FROZEN_VARIANT_ID])
            seed = int(cell["seed"])
            seed_row = by_seed.setdefault(
                seed,
                {"global_static": [], "difficulty_static": []},
            )
            for control, moments in self.margins.items():
                margin = learned - float(cell["control_accuracy"][control])
                moments.add(margin)
                if control in seed_row:
                    seed_row[control].append(margin)
        self.seed_count += len(by_seed)
        for seed_row in by_seed.values():
            for control in self.favorable:
                self.favorable[control] += int(math.fsum(seed_row[control]) / len(seed_row[control]) > 0.0)

    def passed(self) -> bool:
        if not self.seed_count:
            raise ValueError("fresh D1 phase is empty")
        static_gate = all(
            self.margins[name].mean >= float(d1.CRITERIA["minimum_mean_advantage_over_each_static_control"])
            and self.margins[name].lo > float(d1.CRITERIA["comparison_interval_lower_bound_must_exceed"])
            for name in ("global_static", "difficulty_static")
        )
        favorable_gate = all(
            count / self.seed_count >= float(d1.CRITERIA["minimum_favorable_seed_fraction"])
            for count in self.favorable.values()
        )
        context_gap = -self.margins["context_route_nonpromotable"].mean
        return bool(
            static_gate
            and favorable_gate
            and context_gap <= float(d1.CRITERIA["maximum_mean_gap_below_fixed_c2_context_route"])
            and float(d1.CRITERIA["minimum_work_saving_vs_all_five_actors"]) <= 0.8
        )


def _validate_result_shell(value: Mapping[str, Any]) -> None:
    if not _sealed(value, "result_sha256"):
        raise ValueError("horizon result seal drifted")
    if (
        value.get("schema") != horizon.RESULT_SCHEMA
        or value.get("program_id") != horizon.PROGRAM_ID
        or value.get("claim_scope") != horizon.CLAIM_SCOPE
        or value.get("grid", {}).get("epoch_count") != len(horizon.EPOCH_IDS)
        or value.get("decision", {}).get("independent_scientific_confirmation") is not False
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
    ):
        raise ValueError("horizon result semantic boundary drifted")


def _validate_admission(value: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    if not _sealed(value, "admission_sha256"):
        raise ValueError("horizon admission seal drifted")
    binding = result.get("admission") or {}
    if (
        value.get("schema") != horizon.ADMISSION_SCHEMA
        or value.get("program_id") != horizon.PROGRAM_ID
        or value.get("claim_scope") != horizon.CLAIM_SCOPE
        or value.get("epoch_ids") != list(horizon.EPOCH_IDS)
        or value.get("fresh_cycle_indices") != list(horizon.EPOCH_CYCLES)
        or value.get("admission_sha256") != binding.get("admission_sha256")
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("independent_scientific_confirmation") is not False
    ):
        raise ValueError("horizon admission semantic boundary drifted")
    final_binding = value.get("consolidated_authority") or {}
    final_path = _repo_path(final_binding.get("path"))
    final_value = _read_object(final_path)
    if (
        sha256_file(final_path) != final_binding.get("file_sha256")
        or not _sealed(final_value, "result_sha256")
        or final_value.get("result_sha256") != final_binding.get("result_sha256")
        or final_value.get("schema") != consolidated.RESULT_SCHEMA
    ):
        raise ValueError("horizon consolidated authority drifted")
    pruned = final_value.get("authorities", {}).get("successor_mechanics", {}).get("pruned_lanes")
    known_lanes = [lane.lane_id for lane in mechanics.LANES]
    fresh_cycles = final_value.get("fresh_d1_cycles")
    d1_eligible = bool(
        final_value.get("authorities", {}).get("d1", {}).get("frozen_pattern_repeated") is True
        and isinstance(fresh_cycles, dict)
        and fresh_cycles
        and all(
            isinstance(cycle, dict)
            and cycle.get("producer", {}).get("all_frozen_criteria_passed") is True
            and cycle.get("challenge", {}).get("all_frozen_criteria_passed") is True
            for cycle in fresh_cycles.values()
        )
    )
    if (
        not isinstance(pruned, list)
        or any(not isinstance(item, str) for item in pruned)
        or not set(pruned) <= set(known_lanes)
        or value.get("d1_initially_eligible") is not d1_eligible
        or value.get("mechanics_source_pruned_lanes") != sorted(pruned)
        or value.get("mechanics_initially_eligible_lanes")
        != [lane for lane in known_lanes if lane not in pruned]
    ):
        raise ValueError("horizon admission routing does not match consolidated authority")


def _validate_shard(value: Mapping[str, Any], epoch_id: str, lane: str, shard_index: int) -> None:
    if not _sealed(value, "shard_sha256"):
        raise ValueError("horizon shard seal drifted")
    if (
        value.get("schema") != horizon.SHARD_SCHEMA
        or value.get("program_id") != horizon.PROGRAM_ID
        or value.get("epoch_id") != epoch_id
        or value.get("lane") != lane
        or value.get("shard_index") != shard_index
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
        or value.get("independent_scientific_confirmation") is not False
    ):
        raise ValueError("horizon shard identity or safety drifted")


def _expected_d1_config(source_index: int, cycle: int) -> dict[str, int]:
    phase = "producer" if source_index < d1.RUNGS_PER_PHASE else "challenge"
    local = source_index if phase == "producer" else source_index - d1.RUNGS_PER_PHASE
    offset = 0 if phase == "producer" else 2_000_000
    cycle_base = 500_000_001 + cycle * consolidated.D1_CYCLE_STRIDE
    return {
        "train_seed_start": cycle_base + offset + local * d1.SEEDS_PER_RUNG,
        "heldout_seed_start": cycle_base + offset + 1_000_000 + local * d1.SEEDS_PER_RUNG,
        "router_training_seed": (
            700_000_001 + cycle * consolidated.D1_CYCLE_STRIDE + offset + local * 100_003
        ),
    }


def _validate_d1_receipt(value: Mapping[str, Any], source_index: int, cycle: int) -> None:
    if not _sealed(value, "result_sha256"):
        raise ValueError("fresh D1 receipt seal drifted")
    config = value.get("config") or {}
    expected = _expected_d1_config(source_index, cycle)
    expected_visible = [
        "latent_vector_when_variant_enabled",
        "difficulty_index",
        "labeled_training_support_geometry_when_variant_enabled",
    ]
    expected_forbidden = [
        "context_id",
        "truth",
        "actor_predictions",
        "oracle_actor_id",
        "heldout_route_labels",
    ]
    expected_config_fields = {
        "schema",
        "campaign_id",
        "claim_scope",
        "execution_class",
        "train_seed_start",
        "train_seed_count",
        "heldout_seed_start",
        "heldout_seed_count",
        "difficulty_indices",
        "dataset",
        "actor_training",
        "router_training_seed",
        "variants",
        "visible_inputs",
        "forbidden_heldout_inputs",
        "activation_allowed",
        "scientific_promotion",
    }
    if (
        value.get("schema") != "mop-generation1-c3-router-redesign-rung/v1"
        or value.get("campaign_id") != "generation1-c3-d1-visible-router-redesign-v1"
        or value.get("claim_scope") != "generated-visible-router-redesign-exploration-only"
        or config.get("schema") != "mop-generation1-c3-router-redesign-config/v1"
        or config.get("campaign_id") != "generation1-c3-d1-visible-router-redesign-v1"
        or config.get("claim_scope") != "generated-visible-router-redesign-exploration-only"
        or set(config) != expected_config_fields
        or config.get("execution_class") != "paired_nonpromotable_router_redesign_screen"
        or config.get("train_seed_start") != expected["train_seed_start"]
        or config.get("heldout_seed_start") != expected["heldout_seed_start"]
        or config.get("router_training_seed") != expected["router_training_seed"]
        or config.get("train_seed_count") != d1.SEEDS_PER_RUNG
        or config.get("heldout_seed_count") != d1.SEEDS_PER_RUNG
        or config.get("difficulty_indices") != [0, 1, 2, 3, 4]
        or config.get("dataset") != {"n_train": 720, "n_test": 240, "n_classes": 10, "dim": 64}
        or config.get("actor_training") != {"epochs": 6, "torch_threads": 1}
        or config.get("variants") != [EXPECTED_D1_VARIANT]
        or config.get("visible_inputs") != expected_visible
        or config.get("forbidden_heldout_inputs") != expected_forbidden
        or config.get("activation_allowed") is not False
        or config.get("scientific_promotion") is not False
        or value.get("config_sha256") != canonical_sha256(config)
        or value.get("decision")
        != {
            "best_exploratory_variant": d1.FROZEN_VARIANT_ID,
            "redesign_screen_rung_complete": True,
            "ready_for_confirmatory_claim": False,
            "independent_verification_required": True,
        }
        or value.get("heldout_contract")
        != {
            "visible_inputs": expected_visible,
            "forbidden_inputs": expected_forbidden,
            "contract_honored": True,
        }
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
    ):
        raise ValueError("fresh D1 receipt identity or seed boundary drifted")
    grid = value.get("grid") or {}
    cells = value.get("cells")
    expected_cell_count = d1.SEEDS_PER_RUNG * 5
    if (
        grid
        != {
            "train_seed_count": d1.SEEDS_PER_RUNG,
            "heldout_seed_count": d1.SEEDS_PER_RUNG,
            "difficulty_count": 5,
            "variant_count": 1,
            "completed_cell_count": expected_cell_count,
            "shared_actor_evaluation": True,
        }
        or not isinstance(cells, list)
        or len(cells) != expected_cell_count
    ):
        raise ValueError("fresh D1 receipt grid drifted")
    expected_controls = {
        "global_static",
        "difficulty_static",
        "random_actor",
        "context_route_nonpromotable",
        "oracle_nonpromotable",
    }
    observed_pairs: set[tuple[int, int]] = set()
    heldout_start = int(expected["heldout_seed_start"])
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("fresh D1 receipt cell is invalid")
        seed = int(cell.get("seed"))
        difficulty = int(cell.get("difficulty_index"))
        pair = (seed, difficulty)
        variant_accuracy = cell.get("variant_accuracy") or {}
        control_accuracy = cell.get("control_accuracy") or {}
        selected = (cell.get("selected_actor_counts") or {}).get(d1.FROZEN_VARIANT_ID)
        accuracies = [*variant_accuracy.values(), *control_accuracy.values()]
        dataset_digest = cell.get("dataset_sha256")
        if (
            not heldout_start <= seed < heldout_start + d1.SEEDS_PER_RUNG
            or difficulty not in range(5)
            or pair in observed_pairs
            or cell.get("effective_seed") != seed + difficulty * 1_000_003
            or cell.get("observation_count") != 240
            or cell.get("heldout_sensitive_payloads_emitted") is not False
            or cell.get("router_received_fields") != expected_visible
            or set(variant_accuracy) != {d1.FROZEN_VARIANT_ID}
            or set(control_accuracy) != expected_controls
            or any(
                isinstance(accuracy, bool)
                or not isinstance(accuracy, int | float)
                or not math.isfinite(float(accuracy))
                or not 0.0 <= float(accuracy) <= 1.0
                for accuracy in accuracies
            )
            or not isinstance(selected, dict)
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for count in selected.values()
            )
            or sum(selected.values()) != 240
            or not isinstance(dataset_digest, str)
            or len(dataset_digest) != 64
            or any(character not in "0123456789abcdef" for character in dataset_digest)
        ):
            raise ValueError("fresh D1 receipt cell identity drifted")
        observed_pairs.add(pair)


def _phase_gate(rows: list[Mapping[str, Any]]) -> bool:
    accumulator = PhaseGateAccumulator()
    for row in rows:
        accumulator.add(row)
    return accumulator.passed()


def _classify_d1(
    source_indices: set[int],
    producer: PhaseGateAccumulator,
    challenge: PhaseGateAccumulator,
) -> tuple[str, bool | None, bool | None]:
    if not source_indices:
        return "not_run_pruned", None, None
    if source_indices != set(range(d1.DEFAULT_RUNG_COUNT)):
        raise ValueError("fresh D1 epoch is incomplete")
    producer_passed = producer.passed()
    challenge_passed = challenge.passed()
    if producer_passed and challenge_passed:
        classification = "stable_candidate_trace"
    elif producer_passed != challenge_passed:
        classification = "mixed_or_seed_sensitive"
    else:
        classification = "stable_null"
    return classification, producer_passed, challenge_passed


def _validate_mechanics_receipt(value: Mapping[str, Any], source_index: int, cycle: int) -> None:
    if not _sealed(value, "result_sha256"):
        raise ValueError("fresh mechanics receipt seal drifted")
    source = mechanics.WORK_ITEMS[source_index]
    item = value.get("item") or {}
    expected_start = (
        source.seed_start + consolidated.MECHANICS_FRESH_BASE + cycle * consolidated.MECHANICS_CYCLE_STRIDE
    )
    if (
        value.get("schema") != mechanics.RUNG_SCHEMA
        or item.get("index") != source.index
        or item.get("lane_id") != source.lane_id
        or item.get("mechanism") != source.mechanism
        or item.get("rung_index") != source.rung_index
        or item.get("seed_start") != expected_start
        or item.get("seed_count") != source.seed_count
        or value.get("receipt_count") != source.seed_count
        or value.get("confirmation_count") != 0
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
    ):
        raise ValueError("fresh mechanics receipt identity or seed boundary drifted")
    verdict_counts = value.get("verdict_counts")
    control_counts = value.get("control_clear_counts")
    digest = value.get("receipt_digest_fold")
    if (
        not isinstance(verdict_counts, dict)
        or not verdict_counts
        or any(
            not isinstance(verdict, str)
            or not verdict
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for verdict, count in verdict_counts.items()
        )
        or sum(verdict_counts.values()) != source.seed_count
        or not isinstance(control_counts, dict)
        or any(
            not isinstance(control, str)
            or not control
            or isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= source.seed_count
            for control, count in control_counts.items()
        )
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("fresh mechanics receipt count or digest boundary drifted")


def _classification_rows(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    bindings = result.get("classifications") or []
    if not isinstance(bindings, list) or len(bindings) != len(horizon.EPOCH_IDS):
        raise ValueError("horizon classification binding inventory is incomplete")
    for expected_index, binding in enumerate(bindings):
        if not isinstance(binding, dict):
            raise ValueError("horizon classification binding is invalid")
        path = _repo_path(binding.get("path"))
        value = _read_object(path)
        if (
            sha256_file(path) != binding.get("file_sha256")
            or not _sealed(value, "classification_sha256")
            or value.get("classification_sha256") != binding.get("classification_sha256")
            or value.get("schema") != horizon.CLASSIFICATION_SCHEMA
            or value.get("program_id") != horizon.PROGRAM_ID
            or value.get("claim_scope") != horizon.CLAIM_SCOPE
            or value.get("epoch_id") != horizon.EPOCH_IDS[expected_index]
            or value.get("epoch_index") != expected_index
            or value.get("cycle_index") != horizon.EPOCH_CYCLES[expected_index]
            or value.get("complete") is not True
            or value.get("problems") != []
            or value.get("activation_allowed") is not False
            or value.get("scientific_promotion") is not False
            or value.get("independent_scientific_confirmation") is not False
        ):
            raise ValueError("horizon classification binding drifted")
        epoch_id = str(value["epoch_id"])
        if epoch_id in rows:
            raise ValueError("horizon classification binding is duplicated")
        rows[epoch_id] = value
    if set(rows) != set(horizon.EPOCH_IDS):
        raise ValueError("horizon classification inventory is incomplete")
    parent = None
    for epoch_id in horizon.EPOCH_IDS:
        if rows[epoch_id].get("parent_classification_sha256") != parent:
            raise ValueError("horizon classification ancestry drifted")
        parent = rows[epoch_id]["classification_sha256"]
    return rows


def _intervals_disjoint(intervals: list[tuple[int, int]]) -> bool:
    ordered = sorted(intervals)
    return all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:], strict=False))


def _recompute(
    result: Mapping[str, Any],
    admission: Mapping[str, Any],
    classifications: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    d1_sources_by_epoch: dict[str, set[int]] = {epoch: set() for epoch in horizon.EPOCH_IDS}
    d1_phases_by_epoch = {
        epoch: {
            "producer": PhaseGateAccumulator(),
            "challenge": PhaseGateAccumulator(),
        }
        for epoch in horizon.EPOCH_IDS
    }
    mechanics_sources_by_epoch: dict[str, set[int]] = {epoch: set() for epoch in horizon.EPOCH_IDS}
    mechanics_verdicts_by_epoch: dict[str, dict[str, dict[str, int]]] = {
        epoch: {} for epoch in horizon.EPOCH_IDS
    }
    d1_intervals: list[tuple[int, int]] = []
    mechanics_intervals: list[tuple[int, int]] = []
    expected_shards = {
        (epoch, lane, shard)
        for epoch in horizon.EPOCH_IDS
        for lane, count in (
            ("d1", horizon.D1_SHARD_COUNT),
            ("mechanics", horizon.MECHANICS_SHARD_COUNT),
        )
        for shard in range(count)
    }
    bindings = result.get("shard_index") or []
    if not isinstance(bindings, list) or len(bindings) != len(expected_shards):
        raise ValueError("horizon shard binding inventory is incomplete")
    seen_shards: set[tuple[str, str, int]] = set()
    shard_bindings: dict[tuple[str, str, int], tuple[str, str, str]] = {}
    executed_counts = {"d1": 0, "mechanics": 0}
    initial_mechanics = admission.get("mechanics_initially_eligible_lanes")
    if not isinstance(initial_mechanics, list) or any(
        not isinstance(item, str) for item in initial_mechanics
    ):
        raise ValueError("horizon mechanics admission routing is invalid")
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("horizon shard binding is invalid")
        epoch_id = str(binding.get("epoch_id"))
        lane = str(binding.get("lane"))
        shard_index = int(binding.get("shard_index"))
        identity = (epoch_id, lane, shard_index)
        if identity not in expected_shards or identity in seen_shards:
            raise ValueError("horizon shard identity is missing or duplicated")
        path = _repo_path(binding.get("path"))
        shard = _read_object(path)
        if sha256_file(path) != binding.get("file_sha256"):
            raise ValueError("horizon shard file binding drifted")
        _validate_shard(shard, epoch_id, lane, shard_index)
        if shard.get("shard_sha256") != binding.get("shard_sha256"):
            raise ValueError("horizon shard payload binding drifted")
        epoch_index = horizon.EPOCH_IDS.index(epoch_id)
        cycle = horizon.EPOCH_CYCLES[epoch_index]
        previous = classifications[horizon.EPOCH_IDS[epoch_index - 1]] if epoch_index else None
        d1_enabled = admission.get("d1_initially_eligible") is True and (
            previous is None or previous.get("routing", {}).get("continue_d1") is True
        )
        mechanics_enabled = set(
            initial_mechanics
            if previous is None
            else previous.get("routing", {}).get("mechanics_lanes_for_next_epoch") or []
        )
        partition = (
            horizon.D1_PARTITIONS[shard_index] if lane == "d1" else horizon.MECHANICS_PARTITIONS[shard_index]
        )
        expected_sources = {
            source_index
            for source_index in partition
            if (lane == "d1" and d1_enabled)
            or (lane == "mechanics" and mechanics.WORK_ITEMS[source_index].lane_id in mechanics_enabled)
        }
        artifacts = shard.get("artifact_index") or []
        if not isinstance(artifacts, list):
            raise ValueError("horizon raw artifact inventory is invalid")
        seen_sources: set[int] = set()
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ValueError("horizon raw artifact binding is invalid")
            artifact_path = _repo_path(artifact.get("path"))
            if sha256_file(artifact_path) != artifact.get("file_sha256"):
                raise ValueError("horizon raw artifact file binding drifted")
            value = _read_object(artifact_path)
            source_index = int(artifact["source_index"])
            if source_index not in expected_sources or source_index in seen_sources:
                raise ValueError("horizon raw artifact source is missing or duplicated")
            if value.get("result_sha256") != artifact.get("result_sha256"):
                raise ValueError("horizon raw artifact result binding drifted")
            if lane == "d1":
                _validate_d1_receipt(value, source_index, cycle)
                if source_index in d1_sources_by_epoch[epoch_id]:
                    raise ValueError("fresh D1 source is duplicated across shards")
                d1_sources_by_epoch[epoch_id].add(source_index)
                phase = "producer" if source_index < d1.RUNGS_PER_PHASE else "challenge"
                d1_phases_by_epoch[epoch_id][phase].add(value)
                config = value["config"]
                d1_intervals.extend(
                    [
                        (
                            int(config["train_seed_start"]),
                            int(config["train_seed_start"]) + int(config["train_seed_count"]),
                        ),
                        (
                            int(config["heldout_seed_start"]),
                            int(config["heldout_seed_start"]) + int(config["heldout_seed_count"]),
                        ),
                    ]
                )
            elif lane == "mechanics":
                _validate_mechanics_receipt(value, source_index, cycle)
                if source_index in mechanics_sources_by_epoch[epoch_id]:
                    raise ValueError("fresh mechanics source is duplicated across shards")
                mechanics_sources_by_epoch[epoch_id].add(source_index)
                lane_id = mechanics.WORK_ITEMS[source_index].lane_id
                counts = mechanics_verdicts_by_epoch[epoch_id].setdefault(lane_id, {})
                for verdict, count in value["verdict_counts"].items():
                    counts[verdict] = counts.get(verdict, 0) + int(count)
                item = value["item"]
                mechanics_intervals.append(
                    (int(item["seed_start"]), int(item["seed_start"]) + int(item["seed_count"]))
                )
            else:
                raise ValueError("horizon shard declares an unknown lane")
            seen_sources.add(source_index)
        if seen_sources != expected_sources:
            raise ValueError("horizon raw artifact inventory is incomplete")
        if (
            shard.get("planned_item_count") != len(partition)
            or shard.get("executed_item_count") != len(expected_sources)
            or shard.get("skipped_item_count") != len(partition) - len(expected_sources)
        ):
            raise ValueError("horizon shard execution counts drifted")
        executed_counts[lane] += len(expected_sources)
        seen_shards.add(identity)
        shard_bindings[identity] = (
            str(binding.get("path")),
            str(binding.get("file_sha256")),
            str(binding.get("shard_sha256")),
        )

    for epoch_id in horizon.EPOCH_IDS:
        declared_bindings = classifications[epoch_id].get("shard_bindings") or []
        expected_epoch = {identity for identity in expected_shards if identity[0] == epoch_id}
        if not isinstance(declared_bindings, list) or len(declared_bindings) != len(expected_epoch):
            raise ValueError("classification shard inventory is incomplete")
        seen_declared: set[tuple[str, str, int]] = set()
        for binding in declared_bindings:
            if not isinstance(binding, dict):
                raise ValueError("classification shard binding is invalid")
            identity = (epoch_id, str(binding.get("lane")), int(binding.get("shard_index")))
            if identity not in expected_epoch or identity in seen_declared:
                raise ValueError("classification shard identity is missing or duplicated")
            expected_binding = shard_bindings[identity]
            observed_binding = (
                str(binding.get("path")),
                str(binding.get("file_sha256")),
                str(binding.get("shard_sha256")),
            )
            if observed_binding != expected_binding:
                raise ValueError("classification shard binding differs from result inventory")
            seen_declared.add(identity)

    d1_classes: dict[str, str] = {}
    retained_by_epoch: dict[str, list[str]] = {}
    for epoch_id in horizon.EPOCH_IDS:
        d1_class, producer_passed, challenge_passed = _classify_d1(
            d1_sources_by_epoch[epoch_id],
            d1_phases_by_epoch[epoch_id]["producer"],
            d1_phases_by_epoch[epoch_id]["challenge"],
        )
        d1_classes[epoch_id] = d1_class
        d1_declared = classifications[epoch_id]["d1"]
        d1_routes = {
            "stable_candidate_trace": ("positive", True),
            "stable_null": ("null", False),
            "mixed_or_seed_sensitive": ("mixed_or_seed_sensitive", True),
            "not_run_pruned": ("blocked", False),
        }
        route, continuation = d1_routes[d1_class]
        if (
            d1_class != d1_declared.get("classification")
            or d1_declared.get("terminal_route") != route
            or d1_declared.get("continue_d1") is not continuation
            or classifications[epoch_id].get("routing", {}).get("continue_d1") is not continuation
        ):
            raise ValueError("independent D1 classification differs")
        if d1_class == "not_run_pruned":
            if d1_declared.get("producer") is not None or d1_declared.get("challenge") is not None:
                raise ValueError("pruned D1 classification contains phase evidence")
        elif (
            d1_declared.get("producer", {}).get("all_frozen_criteria_passed") is not producer_passed
            or d1_declared.get("challenge", {}).get("all_frozen_criteria_passed") is not challenge_passed
        ):
            raise ValueError("independent D1 phase gates differ")
        lane_verdicts = mechanics_verdicts_by_epoch[epoch_id]
        epoch_index = horizon.EPOCH_IDS.index(epoch_id)
        previous = classifications[horizon.EPOCH_IDS[epoch_index - 1]] if epoch_index else None
        eligible = (
            initial_mechanics if previous is None else previous["routing"]["mechanics_lanes_for_next_epoch"]
        )
        retained = []
        declared_lanes = classifications[epoch_id].get("mechanics") or {}
        for lane_spec in mechanics.LANES:
            lane_id = lane_spec.lane_id
            counts = lane_verdicts.get(lane_id, {})
            total = sum(counts.values())
            clean = lane_id in eligible and counts == {"mechanics-ok": total} and total > 0
            if clean:
                retained.append(lane_id)
            expected_classification = (
                "mechanics_noninferential"
                if clean
                else "mechanics_warning"
                if lane_id in eligible
                else "not_run_pruned"
            )
            expected_route = "descriptive_only" if clean else "null" if lane_id in eligible else "blocked"
            row = declared_lanes.get(lane_id) or {}
            if (
                row.get("classification") != expected_classification
                or row.get("terminal_route") != expected_route
                or row.get("receipt_count") != total
                or row.get("verdict_counts") != counts
                or row.get("continue_lane") is not clean
            ):
                raise ValueError("independent mechanics lane classification differs")
        declared = classifications[epoch_id]["routing"]["mechanics_lanes_for_next_epoch"]
        if retained != declared:
            raise ValueError("independent mechanics routing differs")
        retained_by_epoch[epoch_id] = retained

    if not _intervals_disjoint(d1_intervals) or not _intervals_disjoint(mechanics_intervals):
        raise ValueError("horizon seed spaces overlap")
    if (
        d1_intervals
        and mechanics_intervals
        and max(end for _, end in d1_intervals) >= min(start for start, _ in mechanics_intervals)
    ):
        raise ValueError("D1 and mechanics seed spaces overlap")
    return {
        "d1_classifications": d1_classes,
        "mechanics_lanes_retained": retained_by_epoch,
        "d1_interval_count": len(d1_intervals),
        "mechanics_interval_count": len(mechanics_intervals),
        "all_seed_intervals_disjoint": True,
        "bound_shard_count": len(seen_shards),
        "executed_d1_rung_count": executed_counts["d1"],
        "executed_mechanics_rung_count": executed_counts["mechanics"],
    }


def _mutation_suite(result: Mapping[str, Any]) -> dict[str, Any]:
    mutations = (
        {"schema": "bad-schema"},
        {"program_id": "bad-program"},
        {"claim_scope": "widened"},
        {"activation_allowed": True},
        {"scientific_promotion": True},
        {"complete": False},
        {"problems": ["hidden"]},
        {"grid": {**result["grid"], "epoch_count": 999}},
        {
            "decision": {
                **result["decision"],
                "independent_scientific_confirmation": True,
            }
        },
    )
    rejected = 0
    for mutation in mutations:
        core = {name: item for name, item in result.items() if name != "result_sha256"}
        core.update(mutation)
        candidate = {**core, "result_sha256": canonical_sha256(core)}
        try:
            _validate_result_shell(candidate)
        except ValueError:
            rejected += 1
    return {
        "count": len(mutations),
        "rejected": rejected,
        "all_rejected": rejected == len(mutations),
    }


def build_verification(result_path: Path = horizon.DEFAULT_RESULT) -> dict[str, Any]:
    result_path = Path(result_path).resolve()
    result = _read_object(result_path)
    _validate_result_shell(result)
    admission_binding = result.get("admission") or {}
    admission_path = _repo_path(admission_binding.get("path"))
    if sha256_file(admission_path) != admission_binding.get("file_sha256"):
        raise ValueError("horizon admission file binding drifted")
    admission = _read_object(admission_path)
    _validate_admission(admission, result)
    classifications = _classification_rows(result)
    recomputation = _recompute(result, admission, classifications)
    grid = result.get("grid") or {}
    if (
        grid.get("d1_shard_count") != len(horizon.EPOCH_IDS) * horizon.D1_SHARD_COUNT
        or grid.get("mechanics_shard_count") != len(horizon.EPOCH_IDS) * horizon.MECHANICS_SHARD_COUNT
        or grid.get("executed_d1_rung_count") != recomputation["executed_d1_rung_count"]
        or grid.get("executed_mechanics_rung_count") != recomputation["executed_mechanics_rung_count"]
    ):
        raise ValueError("horizon result grid differs from independent inventory")
    mutations = _mutation_suite(result)
    checks = {
        "result_seal_valid": True,
        "admission_and_consolidated_authority_valid": True,
        "all_shards_and_raw_artifacts_valid": True,
        "classifications_independently_reproduced": True,
        "all_seed_intervals_disjoint": recomputation["all_seed_intervals_disjoint"],
        "mutation_suite_passed": mutations["all_rejected"],
        "independent_generator_family_present": False,
    }
    core = {
        "schema": VERIFICATION_SCHEMA,
        "program_id": horizon.PROGRAM_ID,
        "claim_scope": (
            "independent receipt, seed-boundary, aggregation, classification, and mutation verification; "
            "no second generator family"
        ),
        "source": {
            "path": str(result_path.relative_to(REPO_ROOT.resolve())),
            "file_sha256": sha256_file(result_path),
            "result_sha256": result["result_sha256"],
        },
        "checks": checks,
        "recomputation": recomputation,
        "mutation_suite": mutations,
        "verification_complete": all(
            value is True for key, value in checks.items() if key != "independent_generator_family_present"
        ),
        "independent_scientific_confirmation": False,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "verification_sha256": canonical_sha256(core)}


def validate_verification(value: Mapping[str, Any]) -> None:
    if (
        not _sealed(value, "verification_sha256")
        or value.get("schema") != VERIFICATION_SCHEMA
        or value.get("program_id") != horizon.PROGRAM_ID
        or value.get("verification_complete") is not True
        or value.get("independent_scientific_confirmation") is not False
        or value.get("checks", {}).get("independent_generator_family_present") is not False
        or value.get("complete") is not True
        or value.get("problems") != []
        or value.get("activation_allowed") is not False
        or value.get("scientific_promotion") is not False
    ):
        raise ValueError("successor horizon verification identity or safety drifted")
    source = value.get("source") or {}
    path = _repo_path(source.get("path"))
    result = _read_object(path)
    _validate_result_shell(result)
    if source.get("file_sha256") != sha256_file(path) or source.get("result_sha256") != result.get(
        "result_sha256"
    ):
        raise ValueError("successor horizon verification source binding drifted")


def verify(
    *, result_path: Path = horizon.DEFAULT_RESULT, output: Path = horizon.DEFAULT_VERIFICATION
) -> dict[str, Any]:
    value = build_verification(result_path)
    validate_verification(value)
    consolidated.atomic_write_json(Path(output).resolve(), value)
    return value


__all__ = [
    "VERIFICATION_SCHEMA",
    "build_verification",
    "canonical_bytes",
    "canonical_sha256",
    "validate_verification",
    "verify",
]
