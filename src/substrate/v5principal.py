"""Frozen, resumable developmental DAG for the Substrate v5 campaign."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import os
import statistics
import time
from typing import Any

from substrate import v5config as C
from substrate import v5experiment as E
from substrate import v5io as io
from substrate import v5state

SHARDS = 4
PHASES_PER_SHARD = 5
SPLIT_SEEDS = {
    "principal": tuple(range(5_000, 5_048)),
    "replication": tuple(range(6_000, 6_016)),
    "open_world_review": tuple(range(7_000, 7_016)),
}


class Refused(RuntimeError):
    """A principal unit, checkpoint, or launch violated the frozen authority."""


@dataclasses.dataclass(frozen=True)
class WorkUnit:
    split: str
    history_seed: int
    arm: str
    shard: int

    def __post_init__(self) -> None:
        if self.split not in SPLIT_SEEDS:
            raise Refused(f"unknown split {self.split!r}")
        if self.history_seed not in SPLIT_SEEDS[self.split]:
            raise Refused("history seed outside frozen split")
        if self.arm not in C.ARMS:
            raise Refused(f"unknown arm {self.arm!r}")
        if not 0 <= self.shard < SHARDS:
            raise Refused("shard outside frozen DAG")

    @property
    def identity(self) -> str:
        return (
            f"{self.split}-{self.history_seed}-{self.arm}-"
            f"shard{self.shard:02d}"
        )

    @property
    def phase_indices(self) -> tuple[int, ...]:
        start = self.shard * PHASES_PER_SHARD
        return tuple(range(start, start + PHASES_PER_SHARD))

    @property
    def dependency(self) -> str | None:
        if self.shard == 0:
            return None
        return dataclasses.replace(self, shard=self.shard - 1).identity

    @property
    def event_count(self) -> int:
        return len(self.phase_indices) * E.EPISODES_PER_PHASE

    def document(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "hypotheses": list(C.HYPOTHESES),
            "arm": self.arm,
            "history_seed": self.history_seed,
            "split": self.split,
            "phase_indices": list(self.phase_indices),
            "phases": [C.PHASES[index] for index in self.phase_indices],
            "modalities": sorted(
                {
                    modality
                    for index in self.phase_indices
                    for modality in E.PHASE_MODALITIES[index]
                }
            ),
            "models": "registered model-equivalent modules selected by the v5 fabric",
            "body": "desktop_body or seeded_3d_body",
            "inputs": [E.generator_manifest()["generator_digest"]],
            "outputs": [
                f"units/{self.identity}.json",
                f"checkpoints/{self.identity}.json",
            ],
            "dependencies": [self.dependency] if self.dependency else [],
            "resource_class": "cpu_small",
            "worker_class": "deterministic_developmental_history",
            "native_thread_budget": 1,
            "accelerator_requirement": "none",
            "timeout_seconds": 120,
            "retry": "one deterministic retry; preserve both failure receipts",
            "checkpoint": f"checkpoints/{self.identity}.json",
            "artifact_family": self.split,
            "claim_ceiling": "multimodal_nous_ready_for_review",
            "event_count": self.event_count,
            "activation": False,
        }


def work_units(split: str | None = None) -> list[WorkUnit]:
    splits = (split,) if split else tuple(SPLIT_SEEDS)
    return [
        WorkUnit(name, seed, arm, shard)
        for name in splits
        for seed in SPLIT_SEEDS[name]
        for arm in C.ARMS
        for shard in range(SHARDS)
    ]


def _initial_state(unit: WorkUnit) -> dict[str, Any]:
    identity = E.history_identity(unit.split, unit.history_seed, unit.arm)
    return {
        "entity_identity": identity,
        "birth_identity": identity,
        "completed_phase": -1,
        "developmental_events": 0,
        "semantic_memories": 0,
        "procedural_memories": 0,
        "tracked_objects": 0,
        "unfinished_goals": ["return-to-scene"],
        "model_identity": "vision-temporal-alpha",
        "model_replacements": 0,
        "body_identity": "desktop-body",
        "body_changes": 0,
        "sensor_interruptions": 0,
        "restorations": 0,
        "development_state": {},
        "depth_state_digest": io.sha_obj({}),
        "three_d_state_digest": io.sha_obj({}),
        "body_state_digest": io.sha_obj({}),
        "model_checkpoint_identity": "builtin:vision-temporal-alpha",
        "model_family": "vision_temporal",
        "sensor_environment": "uninitialized",
        "body_variant": "body:none",
        "executed_model_families": [],
        "sensor_environments": [],
        "body_variants": [],
        "diversity_records_complete": False,
        "activation": False,
    }


def _state_model_contract(
    identity: str,
    checkpoint_identity: str,
) -> v5state.ModelContract:
    return v5state.ModelContract(
        identity=identity,
        checkpoint_identity=checkpoint_identity,
        version="v5.0.0",
        license="project-local-deterministic-fixture",
        runtime="python-deterministic",
        hardware_requirements=("cpu",),
        modalities_accepted=("image", "video"),
        modalities_produced=("proposal",),
        training_provenance=("hand-specified", "no-training-data"),
        known_limitations=("bounded synthetic operations only",),
        allowed_roles=(
            "independent_performer",
            "specialist",
            "draft_generator",
        ),
        statefulness="replaceable",
        checkpoint_support=True,
    )


def _new_entity(unit: WorkUnit, entity_identity: str) -> v5state.PermanentEntity:
    entity = v5state.PermanentEntity(entity_identity)
    entity.upsert_goal(
        "goal:return-to-scene",
        "retain the scene and return after interruption",
        provenance=("frozen-v5-curriculum",),
    )
    principal_modalities = sorted(
        {
            modality
            for phase_modalities in E.PHASE_MODALITIES
            for modality in phase_modalities
        }
    )
    for modality in principal_modalities:
        entity.attach_sensor(
            f"sensor:{modality}",
            {
                "modality": modality,
                "coordinate_frame": "world",
                "replaceable": True,
                "activation": False,
            },
        )
    entity.replace_body(
        {
            "identity": "desktop-browser-body-v5",
            "sensors": principal_modalities,
            "actuators": ["inspect", "wait"],
            "coordinate_frames": ["world", "desktop_pixels"],
            "capabilities": ["sandbox_observation"],
            "activation": False,
        }
    )
    entity.register_model(
        _state_model_contract(
            "vision-temporal-alpha",
            "builtin:vision-temporal-alpha",
        )
    )
    return entity


def _restore_entity(
    unit: WorkUnit,
    predecessor: dict[str, Any] | None,
) -> v5state.PermanentEntity:
    if predecessor is None:
        return _new_entity(
            unit,
            E.history_identity(unit.split, unit.history_seed, unit.arm),
        )
    if unit.arm == "fresh_reset":
        return _new_entity(
            unit,
            E.history_identity(
                unit.split,
                unit.history_seed + unit.shard * 100_000,
                unit.arm,
            ),
        )
    checkpoint = predecessor.get("entity_checkpoint")
    if not isinstance(checkpoint, dict):
        raise Refused("predecessor omits the permanent-entity checkpoint")
    try:
        return v5state.PermanentEntity.restore(checkpoint)
    except v5state.Refused as error:
        raise Refused(f"permanent-entity restore failed: {error}") from error


def _project_phase(
    entity: v5state.PermanentEntity,
    unit: WorkUnit,
    row: dict[str, Any],
) -> None:
    phase_index = int(row["phase_index"])
    executed = row["executed"]
    for modality in row["modalities"]:
        sensor_identity = f"sensor:{modality}"
        entity.observe_sensor(
            sensor_identity,
            {
                "phase_index": phase_index,
                "event_digest": row["event_digest"],
                "environment": executed["sensor_environment"],
                "model_identities": executed["model_identities"],
                "commitment_count": row["decisions"]["commitments"],
                "activation": False,
            },
            source_timestamp=phase_index * E.EPISODES_PER_PHASE,
            temporal_uncertainty=float(row["mean_uncertainty"]),
        )
    entity.record_memory(
        "episodic",
        f"phase:{phase_index:02d}",
        {
            "phase": row["phase"],
            "accuracy": row["accuracy"],
            "utility": row["utility"],
            "modalities": row["modalities"],
            "event_digest": row["event_digest"],
            "environment": executed["sensor_environment"],
            "body_variant": executed["body_variant"],
            "model_families": executed["model_families"],
            "activation": False,
        },
        provenance=(f"principal:{unit.identity}", row["event_digest"]),
    )
    entity.update_world(
        "tracked_objects",
        f"track:{unit.history_seed}:persistent",
        {
            "last_phase": phase_index,
            "scene_identity": row["integrity"]["scene_identity"],
            "clip_identity": row["integrity"]["clip_identity"],
            "visible": True,
            "activation": False,
        },
    )
    if {"depth", "three_d"} & set(row["modalities"]):
        entity.update_world(
            "spatial_world",
            f"scene:{unit.history_seed}",
            {
                "scene_identity": row["integrity"]["scene_identity"],
                "environment_checkpoint_digest": executed[
                    "environment_checkpoint_digest"
                ],
                "coordinate_frame": "world",
                "activation": False,
            },
        )
    if phase_index == 13:
        entity.interrupt_sensor("sensor:video")
        entity.observe_sensor(
            "sensor:video",
            {
                "phase_index": phase_index,
                "restored": True,
                "activation": False,
            },
            source_timestamp=phase_index * E.EPISODES_PER_PHASE + 1,
        )
    if phase_index == 14:
        entity.replace_model(
            "vision-temporal-alpha",
            _state_model_contract(
                "vision-temporal-beta",
                "builtin:vision-temporal-beta",
            ),
            measured=True,
            evidence=(row["event_digest"],),
        )
    if phase_index == 15:
        entity.replace_body(
            {
                "identity": "simulator-3d-body-v5",
                "sensors": sorted(
                    {
                        modality
                        for phase_modalities in E.PHASE_MODALITIES
                        for modality in phase_modalities
                    }
                ),
                "actuators": [
                    "inspect",
                    "request_depth",
                    "rotate_view",
                    "wait",
                ],
                "coordinate_frames": ["world", "body", "camera"],
                "capabilities": [
                    "sandbox_observation",
                    "depth_request",
                    "viewpoint_change",
                ],
                "activation": False,
            }
        )


def _checkpoint_body(
    unit: WorkUnit,
    predecessor: dict[str, Any] | None,
    phases: list[dict[str, Any]],
) -> dict[str, Any]:
    entity = _restore_entity(unit, predecessor)
    if predecessor is None:
        state = _initial_state(unit)
    else:
        state = dict(predecessor["state"])
        if (
            unit.arm != "fresh_reset"
            and state.get("entity_identity")
            != E.history_identity(unit.split, unit.history_seed, unit.arm)
        ):
            raise Refused("predecessor changed continuing entity identity")
        if int(state.get("completed_phase", -1)) != unit.phase_indices[0] - 1:
            raise Refused("predecessor phase does not match DAG dependency")
    if unit.arm == "fresh_reset" and unit.shard > 0:
        state = _initial_state(unit)
        state["entity_identity"] = E.history_identity(
            unit.split,
            unit.history_seed + unit.shard * 100_000,
            unit.arm,
        )
    for row in phases:
        _project_phase(entity, unit, row)
    state["completed_phase"] = unit.phase_indices[-1]
    state["developmental_events"] = int(state["developmental_events"]) + sum(
        int(row["episodes"]) for row in phases
    )
    state["semantic_memories"] = int(state["semantic_memories"]) + sum(
        int(row["accuracy"] * row["episodes"]) for row in phases
    )
    state["procedural_memories"] = int(state["procedural_memories"]) + sum(
        bool(row["mechanisms_active"]) for row in phases
    )
    state["tracked_objects"] = max(
        int(state["tracked_objects"]),
        3 + sum("video" in row["modalities"] for row in phases),
    )
    if 13 in unit.phase_indices:
        state["sensor_interruptions"] = int(state["sensor_interruptions"]) + 1
        state["restorations"] = int(state["restorations"]) + 1
    if 14 in unit.phase_indices:
        state["model_identity"] = "vision-temporal-beta"
        state["model_replacements"] = int(state["model_replacements"]) + 1
    if 15 in unit.phase_indices:
        state["body_identity"] = "seeded-3d-body"
        state["body_changes"] = int(state["body_changes"]) + 1
    state["development_state"] = dict(phases[-1]["development_update"])
    entity_checkpoint = entity.checkpoint()
    entity_state = entity_checkpoint["state"]
    state["depth_state_digest"] = io.sha_obj(
        {
            key: value
            for key, value in entity_state["sensory_buffers"].items()
            if "depth" in key or "three_d" in key
        }
    )
    state["three_d_state_digest"] = io.sha_obj(
        entity_state["spatial_world"]
    )
    state["body_state_digest"] = io.sha_obj(entity_state["body_state"])
    state["model_checkpoint_identity"] = (
        "builtin:vision-temporal-beta"
        if state["model_replacements"]
        else "builtin:vision-temporal-alpha"
    )
    state["model_family"] = "vision_temporal"
    state["sensor_environment"] = phases[-1]["executed"][
        "sensor_environment"
    ]
    state["body_variant"] = entity_state["body_state"]["identity"]
    prior_families = set(state.get("executed_model_families", []))
    prior_environments = set(state.get("sensor_environments", []))
    prior_bodies = set(state.get("body_variants", []))
    state["executed_model_families"] = sorted(
        prior_families
        | {
            str(family)
            for row in phases
            for family in row["executed"]["model_families"]
        }
    )
    state["sensor_environments"] = sorted(
        prior_environments
        | {
            str(row["executed"]["sensor_environment"])
            for row in phases
        }
    )
    state["body_variants"] = sorted(
        prior_bodies
        | {
            str(row["executed"]["body_variant"])
            for row in phases
        }
        | {str(entity_state["body_state"]["identity"])}
    )
    state["diversity_records_complete"] = bool(
        state["executed_model_families"]
        and state["sensor_environments"]
        and state["body_variants"]
    )
    state["permanent_entity_state_digest"] = entity_checkpoint["state_sha256"]
    state_digest = io.sha_obj(state)
    predecessor_digest = io.sha_obj(predecessor) if predecessor else None
    checkpoint = {
        "schema": "substrate-v5-developmental-checkpoint/v1",
        "unit": unit.document(),
        "predecessor_checkpoint": predecessor_digest,
        "state": state,
        "state_digest": state_digest,
        "entity_checkpoint": entity_checkpoint,
        "entity_checkpoint_sha256": entity_checkpoint["sha256"],
        "checkpoint_exact": True,
        "activation": False,
    }
    checkpoint["checkpoint_body_digest"] = io.sha_obj(checkpoint)
    return checkpoint


def execute_unit(
    unit: WorkUnit,
    predecessor: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    development_state = (
        dict(predecessor["state"].get("development_state", {}))
        if predecessor
        else {}
    )
    phases = []
    for index in unit.phase_indices:
        row = E.phase_result(
            split=unit.split,
            history_seed=unit.history_seed,
            arm=unit.arm,
            phase_index=index,
            development_state=development_state,
        )
        phases.append(row)
        development_state = dict(row["development_update"])
    checkpoint = _checkpoint_body(unit, predecessor, phases)
    state = checkpoint["state"]
    receipt = {
        "schema": "substrate-v5-principal-unit/v1",
        "unit": unit.document(),
        "predecessor_checkpoint": checkpoint["predecessor_checkpoint"],
        "phase_results": phases,
        "summary": {
            "mean_accuracy": statistics.fmean(
                float(row["accuracy"]) for row in phases
            ),
            "mean_utility": statistics.fmean(
                float(row["utility"]) for row in phases
            ),
            "mean_cost": statistics.fmean(
                float(row["mean_cost"]) for row in phases
            ),
            "mean_uncertainty": statistics.fmean(
                float(row["mean_uncertainty"]) for row in phases
            ),
            "mechanisms_active": sorted(
                {
                    mechanism
                    for row in phases
                    for mechanism in row["mechanisms_active"]
                }
            ),
            "modalities": sorted(
                {modality for row in phases for modality in row["modalities"]}
            ),
            "events": sum(int(row["episodes"]) for row in phases),
            "entity_identity": state["entity_identity"],
            "birth_identity": state["birth_identity"],
            "model_identity": state["model_identity"],
            "body_identity": state["body_identity"],
            "unfinished_goals": state["unfinished_goals"],
            "state_digest": checkpoint["state_digest"],
            "checkpoint_exact": True,
        },
        "source_generator_digest": E.generator_manifest()["generator_digest"],
        "permanent_entity_checkpoint_sha256": checkpoint[
            "entity_checkpoint_sha256"
        ],
        "activation": False,
    }
    return receipt, checkpoint


def validate(
    receipt: dict[str, Any],
    checkpoint: dict[str, Any],
    unit: WorkUnit,
    predecessor: dict[str, Any] | None = None,
) -> bool:
    checkpoint_body = dict(checkpoint)
    supplied_checkpoint_digest = checkpoint_body.pop(
        "checkpoint_body_digest",
        None,
    )
    entity_checkpoint = checkpoint.get("entity_checkpoint")
    entity_valid = False
    if isinstance(entity_checkpoint, dict):
        try:
            validated_entity = io.validate_normalized_seal(entity_checkpoint)
        except io.Refused:
            validated_entity = {}
        entity_state = validated_entity.get("state")
        entity_valid = bool(
            isinstance(entity_state, dict)
            and validated_entity.get("sha256")
            == checkpoint.get("entity_checkpoint_sha256")
            and validated_entity.get("state_sha256")
            == io.sha_obj(entity_state)
        )
    expected_receipt, expected_checkpoint = execute_unit(unit, predecessor)
    return (
        receipt == expected_receipt
        and checkpoint == expected_checkpoint
        and checkpoint.get("state_digest")
        == io.sha_obj(checkpoint.get("state"))
        and supplied_checkpoint_digest == io.sha_obj(checkpoint_body)
        and checkpoint.get("predecessor_checkpoint")
        == (io.sha_obj(predecessor) if predecessor else None)
        and receipt.get("permanent_entity_checkpoint_sha256")
        == checkpoint.get("entity_checkpoint_sha256")
        and entity_valid
        and receipt.get("activation") is False
        and checkpoint.get("activation") is False
        and all(
            row.get("commitment_precedes_target") is True
            and row.get("raw_observation_excludes_target") is True
            and row.get("decisions", {}).get("outcome_information_used")
            is False
            for row in receipt.get("phase_results", [])
        )
    )


def _relative(unit: WorkUnit, family: str) -> str:
    return f"{unit.split}/{family}/{unit.identity}.json"


def _load_if_valid(
    unit: WorkUnit,
    predecessor: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    receipt_path = io.RUNS / _relative(unit, "units")
    checkpoint_path = io.RUNS / _relative(unit, "checkpoints")
    if not receipt_path.is_file() or not checkpoint_path.is_file():
        return None
    try:
        receipt = dict(io.load_json(receipt_path))
        checkpoint = dict(io.load_json(checkpoint_path))
    except io.Refused:
        return None
    for document in (receipt, checkpoint):
        document.pop("program", None)
        document.pop("sha256", None)
        document.pop("source_commit", None)
        document.pop("source_digest", None)
    return (receipt, checkpoint) if validate(
        receipt,
        checkpoint,
        unit,
        predecessor,
    ) else None


def _load_structurally_valid(
    unit: WorkUnit,
    predecessor: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    receipt_path = io.RUNS / _relative(unit, "units")
    checkpoint_path = io.RUNS / _relative(unit, "checkpoints")
    if not receipt_path.is_file() or not checkpoint_path.is_file():
        return None
    try:
        receipt = dict(io.load_json(receipt_path))
        checkpoint = dict(io.load_json(checkpoint_path))
    except io.Refused:
        return None
    for document in (receipt, checkpoint):
        document.pop("program", None)
        document.pop("sha256", None)
        document.pop("source_commit", None)
        document.pop("source_digest", None)
    checkpoint_body = dict(checkpoint)
    supplied = checkpoint_body.pop("checkpoint_body_digest", None)
    entity_checkpoint = checkpoint.get("entity_checkpoint")
    if not isinstance(entity_checkpoint, dict):
        return None
    try:
        validated_entity = io.validate_normalized_seal(entity_checkpoint)
    except io.Refused:
        return None
    entity_state = validated_entity.get("state")
    checks = (
        receipt.get("unit") == unit.document(),
        checkpoint.get("unit") == unit.document(),
        checkpoint.get("state_digest") == io.sha_obj(checkpoint.get("state")),
        supplied == io.sha_obj(checkpoint_body),
        checkpoint.get("predecessor_checkpoint")
        == (io.sha_obj(predecessor) if predecessor else None),
        receipt.get("permanent_entity_checkpoint_sha256")
        == checkpoint.get("entity_checkpoint_sha256")
        == validated_entity.get("sha256"),
        isinstance(entity_state, dict),
        isinstance(entity_state, dict)
        and validated_entity.get("state_sha256") == io.sha_obj(entity_state),
        receipt.get("activation") is False,
        checkpoint.get("activation") is False,
    )
    return (receipt, checkpoint) if all(checks) else None


def _chain(split: str, history_seed: int, arm: str) -> list[
    tuple[dict[str, Any], dict[str, Any], bool]
]:
    predecessor = None
    rows = []
    for shard in range(SHARDS):
        unit = WorkUnit(split, history_seed, arm, shard)
        existing = _load_if_valid(unit, predecessor)
        if existing is None:
            receipt, checkpoint = execute_unit(unit, predecessor)
            reused = False
        else:
            receipt, checkpoint = existing
            reused = True
        rows.append((receipt, checkpoint, reused))
        predecessor = checkpoint
    return rows


def _worker(arguments: tuple[str, int, str]) -> tuple[
    tuple[str, int, str],
    list[tuple[dict[str, Any], dict[str, Any], bool]],
]:
    return arguments, _chain(*arguments)


def prepare() -> dict[str, Any]:
    units = work_units()
    manifest = {
        "schema": "substrate-v5-principal-dag/v1",
        "units": [unit.document() for unit in units],
        "unit_count": len(units),
        "developmental_histories": sum(len(values) for values in SPLIT_SEEDS.values()),
        "principal_histories": len(SPLIT_SEEDS["principal"]),
        "arms": list(C.ARMS),
        "phases": list(C.PHASES),
        "sensory_events_or_cognitive_episodes": sum(
            unit.event_count for unit in units
        ),
        "generator": E.generator_manifest(),
        "source_commit": io.commit(),
        "source_digest": io.source_digest(),
        "configuration_digest": C.configuration()["configuration_digest"],
        "frozen": True,
        "activation": False,
    }
    resource_plan = {
        "schema": "substrate-v5-resource-plan/v1",
        "worker_candidates": [1, 2, 4, 8, 12, 16],
        "selected_workers": min(8, os.cpu_count() or 1),
        "native_threads_per_worker": 1,
        "central_authoritative_publisher": True,
        "accelerator_required": False,
        "minimum_free_disk_gib": 25,
        "activation": False,
    }
    io.config_json("principal_manifest.json", manifest)
    io.seal("SUBSTRATE_V5_PRINCIPAL_DAG.json", manifest)
    io.seal("SUBSTRATE_V5_RESOURCE_PLAN.json", resource_plan)
    io.seal(
        "SUBSTRATE_V5_WORKER_AUTHORITY.json",
        {
            "schema": "substrate-v5-worker-authority/v1",
            "workers_write_staging_only": True,
            "publisher_validates_and_publishes_atomically": True,
            "duplicate_units_refused": True,
            "worker_candidates": resource_plan["worker_candidates"],
            "selected_workers": resource_plan["selected_workers"],
            "activation": False,
        },
    )
    return {"manifest": manifest, "resource_plan": resource_plan}


def run(
    split: str | None = None,
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    from substrate import v5

    gate = v5.principal_gate()
    if not gate["authorized"]:
        raise Refused(
            f"principal launch gate failed inside executor: {gate['checks']}"
        )
    prepare()
    splits = (split,) if split else tuple(SPLIT_SEEDS)
    if any(name not in SPLIT_SEEDS for name in splits):
        raise Refused("unknown principal split")
    if io.STOP.exists():
        raise Refused("v5 stop switch is present")
    selected_workers = workers or min(8, os.cpu_count() or 1)
    selected_workers = max(1, min(16, int(selected_workers)))
    chains = [
        (name, seed, arm)
        for name in splits
        for seed in SPLIT_SEEDS[name]
        for arm in C.ARMS
    ]
    started = time.perf_counter()
    terminal = 0
    newly_published = 0
    resumed = 0
    failed: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=selected_workers,
    ) as executor:
        pending = iter(chains)
        futures: dict[
            concurrent.futures.Future[
                tuple[
                    tuple[str, int, str],
                    list[tuple[dict[str, Any], dict[str, Any], bool]],
                ]
            ],
            tuple[str, int, str],
        ] = {}

        def submit_next() -> bool:
            if io.STOP.exists():
                return False
            try:
                arguments = next(pending)
            except StopIteration:
                return False
            futures[executor.submit(_worker, arguments)] = arguments
            return True

        for _ in range(selected_workers):
            if not submit_next():
                break
        while futures:
            completed, _ = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in completed:
                arguments = futures.pop(future)
                try:
                    _, results = future.result()
                    predecessor = None
                    for shard, (
                        receipt,
                        checkpoint,
                        reused,
                    ) in enumerate(results):
                        unit = WorkUnit(*arguments, shard)
                        if not validate(
                            receipt,
                            checkpoint,
                            unit,
                            predecessor,
                        ):
                            raise Refused(
                                f"worker returned invalid unit {unit.identity}"
                            )
                        if reused:
                            resumed += 1
                        else:
                            if io.STOP.exists():
                                raise Refused(
                                    "v5 stop switch appeared during publication"
                                )
                            io.run_json(_relative(unit, "units"), receipt)
                            io.run_json(
                                _relative(unit, "checkpoints"),
                                checkpoint,
                            )
                            newly_published += 1
                        predecessor = checkpoint
                        terminal += 1
                except Exception as error:  # noqa: BLE001 - receipt required
                    failed.append(
                        {
                            "chain": list(arguments),
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                submit_next()
        if io.STOP.exists():
            failed.append(
                {
                    "chain": [],
                    "error_type": "Stopped",
                    "error": "v5 stop switch prevented further chain submission",
                }
            )
    elapsed = time.perf_counter() - started
    expected = sum(
        len(SPLIT_SEEDS[name]) * len(C.ARMS) * SHARDS for name in splits
    )
    result = {
        "schema": "substrate-v5-principal-execution/v1",
        "splits": list(splits),
        "expected_units": expected,
        "published_units": terminal,
        "newly_published_units": newly_published,
        "resumed_units": resumed,
        "failed_attempts": failed,
        "all_terminal": terminal == expected and not failed,
        "workers": selected_workers,
        "wall_seconds": elapsed,
        "units_per_second": terminal / elapsed if elapsed else None,
        "sensory_events_or_cognitive_episodes": terminal
        * PHASES_PER_SHARD
        * E.EPISODES_PER_PHASE,
        "activation": False,
    }
    io.seal("SUBSTRATE_V5_PRINCIPAL_AUTHORITY.json", result)
    if failed:
        raise Refused(f"{len(failed)} principal chains failed")
    return result


def status() -> dict[str, Any]:
    expected = work_units()
    valid = 0
    split_counts: dict[str, dict[str, int]] = {}
    for name in SPLIT_SEEDS:
        split_counts[name] = {
            "expected": len(work_units(name)),
            "present": 0,
        }
    for split in SPLIT_SEEDS:
        for seed in SPLIT_SEEDS[split]:
            for arm in C.ARMS:
                predecessor = None
                for shard in range(SHARDS):
                    unit = WorkUnit(split, seed, arm, shard)
                    loaded = _load_structurally_valid(unit, predecessor)
                    if loaded is None:
                        predecessor = None
                        continue
                    _, predecessor = loaded
                    valid += 1
                    split_counts[unit.split]["present"] += 1
    return {
        "schema": "substrate-v5-principal-status/v1",
        "expected": len(expected),
        "present": valid,
        "remaining": len(expected) - valid,
        "splits": split_counts,
        "complete": valid == len(expected),
        "activation": False,
    }
