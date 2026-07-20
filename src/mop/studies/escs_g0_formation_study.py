"""Deterministic persisted mechanics study for inert G0 coalition formation.

This module turns the generic construction, shadow-execution, and formation
records into one small end-to-end study.  It deliberately uses synthetic tasks
and a declared mechanics scorer.  The resulting receipt demonstrates replay,
accounting, admission, refusal, and Pareto mechanics; it is not evidence that a
candidate is intelligent, useful on natural data, or safe to activate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.escs.g0_genotype import (
    G0ActorGenotype,
    G0MessageEdge,
    G0OperatorNode,
    G0StateSlot,
)
from mop.escs.perspective_registry import (
    PerspectiveCandidateRegistry,
    load_perspective_candidate_registry,
)
from mop.escs.topology_grammar import (
    OperatorPrimitive,
    StatePrimitive,
    TopologyGrammar,
    load_topology_grammar,
)
from mop.studies.escs_g0_construction import (
    G0ConstructionAttempt,
    G0ConstructionOperation,
    G0ConstructionRequest,
    G0ConstructionSnapshot,
    G0ConstructionStatus,
    attempt_g0_construction,
)
from mop.studies.escs_g0_formation import (
    G0CandidateBundle,
    G0FormationAttempt,
    G0FormationLedger,
    G0FormationStatus,
    G0ParetoArchive,
    G0ParetoDecisionStatus,
    G0TraceAssessment,
    build_g0_pareto_archive,
    verify_g0_formation_attempt,
)
from mop.studies.escs_g0_shadow_coalition import (
    G0ShadowActorState,
    G0ShadowCaps,
    G0ShadowEpisode,
    G0ShadowPortBinding,
    G0ShadowSeed,
    G0ShadowTerminalReason,
    G0ShadowTrace,
    execute_g0_shadow_coalition,
)
from mop.substrate.events import canonical_sha256

CONFIG_SCHEMA = "mop-escs-g0-formation-study-config/v1"
TASK_AUTHORITY_SCHEMA = "mop-escs-g0-formation-task-authority/v1"
SCORER_SCHEMA = "mop-escs-g0-mechanics-scorer/v1"
RECEIPT_SCHEMA = "mop-escs-g0-formation-study-receipt/v1"
VERIFICATION_SCHEMA = "mop-escs-g0-formation-study-verification/v1"
CLAIM_SCOPE = "deterministic-synthetic-shadow-formation-mechanics-only"

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/experiment/escs_g0_formation_study.json"
DEFAULT_RECEIPT_PATH = REPO_ROOT / "proof/ESCS_G0_FORMATION_STUDY.json"
DEFAULT_VERIFICATION_PATH = REPO_ROOT / "proof/ESCS_G0_FORMATION_STUDY.verification.json"
MAX_INPUT_BYTES = 16 * 1024 * 1024
_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")

_CONFIG_FIELDS = {
    "schema",
    "study_id",
    "claim_scope",
    "authorities",
    "base",
    "mutations",
    "tasks",
    "caps",
    "scorer",
    "nonclaims",
}
_RECEIPT_FIELDS = {
    "schema",
    "study_id",
    "claim_scope",
    "config_authority",
    "implementation_authority",
    "grammar_authority",
    "candidate_registry_authority",
    "scorer_authority_sha256",
    "task_authorities",
    "episodes",
    "traces",
    "ledger",
    "pareto_archive",
    "summary",
    "all_ok",
    "problems",
    "counterfactual_only",
    "activation_enabled",
    "shadow_execution_authorized",
    "factual_effects",
    "factual_mutation_authorized",
    "scientific_promotion_allowed",
    "nonclaims",
    "receipt_sha256",
}
_INERT = {
    "counterfactual_only": True,
    "activation_enabled": False,
    "shadow_execution_authorized": False,
    "factual_effects": False,
    "factual_mutation_authorized": False,
    "scientific_promotion_allowed": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    _require(
        actual == expected,
        f"{label} fields mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}",
    )


def _canonical_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical identifier")
    return value


def _read_regular(path: Path, label: str, *, maximum: int = MAX_INPUT_BYTES) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        _require(0 < before.st_size <= maximum, f"{label} byte envelope is invalid")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    _require(identity_before == identity_after, f"{label} changed while being read")
    raw = b"".join(chunks)
    _require(len(raw) == before.st_size, f"{label} size changed while being read")
    return raw


def _decode_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, label)
    payload = _decode_json(raw, label)
    return payload, raw


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _repo_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a path string")
    path = (REPO_ROOT / value).resolve()
    _require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes the repository")
    _require(path.is_file(), f"{label} does not exist")
    return path


def _file_authority(path: Path) -> dict[str, Any]:
    raw = _read_regular(path, f"authority file {path}")
    resolved = path.resolve()
    label = (
        str(resolved.relative_to(REPO_ROOT.resolve()))
        if resolved.is_relative_to(REPO_ROOT.resolve())
        else str(resolved)
    )
    return {"path": label, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _implementation_authority() -> dict[str, Any]:
    relative_paths = (
        "src/mop/studies/escs_g0_formation_study.py",
        "src/mop/studies/escs_g0_formation.py",
        "src/mop/studies/escs_g0_construction.py",
        "src/mop/studies/escs_g0_shadow_coalition.py",
        "src/mop/escs/g0_evaluator.py",
        "src/mop/escs/g0_genotype.py",
        "src/mop/escs/topology_grammar.py",
        "src/mop/escs/perspective_registry.py",
        "src/mop/escs/accounting.py",
        "src/mop/substrate/events.py",
        "src/mop/config.py",
        "scripts/run_escs_g0_formation_study.py",
    )
    rows = [_file_authority(REPO_ROOT / row) for row in relative_paths]
    return {"files": rows, "manifest_sha256": canonical_sha256(rows)}


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    _require(number == number and abs(number) != float("inf"), f"{label} must be finite")
    return number


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    config, raw = _load_json(path, "G0 formation study config")
    _exact(config, _CONFIG_FIELDS, "G0 formation study config")
    _require(config["schema"] == CONFIG_SCHEMA, "unsupported G0 formation study config")
    _require(config["claim_scope"] == CLAIM_SCOPE, "G0 formation claim scope drifted")
    _canonical_id(config["study_id"], "G0 formation study_id")
    authorities = config["authorities"]
    _require(isinstance(authorities, dict), "G0 formation authorities must be an object")
    _exact(authorities, {"grammar", "candidate_registry"}, "G0 formation authorities")
    _repo_path(authorities["grammar"], "grammar authority")
    _repo_path(authorities["candidate_registry"], "candidate registry authority")

    base = config["base"]
    _require(isinstance(base, dict), "G0 formation base must be an object")
    _exact(
        base,
        {
            "candidate_id",
            "planning_actor_id",
            "verification_actor_id",
            "state_capacity_bytes",
            "planning_weights",
            "verification_weights",
        },
        "G0 formation base",
    )
    _canonical_id(base["candidate_id"], "G0 base candidate_id")
    planning_actor_id = _canonical_id(base["planning_actor_id"], "G0 planning actor_id")
    verification_actor_id = _canonical_id(
        base["verification_actor_id"],
        "G0 verification actor_id",
    )
    _require(
        planning_actor_id != verification_actor_id,
        "G0 planning and verification actor identities must differ",
    )
    _positive_int(base["state_capacity_bytes"], "G0 base state capacity")
    for name in ("planning_weights", "verification_weights"):
        rows = base[name]
        _require(isinstance(rows, list) and len(rows) == 1, f"{name} must be one matrix row")
        _require(isinstance(rows[0], list) and len(rows[0]) == 2, f"{name} width must be two")
        for index, value in enumerate(rows[0]):
            _finite_number(value, f"{name}[0][{index}]")

    mutations = config["mutations"]
    _require(isinstance(mutations, list) and bool(mutations), "G0 mutations must be nonempty")
    mutation_ids: list[str] = []
    candidate_ids = [base["candidate_id"]]
    for row in mutations:
        _require(isinstance(row, dict), "G0 mutation row must be an object")
        _exact(
            row,
            {
                "formation_attempt_id",
                "candidate_id",
                "kind",
                "actor_id",
                "value",
                "declared_work",
                "expected_status",
            },
            "G0 mutation row",
        )
        _require(
            row["kind"] in {"replace-update-weights", "adjust-state-capacity"},
            "unsupported G0 mutation kind",
        )
        mutation_id = _canonical_id(
            row["formation_attempt_id"],
            "G0 mutation formation_attempt_id",
        )
        _require(mutation_id != "formation.base", "G0 mutation shadows the base formation identity")
        candidate_id = _canonical_id(row["candidate_id"], "G0 mutation candidate_id")
        actor_id = _canonical_id(row["actor_id"], "G0 mutation actor_id")
        _require(
            actor_id in {planning_actor_id, verification_actor_id},
            "G0 mutation actor is outside the declared base coalition",
        )
        _require(row["expected_status"] in {"applied", "refused"}, "bad expected status")
        _positive_int(row["declared_work"], "G0 mutation declared work")
        if row["kind"] == "replace-update-weights":
            value = row["value"]
            _require(isinstance(value, list) and len(value) == 1, "mutation weights are invalid")
            _require(isinstance(value[0], list) and len(value[0]) == 2, "mutation width is invalid")
            for index, item in enumerate(value[0]):
                _finite_number(item, f"mutation weight {index}")
        else:
            _positive_int(row["value"], "G0 mutation capacity")
        mutation_ids.append(mutation_id)
        candidate_ids.append(candidate_id)
    _require(len(mutation_ids) == len(set(mutation_ids)), "G0 mutation IDs are duplicated")
    _require(len(candidate_ids) == len(set(candidate_ids)), "G0 candidate IDs are duplicated")

    tasks = config["tasks"]
    _require(isinstance(tasks, list) and bool(tasks), "G0 tasks must be nonempty")
    task_ids: list[str] = []
    for row in tasks:
        _require(isinstance(row, dict), "G0 task must be an object")
        _exact(row, {"task_id", "seed", "target"}, "G0 task")
        _canonical_id(row["task_id"], "G0 task_id")
        for name in ("seed", "target"):
            vector = row[name]
            _require(isinstance(vector, list) and len(vector) == 1, f"task {name} is invalid")
            _finite_number(vector[0], f"task {name}")
        task_ids.append(row["task_id"])
    _require(task_ids == sorted(set(task_ids)), "G0 tasks must be uniquely sorted")

    caps = config["caps"]
    _require(isinstance(caps, dict), "G0 shadow caps must be an object")
    _exact(caps, set(G0ShadowCaps.__dataclass_fields__), "G0 shadow caps")
    G0ShadowCaps(**caps)
    scorer = config["scorer"]
    _require(isinstance(scorer, dict), "G0 scorer must be an object")
    _exact(
        scorer,
        {"schema", "quality_error_scale", "diversity_response_scale", "max_microunits"},
        "G0 scorer",
    )
    _require(scorer["schema"] == SCORER_SCHEMA, "unsupported G0 mechanics scorer")
    for field in ("quality_error_scale", "diversity_response_scale", "max_microunits"):
        _positive_int(scorer[field], f"G0 scorer {field}")
    nonclaims = config["nonclaims"]
    _require(
        isinstance(nonclaims, list)
        and bool(nonclaims)
        and all(isinstance(row, str) and row.strip() for row in nonclaims),
        "G0 nonclaims must be nonempty strings",
    )
    return config, raw


def _recurrent_actor(
    candidate_id: str,
    *,
    state_capacity_bytes: int,
    weights: Sequence[Sequence[float]],
    recipient: str | None,
) -> G0ActorGenotype:
    state = G0StateSlot(
        slot_id="memory",
        primitive=StatePrimitive.BOUNDED_RECURRENT_STATE,
        schema_id="memory-v1",
        capacity_bytes=state_capacity_bytes,
    )
    update = G0OperatorNode.create(
        node_id="update",
        operator=OperatorPrimitive.AFFINE_OR_NONLINEAR_LOCAL_UPDATE,
        state_slot_ids=(state.slot_id,),
        parameters={
            "weights": [list(row) for row in weights],
            "bias": [0.0],
            "activation": "identity",
            "state_mode": "concatenate",
        },
        declared_operations=4,
        max_output_bytes=64,
    )
    if recipient is None:
        return G0ActorGenotype.create(
            candidate_id=candidate_id,
            state_slots=(state,),
            operator_nodes=(update,),
            output_node_ids=(update.node_id,),
        )
    edge = G0MessageEdge(edge_id="out", schema_id="claim-v1", max_encoded_bytes=64)
    emit = G0OperatorNode.create(
        node_id="emit",
        operator=OperatorPrimitive.TYPED_MESSAGE_TRANSFORM_AND_EMIT,
        input_node_ids=(update.node_id,),
        message_edge_ids=(edge.edge_id,),
        parameters={
            "schema_id": edge.schema_id,
            "recipient": f"actor:{recipient}",
            "payload_form": "numeric-vector",
        },
        declared_operations=1,
        max_output_bytes=64,
    )
    return G0ActorGenotype.create(
        candidate_id=candidate_id,
        state_slots=(state,),
        operator_nodes=(update, emit),
        output_node_ids=(emit.node_id,),
        message_edges=(edge,),
    )


def _base_snapshot(base: Mapping[str, Any]) -> G0ConstructionSnapshot:
    planning_id = base["planning_actor_id"]
    verification_id = base["verification_actor_id"]
    planning = _recurrent_actor(
        planning_id,
        state_capacity_bytes=base["state_capacity_bytes"],
        weights=base["planning_weights"],
        recipient=verification_id,
    )
    verification = _recurrent_actor(
        verification_id,
        state_capacity_bytes=base["state_capacity_bytes"],
        weights=base["verification_weights"],
        recipient=None,
    )
    return G0ConstructionSnapshot.create(
        actors=(planning, verification),
        factor_scopes={
            planning_id: ("scope.planning",),
            verification_id: ("scope.verification",),
        },
    )


def _replacement_node(weights: Sequence[Sequence[float]]) -> G0OperatorNode:
    return G0OperatorNode.create(
        node_id="update",
        operator=OperatorPrimitive.AFFINE_OR_NONLINEAR_LOCAL_UPDATE,
        state_slot_ids=("memory",),
        parameters={
            "weights": [list(row) for row in weights],
            "bias": [0.0],
            "activation": "identity",
            "state_mode": "concatenate",
        },
        declared_operations=4,
        max_output_bytes=64,
    )


def _task_authority(task: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema": TASK_AUTHORITY_SCHEMA,
            "task_id": task["task_id"],
            "seed": task["seed"],
            "target": task["target"],
        }
    )


def _make_episode(
    *,
    source: G0ConstructionSnapshot,
    task: Mapping[str, Any],
    caps: G0ShadowCaps,
    grammar: TopologyGrammar,
    candidate_registry: PerspectiveCandidateRegistry,
    planning_actor_id: str,
    verification_actor_id: str,
) -> G0ShadowEpisode:
    ports: list[G0ShadowPortBinding] = []
    states: list[G0ShadowActorState] = []
    for actor in source.actors:
        ports.append(
            G0ShadowPortBinding(
                actor_id=actor.candidate_id,
                genotype_sha256=actor.genotype_sha256,
                root_node_id="update",
                schema_id=("stimulus-v1" if actor.candidate_id == planning_actor_id else "claim-v1"),
                payload_form="numeric-vector",
                max_encoded_bytes=64,
            )
        )
        states.append(
            G0ShadowActorState.create(
                actor.candidate_id,
                {slot.slot_id: [0.0] for slot in actor.state_slots},
            )
        )
    _require(
        {actor.candidate_id for actor in source.actors} == {planning_actor_id, verification_actor_id},
        "G0 study candidate actor identities drifted",
    )
    seed = G0ShadowSeed.create(
        seed_id="root",
        actor_id=planning_actor_id,
        schema_id="stimulus-v1",
        payload_form="numeric-vector",
        payload=task["seed"],
    )
    return G0ShadowEpisode.create(
        episode_id=f"episode.{task['task_id']}",
        source=source,
        grammar=grammar,
        candidate_registry=candidate_registry,
        ports=ports,
        initial_states=states,
        seeds=(seed,),
        caps=caps,
    )


def _score_trace(
    *,
    trace: G0ShadowTrace,
    task: Mapping[str, Any],
    scorer: Mapping[str, Any],
    verification_actor_id: str,
) -> tuple[int, int, int]:
    by_actor = {row.actor_id: row.state.value() for row in trace.final_states}
    state = by_actor.get(verification_actor_id, {})
    _require(isinstance(state, dict), "verification state is not an object")
    actual = state.get("memory")
    target = task["target"]
    _require(isinstance(actual, list) and len(actual) == len(target), "bad verification output")
    error = sum(
        abs(_finite_number(left, "actual output") - _finite_number(right, "target output"))
        for left, right in zip(actual, target, strict=True)
    ) / len(target)
    maximum = scorer["max_microunits"]
    quality = max(0, maximum - round(error * scorer["quality_error_scale"]))
    mechanics_stable = (
        trace.terminal_reason is G0ShadowTerminalReason.QUIESCENT
        and not trace.problems
        and trace.pending_delivery_count == 0
        and trace.rollback_snapshot_sha256 == trace.source_snapshot_sha256
    )
    robustness_proxy = maximum if mechanics_stable else 0
    mean_response = sum(abs(_finite_number(value, "actual output")) for value in actual) / len(actual)
    response_activity_proxy = min(
        maximum,
        round(mean_response * scorer["diversity_response_scale"]),
    )
    return quality, robustness_proxy, response_activity_proxy


def _evaluate_candidate(
    *,
    formation_attempt_id: str,
    candidate: G0CandidateBundle,
    config: Mapping[str, Any],
    grammar: TopologyGrammar,
    candidate_registry: PerspectiveCandidateRegistry,
    scorer_sha256: str,
    construction_request: G0ConstructionRequest | None = None,
    construction_attempt: G0ConstructionAttempt | None = None,
) -> tuple[
    G0FormationAttempt,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, G0ShadowEpisode],
    dict[str, G0ShadowTrace],
]:
    caps = G0ShadowCaps(**config["caps"])
    base = config["base"]
    assessments: list[G0TraceAssessment] = []
    episode_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    episodes: dict[str, G0ShadowEpisode] = {}
    traces: dict[str, G0ShadowTrace] = {}
    for task in config["tasks"]:
        episode = _make_episode(
            source=candidate.snapshot,
            task=task,
            caps=caps,
            grammar=grammar,
            candidate_registry=candidate_registry,
            planning_actor_id=base["planning_actor_id"],
            verification_actor_id=base["verification_actor_id"],
        )
        trace = execute_g0_shadow_coalition(
            candidate.snapshot,
            episode,
            grammar=grammar,
            candidate_registry=candidate_registry,
        )
        quality, robustness, diversity = _score_trace(
            trace=trace,
            task=task,
            scorer=config["scorer"],
            verification_actor_id=base["verification_actor_id"],
        )
        authority = _task_authority(task)
        assessment = G0TraceAssessment.create(
            task_id=task["task_id"],
            task_authority_sha256=authority,
            episode=episode,
            trace=trace,
            scorer_sha256=scorer_sha256,
            quality_microunits=quality,
            robustness_microunits=robustness,
            diversity_microunits=diversity,
        )
        assessments.append(assessment)
        episodes[episode.episode_sha256] = episode
        traces[trace.trace_sha256] = trace
        episode_rows.append(
            {
                "formation_attempt_id": formation_attempt_id,
                "candidate_sha256": candidate.bundle_sha256,
                "task_id": task["task_id"],
                "task_authority_sha256": authority,
                "episode": episode.payload(),
            }
        )
        trace_rows.append(
            {
                "formation_attempt_id": formation_attempt_id,
                "candidate_sha256": candidate.bundle_sha256,
                "task_id": task["task_id"],
                "task_authority_sha256": authority,
                "trace": trace.payload(),
            }
        )
    attempt = G0FormationAttempt.create(
        formation_attempt_id=formation_attempt_id,
        candidate=candidate,
        assessments=assessments,
        construction_request=construction_request,
        construction_attempt=construction_attempt,
    )
    return attempt, episode_rows, trace_rows, episodes, traces


def _mutation_request(row: Mapping[str, Any], source: G0ConstructionSnapshot) -> G0ConstructionRequest:
    if row["kind"] == "replace-update-weights":
        operation = G0ConstructionOperation.REPLACE_OPERATOR
        parameters = {
            "actor_id": row["actor_id"],
            "node": _replacement_node(row["value"]).payload(),
        }
    else:
        operation = G0ConstructionOperation.ADJUST_STATE_CAPACITY
        parameters = {
            "actor_id": row["actor_id"],
            "slot_id": "memory",
            "capacity_bytes": row["value"],
        }
    return G0ConstructionRequest.create(
        attempt_id=row["formation_attempt_id"],
        source_snapshot_sha256=source.snapshot_sha256,
        operation=operation,
        parameters=parameters,
        declared_work=row["declared_work"],
    )


def build_receipt(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:

    config_path = config_path.resolve()
    _require(
        config_path.is_relative_to(REPO_ROOT.resolve()),
        "G0 formation config must remain inside the repository",
    )
    config, config_raw = _load_config(config_path)
    authority_config = config["authorities"]
    grammar_path = _repo_path(authority_config["grammar"], "grammar authority")
    registry_path = _repo_path(authority_config["candidate_registry"], "candidate registry authority")
    grammar = load_topology_grammar(grammar_path)
    candidate_registry = load_perspective_candidate_registry(registry_path)
    scorer_sha256 = canonical_sha256(config["scorer"])
    base_snapshot = _base_snapshot(config["base"])
    base_candidate = G0CandidateBundle.create_base(
        candidate_id=config["base"]["candidate_id"],
        snapshot=base_snapshot,
        grammar=grammar,
        candidate_registry=candidate_registry,
    )

    ledger = G0FormationLedger.empty()
    parent_by_snapshot: dict[str, G0CandidateBundle] = {
        base_candidate.snapshot.snapshot_sha256: base_candidate
    }
    episode_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    problems: list[str] = []

    base_attempt, rows_e, rows_t, episodes, traces = _evaluate_candidate(
        formation_attempt_id="formation.base",
        candidate=base_candidate,
        config=config,
        grammar=grammar,
        candidate_registry=candidate_registry,
        scorer_sha256=scorer_sha256,
    )
    episode_rows.extend(rows_e)
    trace_rows.extend(rows_t)
    problems.extend(
        verify_g0_formation_attempt(
            base_attempt,
            grammar=grammar,
            candidate_registry=candidate_registry,
            parents=parent_by_snapshot,
            episodes=episodes,
            traces=traces,
        )
    )
    ledger = ledger.append(base_attempt)

    expected_valid = 1
    for mutation in config["mutations"]:
        request = _mutation_request(mutation, base_candidate.snapshot)
        construction = attempt_g0_construction(
            request,
            source=base_candidate.snapshot,
            grammar=grammar,
            candidate_registry=candidate_registry,
        )
        expected = mutation["expected_status"]
        actual = "applied" if construction.status is G0ConstructionStatus.APPLIED_SHADOW else "refused"
        _require(actual == expected, f"mutation {mutation['formation_attempt_id']} status drifted")
        if construction.status is G0ConstructionStatus.REFUSED:
            formation = G0FormationAttempt.create(
                formation_attempt_id=mutation["formation_attempt_id"],
                candidate=None,
                construction_request=request,
                construction_attempt=construction,
            )
            problems.extend(
                verify_g0_formation_attempt(
                    formation,
                    grammar=grammar,
                    candidate_registry=candidate_registry,
                    parents=parent_by_snapshot,
                    episodes={},
                    traces={},
                )
            )
            ledger = ledger.append(formation)
            continue
        expected_valid += 1
        candidate = G0CandidateBundle.create_derived(
            candidate_id=mutation["candidate_id"],
            parent=base_candidate,
            request=request,
            construction_attempt=construction,
        )
        formation, rows_e, rows_t, episodes, traces = _evaluate_candidate(
            formation_attempt_id=mutation["formation_attempt_id"],
            candidate=candidate,
            config=config,
            grammar=grammar,
            candidate_registry=candidate_registry,
            scorer_sha256=scorer_sha256,
            construction_request=request,
            construction_attempt=construction,
        )
        episode_rows.extend(rows_e)
        trace_rows.extend(rows_t)
        problems.extend(
            verify_g0_formation_attempt(
                formation,
                grammar=grammar,
                candidate_registry=candidate_registry,
                parents=parent_by_snapshot,
                episodes=episodes,
                traces=traces,
            )
        )
        ledger = ledger.append(formation)
        parent_by_snapshot[candidate.snapshot.snapshot_sha256] = candidate

    problems.extend(ledger.verify())
    archive = build_g0_pareto_archive(ledger)
    evaluated = [row.attempt for row in ledger.entries if row.attempt.status is G0FormationStatus.EVALUATED]
    refused = [row.attempt for row in ledger.entries if row.attempt.status is G0FormationStatus.REFUSED]
    _require(len(evaluated) == expected_valid, "valid G0 candidate admission count drifted")
    decisions = {row.attempt_sha256: row.status for row in archive.decisions}
    eligible_statuses = {
        G0ParetoDecisionStatus.RETAINED,
        G0ParetoDecisionStatus.DOMINATED,
        G0ParetoDecisionStatus.OBJECTIVE_TIE,
    }
    _require(
        all(decisions.get(row.attempt_sha256) in eligible_statuses for row in evaluated),
        "evaluated G0 candidate disappeared before Pareto projection",
    )
    _require(
        all(decisions.get(row.attempt_sha256) is G0ParetoDecisionStatus.INELIGIBLE for row in refused),
        "refused G0 construction entered the Pareto frontier",
    )
    cohorts = {row.objective.evaluation_cohort_sha256 for row in evaluated if row.objective is not None}
    _require(len(cohorts) == 1, "evaluated G0 candidates do not share one task cohort")
    _require(
        all(len(row.assessments) == len(config["tasks"]) for row in evaluated),
        "evaluated G0 candidate task coverage drifted",
    )
    proposal_count = len(ledger.entries)
    mutation_proposal_count = len(config["mutations"])
    mutation_evaluated_count = len(evaluated) - 1
    summary = {
        "attempt_count": len(ledger.entries),
        "proposed_candidate_count": proposal_count,
        "structurally_valid_candidate_count": expected_valid,
        "evaluated_candidate_count": len(evaluated),
        "refused_control_count": len(refused),
        "proposal_admission_fraction_microunits": (1_000_000 * len(evaluated) // proposal_count),
        "valid_candidate_admission_fraction_microunits": (1_000_000 * len(evaluated) // expected_valid),
        "mutation_proposal_count": mutation_proposal_count,
        "mutation_evaluated_count": mutation_evaluated_count,
        "mutation_admission_fraction_microunits": (
            1_000_000 * mutation_evaluated_count // mutation_proposal_count
        ),
        "task_count": len(config["tasks"]),
        "trace_count": len(trace_rows),
        "pareto_retained_count": len(archive.retained_attempt_sha256s),
        "ledger_head_sha256": ledger.head_sha256,
        "ledger_sha256": ledger.sha256,
        "pareto_archive_sha256": archive.archive_sha256,
    }
    task_authorities = [
        {"task_id": task["task_id"], "task_authority_sha256": _task_authority(task)}
        for task in config["tasks"]
    ]
    all_ok = not problems
    core = {
        "schema": RECEIPT_SCHEMA,
        "study_id": config["study_id"],
        "claim_scope": CLAIM_SCOPE,
        "config_authority": {
            "path": str(config_path.relative_to(REPO_ROOT.resolve())),
            "bytes": len(config_raw),
            "sha256": hashlib.sha256(config_raw).hexdigest(),
        },
        "implementation_authority": _implementation_authority(),
        "grammar_authority": _file_authority(grammar_path),
        "candidate_registry_authority": _file_authority(registry_path),
        "scorer_authority_sha256": scorer_sha256,
        "task_authorities": task_authorities,
        "episodes": episode_rows,
        "traces": trace_rows,
        "ledger": ledger.payload(),
        "pareto_archive": archive.payload(),
        "summary": summary,
        "all_ok": all_ok,
        "problems": sorted(set(problems)),
        **_INERT,
        "nonclaims": config["nonclaims"],
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def verify_receipt(
    config_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:

    problems: list[str] = []
    receipt_raw = b""
    declared: dict[str, Any] = {}
    try:
        receipt_raw = _read_regular(receipt_path.resolve(), "G0 formation receipt")
        declared = _decode_json(receipt_raw, "G0 formation receipt")
        _exact(declared, _RECEIPT_FIELDS, "G0 formation receipt")
        seal_core = dict(declared)
        seal = seal_core.pop("receipt_sha256")
        _require(seal == canonical_sha256(seal_core), "G0 formation receipt seal mismatch")
        ledger_payload = declared["ledger"]
        archive_payload = declared["pareto_archive"]
        _require(isinstance(ledger_payload, dict), "G0 formation ledger payload is invalid")
        _require(isinstance(archive_payload, dict), "G0 Pareto payload is invalid")
        parsed_ledger = G0FormationLedger.from_payload(ledger_payload)
        parsed_archive = G0ParetoArchive.from_payload(archive_payload)
        _require(
            build_g0_pareto_archive(parsed_ledger) == parsed_archive,
            "G0 Pareto archive does not replay from its ledger",
        )
        expected = build_receipt(config_path.resolve())
        _require(declared == expected, "G0 formation receipt differs from deterministic regeneration")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
    receipt_authority = {
        "path": str(receipt_path.resolve()),
        "bytes": len(receipt_raw),
        "sha256": hashlib.sha256(receipt_raw).hexdigest(),
    }
    core = {
        "schema": VERIFICATION_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "config_path": str(config_path.resolve()),
        "receipt_authority": receipt_authority,
        "declared_receipt_sha256": declared.get("receipt_sha256"),
        "all_ok": not problems,
        "problems": sorted(set(problems)),
        **_INERT,
    }
    return {**core, "verification_sha256": canonical_sha256(core)}


def run_to_path(config_path: Path, receipt_path: Path) -> dict[str, Any]:
    receipt = build_receipt(config_path)
    _atomic_json(receipt_path, receipt)
    return receipt


def verify_to_path(config_path: Path, receipt_path: Path, output_path: Path) -> dict[str, Any]:
    verification = verify_receipt(config_path, receipt_path)
    _atomic_json(output_path, verification)
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="build the deterministic mechanics receipt")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    run_parser.add_argument("--out", type=Path, default=DEFAULT_RECEIPT_PATH)
    verify_parser = subparsers.add_parser("verify", help="regenerate and verify a receipt")
    verify_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    verify_parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT_PATH)
    verify_parser.add_argument("--out", type=Path, default=DEFAULT_VERIFICATION_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run_to_path(args.config, args.out)
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        return 0 if result["all_ok"] else 1
    result = verify_to_path(args.config, args.receipt, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
