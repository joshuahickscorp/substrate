"""Frozen principal developmental campaign.

Core units are full eleven phase histories.  Divergence units are matched specialized histories.  Workers
compute state but never publish; the supervisor validates and atomically owns receipts and checkpoints.

"""

from __future__ import annotations

import json
import math
import resource
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

from substrate import v2config as C
from substrate import v2executor as X
from substrate import v2fabric as F
from substrate import v2io as io
from substrate import v2state as S

READY_TAG = "substrate-v2-developmental-ready"
PRINCIPAL = io.RUNS / "principal"
UNITS = PRINCIPAL / "units"
CHECKPOINTS = PRINCIPAL / "checkpoints"
LOCKS = PRINCIPAL / "locks"
MANIFEST = io.CONFIGS / "principal_manifest.json"


@dataclass(frozen=True)
class WorkUnit:
    identity: str
    hypothesis: str
    arm: str
    history_seed: int
    split: str
    domain_sequence: tuple[str, ...]
    body: str
    dependencies: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    resource_class: str
    timeout: int
    retry_rule: str
    checkpoint_boundary: str
    artifact_family: str
    claim_ceiling: str
    kind: str = "core"


def _unit(
    identity: str,
    hypothesis: str,
    arm: str,
    seed: int,
    split: str,
    body: str,
    kind: str,
) -> WorkUnit:
    return WorkUnit(
        identity=identity,
        hypothesis=hypothesis,
        arm=arm,
        history_seed=seed,
        split=split,
        domain_sequence=("A", "B", "A", "B", "C", "D", "negative", "restore", "body", "terminal"),
        body=body,
        dependencies=(),
        inputs=(
            "configs/substrate/v2/frozen_configuration.json",
            "evidence/substrate/v2/SUBSTRATE_V2_ADMISSION.json",
        ),
        outputs=(
            f"runs/substrate/v2/principal/units/{identity}.json",
            f"runs/substrate/v2/principal/checkpoints/{identity}.json",
        ),
        resource_class="small_cpu",
        timeout=180,
        retry_rule="one exact retry after process failure; deterministic failure is terminal",
        checkpoint_boundary="every phase and final owned state",
        artifact_family=f"principal:{identity}",
        claim_ceiling=C.CLAIM_BOUNDARY["maximum"],
        kind=kind,
    )


def work_units() -> tuple[WorkUnit, ...]:
    units = []
    for seed in C.SPLITS["principal"]:
        for arm in C.CORE_ARMS:
            units.append(
                _unit(
                    f"core-{seed}-{arm}-general",
                    "H_D1,H_D2,H_D3,H_D5",
                    arm,
                    seed,
                    "principal",
                    "general",
                    "core",
                )
            )
    for seed in C.SPLITS["principal"]:
        for variant in C.DIVERGENCE_ARMS:
            units.append(
                _unit(
                    f"divergence-{seed}-{variant}",
                    "H_D4",
                    variant,
                    seed,
                    "principal",
                    "general",
                    "divergence",
                )
            )
    for seed in C.SPLITS["replication"]:
        for body in ("compact", "tool_dominant"):
            units.append(
                _unit(
                    f"body-{seed}-full_v2-{body}",
                    "body_continuity_replication",
                    "full_v2",
                    seed,
                    "replication",
                    body,
                    "body",
                )
            )
    return tuple(units)


def _phase_domains(phase: str, count: int) -> list[str]:
    if phase == "phase_0_cold_baseline":
        return [("A", "B", "C", "D")[index % 4] for index in range(count)]
    if phase in {"phase_1_domain_A_development", "phase_3_return_held_out_A", "phase_8_interruption_exact_restore"}:
        return ["A"] * count
    if phase in {"phase_2_domain_B_development", "phase_4_held_out_B_transfer"}:
        return ["B"] * count
    if phase == "phase_5_positive_transfer_C":
        return ["C"] * count
    if phase in {"phase_6_positive_transfer_D", "phase_7_negative_transfer_challenge"}:
        return ["D"] * count
    if phase == "phase_9_body_or_tool_change":
        return ["D"] * count
    return [("A", "B", "C", "D")[index % 4] for index in range(count)]


def _episode_row(episode: S.DevelopmentalEpisode, phase: str) -> dict:
    procedure = next(
        (component.removeprefix("procedure:") for component in episode.components_used if component.startswith("procedure:")),
        None,
    )
    semantic = next(
        (component.removeprefix("semantic:") for component in episode.components_used if component.startswith("semantic:")),
        None,
    )
    return {
        "identity": episode.identity,
        "phase": phase,
        "domain": episode.domain,
        "correct": episode.outcome["correct"],
        "utility": float(episode.outcome["correct"]) - C.COMPUTE_PRICE * episode.compute,
        "compute": episode.compute,
        "procedure": procedure,
        "semantic": semantic,
        "components": episode.components_used,
        "prediction_precedes_outcome": True,
        "activation": False,
    }


def _phase_summary(phase: str, rows: list[dict], identity: str, exact: bool) -> dict:
    return {
        "phase": phase,
        "episodes": len(rows),
        "accuracy": statistics.fmean(float(row["correct"]) for row in rows),
        "utility": statistics.fmean(row["utility"] for row in rows),
        "compute": sum(row["compute"] for row in rows),
        "procedure_uses": sum(row["procedure"] is not None for row in rows),
        "semantic_uses": sum(row["semantic"] is not None for row in rows),
        "checkpoint_identity": identity,
        "exact_restore": exact,
    }


def _self_model_probe(seed: int) -> dict:
    model = S.ConditionalSelfModel("domain_plus_procedure_conditional_estimate")
    failures = []
    index = 20_000
    while len(failures) < 8:
        task = F.generate_task(seed, "A", index, "principal_self_model_training")
        proposal = F.execute("always_first", task.observation, task.alternatives)
        outcome = task.reveal(proposal)
        if not outcome["correct"]:
            prediction = model.predict(
                kind="accuracy",
                domain="A",
                task_signature=task.task_signature,
                procedure=None,
                body="general",
                step=index * 2,
            )
            model.observe(prediction, 0.0, step=index * 2 + 1)
            failures.append(task.identity)
        index += 1
    while True:
        task = F.generate_task(seed, "A", index, "principal_self_model_held_out")
        baseline = F.execute("always_first", task.observation, task.alternatives)
        if baseline != task.private_target:
            break
        index += 1
    prediction = model.predict(
        kind="accuracy",
        domain="A",
        task_signature=task.task_signature,
        procedure=None,
        body="general",
        step=index * 2,
    )
    action = "verify" if prediction.predicted < 0.45 else "continue"
    with_utility = 1.0 - (C.COMPUTE_PRICE if action == "verify" else 0.0)
    without_utility = 0.0
    model.observe(prediction, 1.0, step=index * 2 + 1)
    return {
        "training_failures": failures,
        "held_out_task": task.identity,
        "prediction": prediction.predicted,
        "action": action,
        "with_self_model_utility": with_utility,
        "without_self_model_utility": without_utility,
        "margin": with_utility - without_utility,
        "prediction_precedes_outcome": prediction.made_at_step < prediction.outcome_step,
    }


def _allocation_probe(seed: int) -> dict:
    cases = S.allocation_cases(seed, 256)
    train, evaluate = cases[:128], cases[128:]
    simple = S.evaluate_allocator("always_verify", train, evaluate)
    learned = S.evaluate_allocator("tabular_contextual_policy", train, evaluate)
    oracle = S.evaluate_allocator("oracle", train, evaluate)
    return {
        "strongest_simple_policy": "always_verify",
        "strongest_simple_utility": simple["mean_utility"],
        "learned_utility": learned["mean_utility"],
        "oracle_utility": oracle["mean_utility"],
        "oracle_residual": oracle["mean_utility"] - simple["mean_utility"],
        "learned_margin": learned["mean_utility"] - simple["mean_utility"],
        "simple_compute": simple["compute"],
        "learned_compute": learned["compute"],
        "oracle_compute": oracle["compute"],
        "learned_rows": learned["rows"],
    }


def run_core(unit: WorkUnit) -> dict:
    entity = S.DevelopmentalEntity(unit.arm, body=unit.body, entity_id=f"principal:{unit.history_seed}:{unit.arm}")
    phase_rows = []
    ledger = []
    body_report = None
    reset_count = 0
    started = time.perf_counter()
    for phase_index, (phase, count) in enumerate(C.EPISODES_PER_PHASE.items()):
        if unit.arm == "fresh_control" and phase_index:
            entity = S.DevelopmentalEntity(
                unit.arm,
                body=unit.body,
                entity_id=f"principal:{unit.history_seed}:{unit.arm}:reset:{phase_index}",
            )
            reset_count += 1
        if phase == "phase_8_interruption_exact_restore":
            entity.unfinished_tasks.append("finish terminal held out evaluation")
            entity.unresolved_hypotheses.append("whether competence survives interruption")
            entity = S.DevelopmentalEntity.restore(entity.checkpoint())
        if phase == "phase_9_body_or_tool_change":
            replacement = "compact" if entity.body_state["name"] == "tool_dominant" else "tool_dominant"
            body_report = entity.replace_body(replacement)
        rows = []
        for index, domain in enumerate(_phase_domains(phase, count)):
            task = F.generate_task(
                unit.history_seed,
                domain,
                phase_index * 10_000 + index,
                phase,
            )
            episode = entity.experience(
                task,
                allow_verification=unit.arm == "more_compute",
            )
            row = _episode_row(episode, phase)
            rows.append(row)
            ledger.append(row)
        checkpoint = entity.checkpoint()
        restored = S.DevelopmentalEntity.restore(checkpoint)
        exact = restored.identity_hash() == checkpoint["identity"]
        entity = restored
        phase_rows.append(_phase_summary(phase, rows, checkpoint["identity"], exact))
    negative_wrong_selected = any(
        row["phase"] == "phase_7_negative_transfer_challenge"
        and row["procedure"]
        and "boundary_route" in row["procedure"]
        for row in ledger
    )
    transfer_ab = [
        row
        for row in ledger
        if row["phase"] == "phase_2_domain_B_development"
    ][:4]
    transfer_cd = [
        row
        for row in ledger
        if row["phase"] == "phase_6_positive_transfer_D"
    ]
    return_phase = next(row for row in phase_rows if row["phase"] == "phase_3_return_held_out_A")
    acquired_a_rows = [
        row for row in ledger if row["phase"] == "phase_1_domain_A_development"
    ][-4:]
    acquired_a = statistics.fmean(float(row["correct"]) for row in acquired_a_rows)
    retention_loss = max(0.0, acquired_a - return_phase["accuracy"])
    allocation = _allocation_probe(unit.history_seed) if unit.arm == "full_v2" else None
    self_model = _self_model_probe(unit.history_seed) if unit.arm == "full_v2" else None
    checkpoint = entity.checkpoint()
    payload = {
        "kind": unit.kind,
        "arm": unit.arm,
        "seed": unit.history_seed,
        "split": unit.split,
        "body": unit.body,
        "phase_rows": phase_rows,
        "episode_ledger": ledger,
        "episode_count": len(ledger),
        "expected_episode_count": sum(C.EPISODES_PER_PHASE.values()),
        "A_acquired_accuracy": acquired_a,
        "A_return_accuracy": return_phase["accuracy"],
        "retention_loss": retention_loss,
        "B_transfer_early_utility": statistics.fmean(row["utility"] for row in transfer_ab),
        "B_held_out_utility": next(
            row["utility"] for row in phase_rows if row["phase"] == "phase_4_held_out_B_transfer"
        ),
        "C_to_D_transfer_utility": statistics.fmean(row["utility"] for row in transfer_cd),
        "negative_wrong_procedure_selected": negative_wrong_selected,
        "identity_exact_every_phase": all(row["exact_restore"] for row in phase_rows),
        "body_report": body_report,
        "body_continuity": bool(body_report)
        and all(
            body_report[key]
            for key in ("continuing_entity", "goals_preserved", "procedures_preserved", "body_change_visible_in_identity")
        ),
        "interruption_recovery": next(
            row["exact_restore"] for row in phase_rows if row["phase"] == "phase_8_interruption_exact_restore"
        ),
        "procedures_induced": len(entity.procedures),
        "procedures_transferred": sum(
            procedure.status == "transferable" for procedure in entity.procedures.values()
        ),
        "procedure_uses": len(entity.procedure_use_receipts),
        "semantic_records": len(entity.semantic),
        "allocation_probe": allocation,
        "self_model_probe": self_model,
        "fresh_reset_count": reset_count,
        "final_identity": checkpoint["identity"],
        "runtime_seconds": time.perf_counter() - started,
        "activation": False,
    }
    return {"payload": payload, "checkpoint": checkpoint}


def run_divergence(unit: WorkUnit) -> dict:
    started = time.perf_counter()
    variant = unit.arm
    domain = "C" if variant == "history_B" else "A"
    history_label = "history_B" if variant == "history_B" else "history_A"
    tasks = [
        F.generate_task(unit.history_seed, domain, index, f"divergence_{history_label}")
        for index in range(12)
    ]
    if variant == "shuffled_history":
        tasks = list(reversed(tasks))
    entity = S.DevelopmentalEntity(
        "full_v2",
        body="general",
        entity_id=f"divergence:{unit.history_seed}",
    )
    ledger = []
    for task in tasks:
        ledger.append(_episode_row(entity.experience(task, allow_verification=False), "development"))
    for target in ("B", "D"):
        for index in range(8):
            task = F.generate_task(unit.history_seed, target, 30_000 + index, f"divergence_eval_{target}")
            ledger.append(_episode_row(entity.experience(task, allow_verification=False), f"evaluation_{target}"))
    checkpoint = entity.checkpoint()
    evaluation = {
        target: statistics.fmean(
            float(row["correct"]) for row in ledger if row["phase"] == f"evaluation_{target}"
        )
        for target in ("B", "D")
    }
    payload = {
        "kind": "divergence",
        "variant": variant,
        "seed": unit.history_seed,
        "split": unit.split,
        "development_domain": domain,
        "episode_ledger": ledger,
        "episode_count": len(ledger),
        "state_identity": checkpoint["identity"],
        "evaluation": evaluation,
        "procedures": sorted(entity.procedures),
        "runtime_seconds": time.perf_counter() - started,
        "activation": False,
    }
    return {"payload": payload, "checkpoint": checkpoint}


def compute_unit(unit: WorkUnit) -> dict:
    result = run_divergence(unit) if unit.kind == "divergence" else run_core(unit)
    return {"unit": asdict(unit), **result}


def _power(pilot: list[float], n: int) -> dict:
    deviation = statistics.stdev(pilot) if len(pilot) > 1 else 0.0
    if deviation == 0:
        power = 1.0
    else:
        signal = math.sqrt(n) * C.SESOI / deviation
        power = statistics.NormalDist().cdf(signal - 1.959963984540054)
    return {
        "pilot_n": len(pilot),
        "paired_standard_deviation": deviation,
        "n": n,
        "sesoi": C.SESOI,
        "approximate_two_sided_power": power,
        "target_met": power >= 0.8,
    }


def prepare() -> dict:
    admission = io.load("SUBSTRATE_V2_ADMISSION.json")
    if not admission["principal_execution_licensed"]:
        raise X.Refused("principal authority cannot freeze before terminal admission")
    units = work_units()
    context = X.context()
    canary = io.load("SUBSTRATE_V2_PROCEDURE_TRANSFER_CANARY.json")
    pilot = [row["margin"] for row in canary["positive_pairs"]["A_to_B"]]
    power = _power(pilot, len(C.SPLITS["principal"]))
    if not power["target_met"]:
        raise X.Refused("the frozen minimum principal N does not provide target power for the SESOI")
    nodes = [asdict(unit) for unit in units]
    manifest = {
        "schema": "substrate-v2-principal-manifest/v1",
        **context,
        "ready_tag": READY_TAG,
        "units": [unit.identity for unit in units],
        "unit_registry_digest": io.sha_obj(nodes),
        "core_independent_units": len(C.SPLITS["principal"]),
        "replication_independent_units": len(C.SPLITS["replication"]),
        "principal_work_units": len(units),
        "core_history_episodes": sum(C.EPISODES_PER_PHASE.values()),
        "power": power,
        "activation": False,
    }
    io.config_json("principal_manifest.json", manifest)
    documents = {
        "SUBSTRATE_V2_PRINCIPAL_AUTHORITY.json": {
            "schema": "substrate-v2-principal-authority/v1",
            **manifest,
            "independent_unit": C.STATISTICS["independent_unit"],
            "source_edit_after_launch": "forbidden",
            "publication": "supervisor only after validation",
            "activation": False,
        },
        "SUBSTRATE_V2_PRINCIPAL_DAG.json": {
            "schema": "substrate-v2-principal-dag/v1",
            "nodes": nodes,
            "node_count": len(nodes),
            "registry_digest": io.sha_obj(nodes),
            "deterministic_order": True,
            "exclusive_artifact_families": len({unit.artifact_family for unit in units}) == len(units),
            "activation": False,
        },
        "SUBSTRATE_V2_RESOURCE_PLAN.json": {
            "schema": "substrate-v2-resource-plan/v1",
            "supervisors": 1,
            "workers": admission["worker_count"],
            "native_threads_per_worker": 1,
            "estimated_checkpoint_mib_per_unit": 1,
            "disk_floor_gib": 25,
            "memory_free_floor_percent": 10,
            "swap_free_floor_mib": 512,
            "mps": False,
            "hawking": "observation only",
            "activation": False,
        },
        "SUBSTRATE_V2_STOP_AND_FUTILITY.json": {
            "schema": "substrate-v2-stop-and-futility/v1",
            **C.STOP_AND_FUTILITY,
            "activation": False,
        },
        "SUBSTRATE_V2_CLAIM_CEILING.json": {
            "schema": "substrate-v2-claim-ceiling/v1",
            **C.CLAIM_BOUNDARY,
        },
    }
    for name, document in documents.items():
        io.seal(name, document)
    return {"manifest": manifest, "documents": documents}


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=io.ROOT, text=True).strip()


def _ready_identity() -> dict:
    tag_commit = _git("rev-parse", f"{READY_TAG}^{{}}")
    tag_tree = _git("rev-parse", f"{READY_TAG}^{{tree}}")
    current_tree = _git("rev-parse", "HEAD^{tree}")
    remote = _git("ls-remote", "origin", f"refs/tags/{READY_TAG}")
    return {
        "tag_commit": tag_commit,
        "tag_tree": tag_tree,
        "current_tree": current_tree,
        "tree_identical": tag_tree == current_tree,
        "remote_tag_exists": bool(remote),
    }


def _prelaunch() -> dict:
    admission = io.load("SUBSTRATE_V2_ADMISSION.json")
    if not admission["principal_execution_licensed"]:
        raise X.Refused("principal execution is not licensed")
    manifest = json.loads(MANIFEST.read_text())
    expected = X.context()
    for key in ("source_digest", "configuration_digest", "split_digest"):
        if manifest.get(key) != expected[key]:
            raise X.Refused(f"principal manifest drift: {key}")
    ready = _ready_identity()
    if not ready["tree_identical"] or not ready["remote_tag_exists"]:
        raise X.Refused("principal tree is not the remotely published ready tree")
    stale_locks = list(LOCKS.glob("*.lock")) if LOCKS.exists() else []
    if stale_locks:
        raise X.Refused(f"stale principal locks: {[path.name for path in stale_locks]}")
    from substrate.v2 import _hawking_snapshot, _resource_snapshot

    resources = _resource_snapshot()
    hawking = _hawking_snapshot()
    safe = (
        resources["disk_available_gib"] >= 25
        and (resources["memory_free_percent"] or 0) >= 10
        and (resources["swap_free_mib"] or 0) >= 512
    )
    if not safe:
        raise X.Refused("principal resource floor is not safe")
    return {
        "admission": True,
        "context": expected,
        "ready": ready,
        "resources": resources,
        "hawking": hawking,
        "stale_locks": [],
        "safe": safe,
        "activation": False,
    }


def status() -> dict:
    units = work_units()
    complete = []
    invalid = []
    for unit in units:
        path = UNITS / f"{unit.identity}.json"
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text())
            if X.validate_receipt(document):
                complete.append(unit.identity)
            else:
                invalid.append(unit.identity)
        except json.JSONDecodeError:
            invalid.append(unit.identity)
    state = "principal development"
    if len(complete) == len(units) and not invalid:
        state = "verification"
    return {
        "phase": state,
        "complete": len(complete),
        "total": len(units),
        "remaining": len(units) - len(complete),
        "invalid": invalid,
        "stop_requested": io.STOP.is_file(),
        "activation": False,
    }


def run() -> dict:
    prelaunch = _prelaunch()
    units = work_units()
    pending = [
        unit
        for unit in units
        if not (UNITS / f"{unit.identity}.json").is_file()
    ]
    if not pending:
        return {"prelaunch": prelaunch, "status": status(), "runtime_seconds": 0.0, "peak_rss_raw": 0}
    workers = io.load("SUBSTRATE_V2_WORKER_AUTHORITY.json")["selected_workers"]
    started = time.perf_counter()
    published = 0
    with X.exclusive_claim(LOCKS / "supervisor.lock"), ProcessPoolExecutor(max_workers=workers) as pool:
        for unit, result in zip(pending, pool.map(compute_unit, pending, chunksize=1), strict=True):
            if io.STOP.is_file():
                break
            if result["unit"]["identity"] != unit.identity:
                raise X.Refused("worker result identity mismatch")
            X.validate_context(
                prelaunch["context"],
                split=unit.split,
                seed=unit.history_seed,
            )
            checkpoint = result["checkpoint"]
            restored = S.DevelopmentalEntity.restore(checkpoint)
            if restored.identity_hash() != checkpoint["identity"]:
                raise X.Refused(f"checkpoint failed for {unit.identity}")
            io.run_json(f"principal/checkpoints/{unit.identity}.json", checkpoint)
            receipt = X.receipt_body(unit.identity, result["payload"], prelaunch["context"])
            X.publish_unit(f"principal/units/{unit.identity}.json", receipt)
            published += 1
    runtime = time.perf_counter() - started
    report = {
        "schema": "substrate-v2-principal-run/v1",
        "prelaunch": prelaunch,
        "published_this_invocation": published,
        "runtime_seconds": runtime,
        "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "workers": workers,
        "status": status(),
        "activation": False,
    }
    io.run_json("principal/run.json", report)
    return report
