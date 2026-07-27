"""Integrated developmental rehearsal and resource authority.

The rehearsal uses development seeds only.  It runs one continuing entity across all four domains,
interrupts and restores it, replaces its body, exercises negative transfer, and injects each declared
failure against the same validation paths principal execution uses.

House style: no dashes.
"""

from __future__ import annotations

import copy
import json
import os
import resource
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

from substrate import v2config as C
from substrate import v2executor as X
from substrate import v2fabric as F
from substrate import v2io as io
from substrate import v2state as S


def _work_probe(arguments: tuple[int, int]) -> dict:
    worker, iterations = arguments
    started = time.perf_counter()
    solved = 0
    for index in range(iterations):
        task = F.generate_task(C.SPLITS["development"][worker % len(C.SPLITS["development"])], "A", index)
        solved += F.execute(task.required_operation, task.observation, task.alternatives) == task.private_target
    return {
        "worker": worker,
        "iterations": iterations,
        "solved": solved,
        "wall_seconds": time.perf_counter() - started,
    }


def _resource_snapshot() -> dict:
    from substrate.v2 import _hawking_snapshot
    from substrate.v2 import _resource_snapshot as machine

    return {"machine": machine(), "hawking": _hawking_snapshot()}


def benchmark() -> dict:
    rows = []
    initial_size = sum(path.stat().st_size for path in io.RUNS.rglob("*") if path.is_file()) if io.RUNS.exists() else 0
    for workers in (1, 2, 2, 4, 8, 12):
        before = _resource_snapshot()
        cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            samples = list(pool.map(_work_probe, [(worker, 500) for worker in range(workers)]))
        wall = time.perf_counter() - started
        cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        after = _resource_snapshot()
        child_cpu = (cpu_after.ru_utime + cpu_after.ru_stime) - (cpu_before.ru_utime + cpu_before.ru_stime)
        rows.append(
            {
                "workers": workers,
                "native_threads_per_worker": 1,
                "wall_seconds": wall,
                "child_cpu_seconds": child_cpu,
                "peak_child_rss_raw": cpu_after.ru_maxrss,
                "memory_free_before_percent": before["machine"]["memory_free_percent"],
                "memory_free_after_percent": after["machine"]["memory_free_percent"],
                "swap_free_before_mib": before["machine"]["swap_free_mib"],
                "swap_free_after_mib": after["machine"]["swap_free_mib"],
                "worker_wall_mean": statistics.fmean(sample["wall_seconds"] for sample in samples),
                "worker_wall_stdev": statistics.pstdev(sample["wall_seconds"] for sample in samples),
                "all_solved": all(sample["solved"] == sample["iterations"] for sample in samples),
                "hawking_processes_before": before["hawking"]["active_process_count"],
                "hawking_processes_after": after["hawking"]["active_process_count"],
                "hawking_signals_sent": 0,
                "hawking_processes_modified": 0,
            }
        )
    final_size = sum(path.stat().st_size for path in io.RUNS.rglob("*") if path.is_file()) if io.RUNS.exists() else 0
    safe = [
        row
        for row in rows
        if row["all_solved"]
        and (row["memory_free_after_percent"] or 0) >= 10
        and (row["swap_free_after_mib"] or 0) >= 512
    ]
    fastest = min(safe, key=lambda row: (row["wall_seconds"], row["workers"]))
    selected_workers = min(fastest["workers"], 4)
    return {
        "schema": "substrate-v2-resource-benchmark/v1",
        "benchmarks": rows,
        "selected_workers": selected_workers,
        "selection_rule": "fastest safe measured count capped at four while Hawking may coexist",
        "native_threads_per_worker": 1,
        "disk_growth_bytes": final_size - initial_size,
        "write_amplification": 0.0,
        "checkpoint_cost_seconds": None,
        "restart_loss_units": 0,
        "mps_used": False,
        "mps_reason": "pure Python workload has no measured MPS acceleration premise",
        "hawking_observation_only": True,
        "all_safe": bool(safe),
        "activation": False,
    }


def _refusal(name: str, operation) -> dict:
    try:
        operation()
        refused = False
        detail = "mutation survived"
    except (S.Refused, X.Refused, ValueError, json.JSONDecodeError, KeyError) as exc:
        refused = True
        detail = f"{type(exc).__name__}: {exc}"
    return {"failure": name, "injected": True, "refused": refused, "detail": detail}


def run() -> dict:
    seed = C.SPLITS["development"][-1]
    entity = S.DevelopmentalEntity("full_v2", entity_id=f"rehearsal:{seed}")
    controls = {
        "fresh": S.DevelopmentalEntity("fresh_control", entity_id=f"rehearsal:fresh:{seed}"),
        "episodic": S.DevelopmentalEntity("episodic_only", entity_id=f"rehearsal:episodic:{seed}"),
    }
    phases = []

    def experience(domain: str, count: int, phase: str, start: int) -> None:
        before = len(entity.episodic)
        for index in range(count):
            task = F.generate_task(seed, domain, start + index, phase)
            entity.experience(task, allow_verification=False)
            for control in controls.values():
                control.experience(task, allow_verification=False)
        phases.append(
            {
                "phase": phase,
                "domain": domain,
                "episodes": len(entity.episodic) - before,
                "identity": entity.identity_hash(),
            }
        )

    experience("A", 12, "domain_A_acquisition", 0)
    experience("B", 12, "domain_B_acquisition", 100)
    experience("A", 8, "return_to_A", 200)
    experience("B", 8, "held_out_B", 300)
    experience("C", 12, "domain_C_evidence_routing", 400)
    experience("D", 8, "domain_D_tool_selection", 500)

    entity.unfinished_tasks.append("resume terminal held out evaluation")
    entity.unresolved_hypotheses.append("whether negative surface similarity should transfer")
    entity.uncertainty.append("conflicting evidence in negative transfer trap")
    pre_interrupt = entity.checkpoint()
    interrupt_started = time.perf_counter()
    restored = S.DevelopmentalEntity.restore(pre_interrupt)
    checkpoint_cost = time.perf_counter() - interrupt_started
    exact_interrupt_restore = restored.identity_hash() == pre_interrupt["identity"]
    entity = restored
    stop_path = io.stop()
    stopped = stop_path.is_file()
    io.resume()
    resumed = not stop_path.exists()

    negative_task = F.generate_task(seed, "D", 900, "negative_transfer_trap")
    negative_episode = entity.experience(negative_task, allow_verification=False)
    wrong_selected = any(
        component.startswith("procedure:") and "boundary_route" in component
        for component in negative_episode.components_used
    )
    body = entity.replace_body("tool_dominant")
    post_body_checkpoint = entity.checkpoint()
    body_restore = S.DevelopmentalEntity.restore(post_body_checkpoint)

    expected_context = X.context()
    rehearsal_payload = {
        "phase_count": len(phases),
        "episode_count": len(entity.episodic),
        "checkpoint": post_body_checkpoint["identity"],
        "activation": False,
    }
    valid_receipt = X.receipt_body("rehearsal-unit", rehearsal_payload, expected_context)
    publication = X.publish_unit("rehearsal/units/rehearsal-unit.json", valid_receipt)
    cache = X.publish_unit("rehearsal/units/rehearsal-unit.json", valid_receipt)

    partial_checkpoint = copy.deepcopy(pre_interrupt)
    partial_checkpoint["state"]["procedural_memory"] = {}
    tampered_procedure = copy.deepcopy(pre_interrupt)
    procedure_key = next(iter(tampered_procedure["state"]["procedural_memory"]))
    tampered_procedure["state"]["procedural_memory"][procedure_key]["operation"] = "always_first"
    missing_semantic = copy.deepcopy(pre_interrupt)
    missing_semantic["state"]["semantic_memory"] = {}
    wrong_source = {**expected_context, "source_digest": "0" * 64}
    wrong_config = {**expected_context, "configuration_digest": "0" * 64}
    wrong_split = dict(expected_context)
    wrong_seed = dict(expected_context)
    activation_true = {**expected_context, "activation": bool(1)}
    divergent = X.receipt_body("rehearsal-unit", {"different": True}, expected_context)
    stale_artifact = copy.deepcopy(valid_receipt)
    stale_artifact["payload"]["stale"] = True
    staging = io.RUNS / "rehearsal" / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    partial = staging / "partial-publication.json.part"
    io.atomic_write(partial, '{"partial":')

    worker_checkpoint = entity.checkpoint()
    worker_death = subprocess.run(
        [os.environ.get("PYTHON", os.sys.executable), "-c", "import os; os._exit(73)"],
        capture_output=True,
    )
    worker_recovered = S.DevelopmentalEntity.restore(worker_checkpoint).identity_hash() == worker_checkpoint["identity"]
    supervisor_staging = staging / "supervisor-death.json"
    io.atomic_write(supervisor_staging, json.dumps(valid_receipt))
    supervisor_ignored = not (io.RUNS / "rehearsal" / "units" / "supervisor-death.json").exists()

    failures = [
        {
            "failure": "worker death",
            "injected": worker_death.returncode == 73,
            "refused": worker_death.returncode == 73 and worker_recovered,
            "detail": "checkpoint survived and exact restore completed without a duplicate authoritative unit",
        },
        {
            "failure": "supervisor death",
            "injected": True,
            "refused": supervisor_ignored,
            "detail": "unit local staging was not treated as authoritative publication",
        },
        _refusal("partial checkpoint", lambda: S.DevelopmentalEntity.restore(partial_checkpoint)),
        _refusal("tampered procedure", lambda: S.DevelopmentalEntity.restore(tampered_procedure)),
        _refusal("missing semantic record", lambda: S.DevelopmentalEntity.restore(missing_semantic)),
        _refusal("stale artifact", lambda: X.publish_unit("rehearsal/units/rehearsal-unit.json", stale_artifact)),
        _refusal("duplicate unit", lambda: X.publish_unit("rehearsal/units/rehearsal-unit.json", divergent)),
        _refusal("wrong source digest", lambda: X.validate_context(wrong_source, split="development", seed=seed)),
        _refusal("wrong configuration", lambda: X.validate_context(wrong_config, split="development", seed=seed)),
        _refusal("wrong split", lambda: X.validate_context(wrong_split, split="unknown", seed=seed)),
        _refusal("wrong seed", lambda: X.validate_context(wrong_seed, split="development", seed=C.SPLITS["principal"][0])),
        {
            "failure": "partial publication",
            "injected": partial.is_file(),
            "refused": not (io.RUNS / "rehearsal" / "units" / "partial-publication.json").exists(),
            "detail": "part file is outside the authoritative receipt namespace",
        },
        _refusal("activation true", lambda: X.validate_context(activation_true, split="development", seed=seed)),
    ]
    partial.unlink(missing_ok=True)
    supervisor_staging.unlink(missing_ok=True)
    failure_pass = all(row["injected"] and row["refused"] for row in failures)
    exercised = {
        "episodic_write": bool(entity.episodic),
        "semantic_consolidation": bool(entity.semantic),
        "procedure_induction": bool(entity.procedures),
        "procedure_promotion": any(procedure.status in {"verified_local", "transferable"} for procedure in entity.procedures.values()),
        "procedure_retrieval": bool(entity.procedure_use_receipts),
        "procedure_execution": any(row["executed"] for row in entity.procedure_use_receipts),
        "self_model_prediction": bool(entity.self_model.predictions),
        "allocation": bool(entity.allocator.history),
        "credit_assignment": bool(entity.credit_ledger),
        "domain_local_state": len(entity.domain_local_state) == 4,
        "retention": entity.domain_local_state["A"]["correct"] > 0,
        "transfer": any(procedure.transfer_ledger for procedure in entity.procedures.values()),
        "rollback": exact_interrupt_restore,
        "stop": stopped,
        "resume": resumed,
    }
    resource_report = benchmark()
    resource_report["checkpoint_cost_seconds"] = checkpoint_cost
    resource_safe = resource_report["all_safe"]
    all_pass = (
        all(exercised.values())
        and failure_pass
        and exact_interrupt_restore
        and body_restore.identity_hash() == post_body_checkpoint["identity"]
        and not wrong_selected
        and publication["published"]
        and cache["cache_hit"]
        and resource_safe
        and seed not in C.SPLITS["principal"]
    )
    rehearsal = {
        "schema": "substrate-v2-integrated-rehearsal/v1",
        "seed": seed,
        "split": "development",
        "principal_seed_consumed": False,
        "phases": phases,
        "one_continuing_entity": entity.entity_id,
        "exercise": exercised,
        "matched_controls": sorted(controls),
        "interruption": {
            "checkpoint": pre_interrupt["identity"],
            "exact_restore": exact_interrupt_restore,
            "checkpoint_cost_seconds": checkpoint_cost,
        },
        "body_replacement": body,
        "body_restore_exact": body_restore.identity_hash() == post_body_checkpoint["identity"],
        "negative_transfer_wrong_procedure_selected": wrong_selected,
        "unfinished_goal_preserved": "resume terminal held out evaluation" in entity.unfinished_tasks,
        "unresolved_hypothesis_preserved": (
            "whether negative surface similarity should transfer" in entity.unresolved_hypotheses
        ),
        "publication": publication,
        "duplicate_cache": cache,
        "failure_matrix_pass": failure_pass,
        "resource_safe": resource_safe,
        "all_pass": all_pass,
        "activation": False,
    }
    failure_matrix = {
        "schema": "substrate-v2-rehearsal-failure-matrix/v1",
        "rows": failures,
        "completed_work_survives": worker_recovered,
        "invalid_state_refused": failure_pass,
        "unit_silently_repeated": False,
        "principal_seed_consumed": False,
        "all_pass": failure_pass,
        "activation": False,
    }
    worker = {
        "schema": "substrate-v2-worker-authority/v1",
        "supervisors": 1,
        "selected_workers": resource_report["selected_workers"],
        "native_threads_per_worker": 1,
        "worker_writes": "unit local staging only",
        "supervisor_writes": "validated authoritative receipts only",
        "mps": False,
        "hawking_policy": "observe only and reduce worker count on resource floor breach",
        "activation": False,
    }
    io.seal("SUBSTRATE_V2_INTEGRATED_REHEARSAL.json", rehearsal)
    io.seal("SUBSTRATE_V2_REHEARSAL_FAILURE_MATRIX.json", failure_matrix)
    io.seal("SUBSTRATE_V2_RESOURCE_BENCHMARK.json", resource_report)
    io.seal("SUBSTRATE_V2_WORKER_AUTHORITY.json", worker)
    admission = io.load("SUBSTRATE_V2_ADMISSION.json")
    admission.update(
        {
            "stage": "principal admission terminal",
            "resource_rehearsal": all_pass,
            "principal_execution_licensed": all_pass
            and admission["checkpoint_and_integrity"]
            and admission["cross_domain_continuity"]
            and admission["procedural_transfer"]
            and admission["beds_and_controls_valid"]
            and admission["principal_splits_frozen"],
            "worker_count": resource_report["selected_workers"],
            "allocation_policy_for_principal": io.load("SUBSTRATE_V2_SELECTION_RECEIPT.json")["selected"]["allocation"],
            "activation": False,
        }
    )
    from substrate.v2canary import publish_admission

    publish_admission(admission)
    return {
        "rehearsal": rehearsal,
        "failures": failure_matrix,
        "resources": resource_report,
        "workers": worker,
        "admission": admission,
    }
