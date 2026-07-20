
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import REPO_ROOT, compose
from ..experiments import REGISTRY

SCHEMA = "mop-project-experiment-exhaustion/v1"
ATTEMPT_SCHEMA = "mop-project-experiment-attempt/v1"
DIAGNOSTIC_SCHEMA = "mop-project-diagnostic-execution/v1"

FRESH = "freshly-executed-verified"
DURABLE = "already-durable-hash-verifiable"
PENDING = "runnable-not-yet-run"
IMPLEMENTATION_BLOCKED = "implementation-blocked"
DATA_BLOCKED = "rights-data-blocked"
MODEL_BLOCKED = "upstream-model-blocked"
HARDWARE_BLOCKED = "measured-hardware-blocked"
CATEGORIES = {
    FRESH,
    DURABLE,
    PENDING,
    IMPLEMENTATION_BLOCKED,
    DATA_BLOCKED,
    MODEL_BLOCKED,
    HARDWARE_BLOCKED,
}

ENV_LATER_CLASS_BLOCKERS: dict[str, tuple[str, str]] = {
    "e6_relational": (
        DATA_BLOCKED,
        "official runtime and cache/control integration are verified; the registered claim now "
        "requires a rights-clean annotated natural cohort, untouched test membership, serial "
        "learned/random caches, and independent verification",
    ),
    "e10_openended": (
        IMPLEMENTATION_BLOCKED,
        "persistent action mechanics now exist; the registered capstone still needs bounded "
        "population search, environment generation, sustained-horizon evaluation, and transfer",
    ),
    "mop_cm8_custom_jepa_pilot": (
        MODEL_BLOCKED,
        "the local fail-closed preflight exists; the registered claim still requires positive CM7 "
        "control evidence plus citable same-referent teacher inputs",
    ),
}

DEFAULT_EXECUTION_ROOT = REPO_ROOT / "runs" / "project_exhaustion" / "current"
DEFAULT_LEDGER_PATH = REPO_ROOT / "proof" / "PROJECT_EXPERIMENT_EXHAUSTION.json"
EMBEDDED_COMMITTED_RECEIPTS = frozenset({"proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json"})
BLOCKED_SMOKE_IDS = {
    "e10_openended",
    "mop_cm8_custom_jepa_pilot",
}

E6_RUNTIME_EVIDENCE = (
    "proof/VJEPA21_VITB_LOAD.json",
    "proof/VJEPA21_VITB_FORWARD.json",
    "proof/VJEPA21_VITB_FORWARD_64F.json",
)
E6_INTEGRATION_EVIDENCE = (*E6_RUNTIME_EVIDENCE, "proof/E6_VITB_DENSE_PREFLIGHT.json")

STANDALONE_EVIDENCE: dict[str, tuple[str, ...]] = {
    "mop_mt4_reasoning_router": ("runs/frontier_localization/local_preflights.json",),
    "mop_mt1_router_vs_best": ("runs/mot/mt123_router_pilots.json",),
    "mop_mt2_routing_vs_ensemble": ("runs/mot/mt123_router_pilots.json",),
    "mop_mt3_hetero_vs_homo": ("runs/mot/mt123_router_pilots.json",),
    "mop_mt5_adaptive_halting": ("runs/mot/mt5_adaptive_halting_recal.json",),
    "mop_mt6_confidence_stop": ("runs/mot/mt6_confidence_stop.json",),
    "mop_mt7_beam_search": ("runs/mot/mt7_beam_search.json",),
    "mop_mt8_latent_debate": ("runs/mot/mt8_latent_debate.json",),
    "mop_dr6_rollout_planning": ("runs/mot/dr6_rollout_planning.json",),
    "mop_dr8_fixed_point": (
        "runs/mot/dr8_fixed_point_vjepa.json",
        "runs/mot/dr8_fixed_point_randominit.json",
    ),
    "mop_dr9_verify_revise": ("runs/mot/dr9_verify_revise.json",),
    "mop_dr10_retrieve_reason": ("runs/mot/dr10_retrieve_reason.json",),
    "mop_dr11_mc_rollouts": ("runs/mot/dr11_mc_rollouts.json",),
    "mop_dr12_disagreement": ("runs/mot/dr12_disagreement_recal.json",),
    "mop_dr13_horizon_limit": (
        "runs/mot/dr13_horizon_limit.json",
        "runs/mot/dr13_predictor_fidelity.json",
        "runs/mot/dr13_readout_adapter.json",
    ),
    "mop_pr1_error_disjointness": ("runs/pre_studio/pr1_mode_error_disjointness.json",),
    "mop_pr2_plasticity_substrates": ("runs/mot/pr2_plasticity_substrates.json",),
    "mop_pr3_modular_real": ("runs/mot/dr2_sparse_real_pilot_seeds10.json",),
    "mop_pr4_epistemic_gate": ("runs/mot/pr4_epistemic_gate.json",),
    "mop_pr5_content_gated_cp": ("runs/mot/pr5_content_gated_cp_seeds10.json",),
    "mop_pr6_sleep_consolidation": ("runs/mot/pr6_sleep_consolidation.json",),
    "mop_pr7_fast_slow": ("runs/mot/pr7_fast_slow_seeds10.json",),
    "mop_pr8_retrieval_head": ("runs/mot/pr8_retrieval_head.json",),
    "mop_ws1_agreement_vs_confidence": ("runs/mot/ws1_agreement_vs_confidence.json",),
    "mop_ws2_fusion_tournament": ("runs/mot/ws2_fusion_tournament_seeds10.json",),
    "mop_ws3_arbitration": ("runs/mot/ws3_arbitration_recal.json",),
    "mop_ws4_bandwidth_sweep": ("runs/mot/ws4_bandwidth_sweep.json",),
    "mop_ws5_router_slot": ("runs/mot/ws5_slot_ablation_pilot_seeds10.json",),
    "mop_at3_time_axis": ("runs/mot/at3_time_axis_seeds10.json",),
    "mop_at4_programmatic_ceiling": ("runs/mot/at4_programmatic_ceiling.json",),
    "mop_at5_probe_class_sweep": ("runs/mot/at5_probe_class_sweep.json",),
    "mop_al1_uncertainty_router": ("runs/mot/al1_uncertainty_router_recal.json",),
    "mop_at2_mode_substrate_dep": ("runs/frontier_localization/local_preflights.json",),
    "mop_cm5_studio_rejuvenation": ("runs/frontier_localization/local_preflights.json",),
    "mop_cm11_developmental_plasticity": ("runs/frontier_localization/local_preflights.json",),
    "mop_cm12_mop_substrate_capstone": ("runs/frontier_localization/local_preflights.json",),
    "mop_cm10_action_forward_model": ("proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json",),
}

STANDALONE_SCRIPTS: dict[str, str] = {
    "mop_mt4_reasoning_router": "scripts/frontier_localization.py",
    "mop_mt1_router_vs_best": "scripts/mop_mt123_router_pilots.py",
    "mop_mt2_routing_vs_ensemble": "scripts/mop_mt123_router_pilots.py",
    "mop_mt3_hetero_vs_homo": "scripts/mop_mt123_router_pilots.py",
    "mop_mt5_adaptive_halting": "scripts/mop_mt5_adaptive_halting.py",
    "mop_mt6_confidence_stop": "scripts/mop_mt6_confidence_stop.py",
    "mop_mt7_beam_search": "scripts/mop_mt7_beam_search.py",
    "mop_mt8_latent_debate": "scripts/mop_mt8_latent_debate.py",
    "mop_dr6_rollout_planning": "scripts/mop_dr6_rollout_planning.py",
    "mop_dr8_fixed_point": "scripts/mop_dr8_fixed_point.py",
    "mop_dr9_verify_revise": "scripts/mop_dr9_verify_revise.py",
    "mop_dr10_retrieve_reason": "scripts/mop_dr10_retrieve_reason.py",
    "mop_dr11_mc_rollouts": "scripts/mop_dr11_mc_rollouts.py",
    "mop_dr12_disagreement": "scripts/mop_dr12_disagreement.py",
    "mop_dr13_horizon_limit": "scripts/mop_dr13_horizon_limit.py",
    "mop_pr1_error_disjointness": "scripts/pr1_mode_error_disjointness.py",
    "mop_pr2_plasticity_substrates": "scripts/mop_pr2_plasticity_substrates.py",
    "mop_pr3_modular_real": "scripts/mop_dr2_sparse_real_pilot.py",
    "mop_pr4_epistemic_gate": "scripts/mop_pr4_epistemic_gate.py",
    "mop_pr5_content_gated_cp": "scripts/mop_pr5_content_gated_cp.py",
    "mop_pr6_sleep_consolidation": "scripts/mop_pr6_sleep_consolidation.py",
    "mop_pr7_fast_slow": "scripts/mop_pr7_fast_slow.py",
    "mop_pr8_retrieval_head": "scripts/mop_pr8_retrieval_head.py",
    "mop_ws1_agreement_vs_confidence": "scripts/mop_ws1_agreement_vs_confidence.py",
    "mop_ws2_fusion_tournament": "scripts/mop_ws2_fusion_tournament.py",
    "mop_ws3_arbitration": "scripts/mop_ws3_arbitration.py",
    "mop_ws4_bandwidth_sweep": "scripts/mop_ws4_bandwidth_sweep.py",
    "mop_ws5_router_slot": "scripts/mop_ws5_slot_ablation_pilot.py",
    "mop_at3_time_axis": "scripts/mop_at3_time_axis.py",
    "mop_at4_programmatic_ceiling": "scripts/mop_at4_programmatic_ceiling.py",
    "mop_at5_probe_class_sweep": "scripts/mop_at5_probe_class_sweep.py",
    "mop_al1_uncertainty_router": "scripts/mop_al1_uncertainty_router.py",
    "mop_at2_mode_substrate_dep": "scripts/frontier_localization.py",
    "mop_cm5_studio_rejuvenation": "scripts/frontier_localization.py",
    "mop_cm11_developmental_plasticity": "scripts/frontier_localization.py",
    "mop_cm12_mop_substrate_capstone": "scripts/frontier_localization.py",
    "mop_cm10_action_forward_model": "scripts/p7_action_world_model_preflight.py",
}

STANDALONE_SCIENTIFIC_BLOCKERS: dict[str, str] = {
    "mop_mt4_reasoning_router": (
        "fixture mechanics run locally; the registered claim still needs compatible outputs from "
        "six distinct, verified reasoning primitives plus an independent verifier"
    ),
    "mop_at2_mode_substrate_dep": (
        "fixture mechanics run locally; the registered claim still needs a verified winning mode "
        "and citable matched real and same-architecture random-init caches"
    ),
    "mop_cm5_studio_rejuvenation": (
        "four-arm mechanics run locally; the effect is uninterpretable until the control arm first "
        "exhibits calibrated plasticity loss at a progressive local rung"
    ),
    "mop_cm11_developmental_plasticity": (
        "signature mechanics run locally; the registered claim still needs a calibrated curriculum "
        "and independent recomputation of the three developmental signatures"
    ),
    "mop_cm12_mop_substrate_capstone": (
        "routing mechanics run locally; the capstone still needs cleared compatible expert pilots, "
        "a shared battery, and the declared open-model control"
    ),
    "mop_cm10_action_forward_model": (
        "the deterministic rendering, paired same-parent interventions, and eight-arm action-forward "
        "comparison run locally with matched controls, calibration, horizons, planning, and mutations; "
        "the fixture supports the null, and the registered scientific claim now needs independently "
        "sourced action trajectories, an exact-referent action control, and predeclared replication"
    ),
}

REGISTRY_ONLY_BLOCKERS: dict[str, tuple[str, str]] = {
    "mop_dr1_video_cache": (
        DATA_BLOCKED,
        "requires rights-cleared natural clips with bound attributes; "
        "programmatic clips do not satisfy the row",
    ),
    "mop_dr2_sparse_real": (
        MODEL_BLOCKED,
        "a durable small-cache pilot exists, but the registered full real-latent stream is absent",
    ),
    "mop_dr3_latent_scratchpad": (
        DATA_BLOCKED,
        "requires DR1 dense tokens and a working-memory-heavy natural-video task",
    ),
    "mop_dr4_causal_intervention": (
        DATA_BLOCKED,
        "requires factor-annotated clips and true counterfactual clips from DR1",
    ),
    "mop_dr5_cross_substrate_consistency": (
        MODEL_BLOCKED,
        "the registered reasoning claim first needs a verified gain on one compatible task, a "
        "rights-clean same-task cohort, and active ViT-B/custom matched controls; the retired scale "
        "pilot is not a current dependency",
    ),
    "mop_dr7_latent_cot": (
        DATA_BLOCKED,
        "requires a DR1 multi-step relational task with supervised intermediate states",
    ),
    "mop_dr14_corruption": (
        DATA_BLOCKED,
        "the registered deterministic nested dropped-channel path is implemented; it now requires "
        "the citable dense natural-task cache and independent verification",
    ),
    "mop_dr15_modality_general": (
        MODEL_BLOCKED,
        "requires citable video, relational/language, and audio encoder families",
    ),
    "mop_at1_nuisance_grid": (
        MODEL_BLOCKED,
        "pilot exists, but the registered multi-encoder grid and matched random-init columns "
        "are not citable active inputs",
    ),
    "mop_al2_shared_latent_alignment": (
        DATA_BLOCKED,
        "requires an expanded meaningful shared-content cohort with session-level units, then "
        "equal-rank active ViT-B/custom and random-map controls; the retired scale pilot is not a "
        "current dependency",
    ),
    "mop_al3_audio_video_alignment": (
        DATA_BLOCKED,
        "requires rights-cleared temporally aligned audio-video clips and their citable caches",
    ),
    "mop_cm1_compositional_gate": (
        DATA_BLOCKED,
        "requires DR1 rights-cleared bound-attribute natural video",
    ),
    "mop_cm2_atlas_gate": (
        MODEL_BLOCKED,
        "requires CM1 failure plus a complete citable multi-encoder atlas",
    ),
    "mop_cm3_dense_vs_pooled": (
        DATA_BLOCKED,
        "requires DR1 dense natural-video tokens on the held-out-combination task",
    ),
    "mop_cm4_workspace_shell": (
        MODEL_BLOCKED,
        "laptop pilot exists, but the registered full real-cache grid is not present as citable evidence",
    ),
    "mop_cm6_distilled_density": (
        MODEL_BLOCKED,
        "requires a citable teacher cache and matched trainable student checkpoints",
    ),
    "mop_cm8_custom_jepa_pilot": (
        MODEL_BLOCKED,
        "a runnable local fail-closed preflight and predictive workbench exist, but the scientific "
        "CM8 row still requires a positive CM7 control result, CM1/CM2/DR1, and at least 64 "
        "same-referent citable teacher rows",
    ),
    "mop_cm9_slot_jepa_binding": (
        DATA_BLOCKED,
        "requires DR1 multi-object dense-token video and binding annotations",
    ),
    "mop_cm10_action_forward_model": (
        DATA_BLOCKED,
        "the P7 rendered action/world-model mechanics pass with a fixture null; the remaining "
        "registered claim requires independently sourced trajectories, an exact-referent action "
        "control, and replication",
    ),
}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_evidence(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return out
    out.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text())
            out["json_parse_ok"] = isinstance(payload, (dict, list))
            out["json_top_level"] = type(payload).__name__
        except (OSError, json.JSONDecodeError) as exc:
            out["json_parse_ok"] = False
            out["json_error"] = str(exc)
    return out


def _vjepa21_vitb_runtime_ready() -> tuple[bool, list[dict[str, Any]]]:

    evidence = [file_evidence(REPO_ROOT / relative) for relative in E6_RUNTIME_EVIDENCE]
    if not all(row.get("exists") and row.get("json_parse_ok") for row in evidence):
        return False, evidence
    receipts = [json.loads((REPO_ROOT / relative).read_text()) for relative in E6_RUNTIME_EVIDENCE]
    load, forward_8f, forward_64f = receipts
    checkpoint_hashes = {
        str((receipt.get("authority") or {}).get("checkpoint_sha256") or "") for receipt in receipts
    }
    ready = bool(
        len(checkpoint_hashes) == 1
        and all(len(value) == 64 for value in checkpoint_hashes)
        and load.get("status") == "passed"
        and (load.get("child") or {}).get("strict_load") is True
        and (load.get("child") or {}).get("parameters") == 86_833_152
        and (load.get("child") or {}).get("trainable_parameters") == 0
        and all(
            receipt.get("status") == "passed"
            and receipt.get("hardware_limit_reached") is False
            and (receipt.get("child") or {}).get("completed") is True
            and (receipt.get("child") or {}).get("output_finite") is True
            and (receipt.get("child") or {}).get("shape_matches") is True
            and (receipt.get("claim_boundary") or {}).get("vitb_runtime_evidence_gate_passed") is True
            for receipt in (forward_8f, forward_64f)
        )
    )
    return ready, evidence


def registry_rows() -> list[dict[str, Any]]:
    payload = yaml.safe_load((REPO_ROOT / "registry" / "experiments.yaml").read_text())
    rows = [dict(row) for row in payload["experiments"] if not str(row["id"]).startswith("f")]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate non-F ids in registry/experiments.yaml")
    return rows


def class_source_evidence(experiment_id: str) -> list[dict[str, Any]]:
    cls = REGISTRY[experiment_id]
    source = inspect.getsourcefile(cls)
    paths = [Path(source)] if source else []
    cfg = REPO_ROOT / "configs" / "experiment" / f"{experiment_id}.yaml"
    if cfg.is_file():
        paths.append(cfg)
    return [file_evidence(path.resolve()) for path in paths]


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(v) for v in value)
    return True


def verify_experiment_run(run_dir: Path, experiment_id: str) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.yaml"
    receipt_path = run_dir / "attempt_receipt.json"
    checks: dict[str, bool] = {
        "manifest_exists": manifest_path.is_file(),
        "config_exists": config_path.is_file(),
        "attempt_receipt_exists": receipt_path.is_file(),
    }
    errors: list[str] = []
    manifest: dict[str, Any] = {}
    cfg: dict[str, Any] = {}
    attempt_receipt: dict[str, Any] = {}
    if checks["manifest_exists"]:
        try:
            manifest = json.loads(manifest_path.read_text())
            checks["manifest_json"] = isinstance(manifest, dict)
        except (OSError, json.JSONDecodeError) as exc:
            checks["manifest_json"] = False
            errors.append(f"manifest: {exc}")
    else:
        checks["manifest_json"] = False
    if checks["config_exists"]:
        try:
            cfg = yaml.safe_load(config_path.read_text())
            checks["config_yaml"] = isinstance(cfg, dict)
        except (OSError, yaml.YAMLError) as exc:
            checks["config_yaml"] = False
            errors.append(f"config: {exc}")
    else:
        checks["config_yaml"] = False
    if checks["attempt_receipt_exists"]:
        try:
            attempt_receipt = json.loads(receipt_path.read_text())
            checks["attempt_receipt_json"] = isinstance(attempt_receipt, dict)
        except (OSError, json.JSONDecodeError) as exc:
            checks["attempt_receipt_json"] = False
            errors.append(f"attempt receipt: {exc}")
    else:
        checks["attempt_receipt_json"] = False

    live_experiment_path = REPO_ROOT / "configs" / "experiment" / f"{experiment_id}.yaml"
    live_experiment: dict[str, Any] | None = None
    if live_experiment_path.is_file():
        try:
            loaded = yaml.safe_load(live_experiment_path.read_text())
            live_experiment = loaded if isinstance(loaded, dict) else None
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"live experiment config: {exc}")
    expected_run_dir: str | None
    try:
        expected_run_dir = str(run_dir.relative_to(REPO_ROOT))
    except ValueError as exc:
        raise ValueError(f"run directory is outside repository: {run_dir}") from exc
    expected_source_evidence = class_source_evidence(experiment_id)

    metrics = manifest.get("metrics")
    contract = (manifest.get("extra") or {}).get("contract") or {}
    worker_report = attempt_receipt.get("worker_report") or {}
    checks.update(
        {
            "manifest_name_matches": manifest.get("name") == experiment_id,
            "status_ok": manifest.get("status") == "ok",
            "finished_recorded": isinstance(manifest.get("finished"), (int, float)),
            "metrics_nonempty": isinstance(metrics, dict) and bool(metrics),
            "metrics_finite": _all_finite(metrics),
            "contract_id_matches": contract.get("id") == experiment_id,
            "config_id_matches": (cfg.get("experiment") or {}).get("id") == experiment_id,
            "cpu_device_recorded": manifest.get("device") == "cpu",
            "attempt_schema_matches": attempt_receipt.get("schema") == ATTEMPT_SCHEMA,
            "attempt_experiment_id_matches": attempt_receipt.get("experiment_id") == experiment_id,
            "attempt_run_dir_matches": expected_run_dir is not None
            and attempt_receipt.get("run_dir") == expected_run_dir,
            "attempt_returncode_zero": attempt_receipt.get("returncode") == 0,
            "attempt_not_timed_out": attempt_receipt.get("timed_out") is False,
            "attempt_worker_report_matches": worker_report.get("experiment_id") == experiment_id,
            "attempt_source_evidence_current": attempt_receipt.get("source_evidence")
            == expected_source_evidence,
            "resolved_experiment_config_current": live_experiment is None
            or cfg.get("experiment") == live_experiment,
        }
    )
    verified = all(checks.values())
    evidence = [file_evidence(path) for path in (manifest_path, config_path) if path.is_file()]
    if receipt_path.is_file():
        evidence.append(file_evidence(receipt_path))
    return {
        "verified": verified,
        "checks": checks,
        "errors": errors,
        "run_dir": expected_run_dir,
        "seed": manifest.get("seed"),
        "result_tag": manifest.get("result_tag"),
        "metrics": metrics if isinstance(metrics, dict) else None,
        "evidence": evidence,
    }


def verified_class_run(execution_root: Path, experiment_id: str) -> dict[str, Any] | None:
    parent = execution_root / "classes" / experiment_id
    if not parent.is_dir():
        return None
    for run_dir in sorted(parent.glob("attempt_*"), reverse=True):
        result = verify_experiment_run(run_dir, experiment_id)
        if result["verified"]:
            return result
    return None


def next_attempt_dir(execution_root: Path, experiment_id: str) -> Path:
    parent = execution_root / "classes" / experiment_id
    parent.mkdir(parents=True, exist_ok=True)
    used = [int(p.name.split("_")[-1]) for p in parent.glob("attempt_[0-9][0-9][0-9]")]
    return parent / f"attempt_{(max(used, default=0) + 1):03d}"


@dataclass(frozen=True)
class AttemptResult:
    experiment_id: str
    run_dir: str
    command: list[str]
    returncode: int | None
    timed_out: bool
    seconds: float
    stdout_tail: str
    stderr_tail: str
    worker_report: dict[str, Any] | None


def _run_one_subprocess(
    experiment_id: str,
    run_dir: Path,
    seed: int,
    timeout_s: float,
    script_path: Path,
) -> AttemptResult:
    command = [
        sys.executable,
        str(script_path),
        "--worker-experiment",
        experiment_id,
        "--run-dir",
        str(run_dir),
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLCONFIGDIR": str(run_dir / ".mplconfig"),
        }
    )
    started = time.perf_counter()
    timed_out = False
    returncode: int | None = None
    stdout = ""
    stderr = ""
    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        returncode = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
    seconds = time.perf_counter() - started
    worker_report = None
    for line in reversed(stdout.splitlines()):
        if line.startswith("PROJECT_EXHAUSTION_WORKER="):
            with suppress(json.JSONDecodeError):
                worker_report = json.loads(line.partition("=")[2])
            break
    result = AttemptResult(
        experiment_id=experiment_id,
        run_dir=str(run_dir.relative_to(REPO_ROOT)),
        command=command,
        returncode=returncode,
        timed_out=timed_out,
        seconds=round(seconds, 6),
        stdout_tail=stdout[-12000:],
        stderr_tail=stderr[-12000:],
        worker_report=worker_report,
    )
    receipt = {
        "schema": ATTEMPT_SCHEMA,
        **asdict(result),
        "recorded_at_unix": time.time(),
        "source_evidence": class_source_evidence(experiment_id),
    }
    _atomic_json(run_dir / "attempt_receipt.json", receipt)
    return result


def execute_cpu_classes(
    execution_root: Path = DEFAULT_EXECUTION_ROOT,
    *,
    seed: int = 20260709,
    max_workers: int = 6,
    timeout_s: float = 300.0,
    wall_seconds: float = 5100.0,
    script_path: Path | None = None,
) -> dict[str, Any]:
    script_path = script_path or REPO_ROOT / "scripts" / "project_experiment_exhaustion.py"
    ids = sorted(
        experiment_id
        for experiment_id, cls in REGISTRY.items()
        if not experiment_id.startswith("f") and cls.tier == "cpu-now"
    )
    already = {experiment_id for experiment_id in ids if verified_class_run(execution_root, experiment_id)}
    pending = [experiment_id for experiment_id in ids if experiment_id not in already]
    started = time.perf_counter()
    submitted: list[str] = []
    results: dict[str, dict[str, Any]] = {}
    batch_size = max(1, max_workers)
    for offset in range(0, len(pending), batch_size):
        remaining_wall = wall_seconds - (time.perf_counter() - started)
        if remaining_wall <= 0:
            break
        batch = pending[offset : offset + batch_size]
        per_run_timeout = min(timeout_s, remaining_wall)
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {}
            for experiment_id in batch:
                run_dir = next_attempt_dir(execution_root, experiment_id)
                run_dir.mkdir(parents=True, exist_ok=False)
                future = pool.submit(
                    _run_one_subprocess,
                    experiment_id,
                    run_dir,
                    seed,
                    per_run_timeout,
                    script_path,
                )
                futures[future] = experiment_id
                submitted.append(experiment_id)
            for future in as_completed(futures):
                experiment_id = futures[future]
                try:
                    result = future.result()
                    results[experiment_id] = asdict(result)
                except Exception as exc:  # the orchestrator records every failure and continues
                    results[experiment_id] = {"orchestrator_error": repr(exc)}

    verified = {
        experiment_id: verified_class_run(execution_root, experiment_id)
        for experiment_id in ids
        if verified_class_run(execution_root, experiment_id) is not None
    }
    summary = {
        "schema": "mop-project-experiment-class-batch/v1",
        "seed": seed,
        "max_workers": max_workers,
        "per_experiment_timeout_s": timeout_s,
        "wall_budget_s": wall_seconds,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "eligible_ids": ids,
        "eligible_count": len(ids),
        "resumed_verified_ids": sorted(already),
        "submitted_ids": submitted,
        "verified_ids": sorted(verified),
        "verified_count": len(verified),
        "remaining_queue": sorted(set(ids) - set(verified)),
        "attempt_results": results,
        "complete": len(verified) == len(ids),
    }
    _atomic_json(execution_root / "class_batch_receipt.json", summary)
    return summary


def execute_blocked_class_smokes(
    execution_root: Path = DEFAULT_EXECUTION_ROOT,
    *,
    seed: int = 20260709,
    timeout_s: float = 300.0,
    script_path: Path | None = None,
) -> dict[str, Any]:
    script_path = script_path or REPO_ROOT / "scripts" / "project_experiment_exhaustion.py"
    already = {
        experiment_id
        for experiment_id in BLOCKED_SMOKE_IDS
        if verified_class_run(execution_root, experiment_id)
    }
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(BLOCKED_SMOKE_IDS)) as pool:
        futures = {}
        for experiment_id in sorted(BLOCKED_SMOKE_IDS - already):
            run_dir = next_attempt_dir(execution_root, experiment_id)
            run_dir.mkdir(parents=True, exist_ok=False)
            futures[
                pool.submit(
                    _run_one_subprocess,
                    experiment_id,
                    run_dir,
                    seed,
                    timeout_s,
                    script_path,
                )
            ] = experiment_id
        for future in as_completed(futures):
            experiment_id = futures[future]
            try:
                results[experiment_id] = asdict(future.result())
            except Exception as exc:
                results[experiment_id] = {"orchestrator_error": repr(exc)}
    verified = {
        experiment_id: verified_class_run(execution_root, experiment_id)
        for experiment_id in BLOCKED_SMOKE_IDS
        if verified_class_run(execution_root, experiment_id)
    }
    summary = {
        "schema": "mop-project-blocked-class-smokes/v1",
        "scope": "code-path smoke only; registry scientific blockers remain in force",
        "eligible_ids": sorted(BLOCKED_SMOKE_IDS),
        "verified_ids": sorted(verified),
        "remaining_queue": sorted(BLOCKED_SMOKE_IDS - set(verified)),
        "complete": len(verified) == len(BLOCKED_SMOKE_IDS),
        "attempt_results": results,
    }
    _atomic_json(execution_root / "blocked_class_smoke_receipt.json", summary)
    return summary


def run_diagnostic_rows(seed: int = 20260709) -> dict[str, dict[str, Any]]:
    import torch

    from ..diagnostics.buffer_compression import retention_per_byte
    from ..diagnostics.compute import accounting, matched_within
    from ..diagnostics.difficulty_calibration import reference_separation
    from ..diagnostics.geometry import geometry_report
    from ..diagnostics.latent_robustness import degradation_curve
    from ..diagnostics.substrate_ablation import frozen_random_projection, substrate_ablation
    from ..diagnostics.sysid import sysid_report
    from ..diagnostics.transfer_matrix import transfer_matrix
    from ..substrate.datasets import make_task_stream

    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(192) % 3
    centers = torch.randn(3, 12, generator=g) * 2.5
    x = centers[y] + 0.25 * torch.randn(192, 12, generator=g)
    reference = x + 0.05 * torch.randn(192, 12, generator=g)
    tasks = make_task_stream(
        n_tasks=2,
        dim=12,
        classes_per_task=3,
        samples_per_task=72,
        separation=2.0,
        incremental="domain",
        seed=seed,
    )
    action = torch.randn(192, 3, generator=g)
    a_true = torch.eye(12) * 0.75
    b_true = torch.randn(12, 3, generator=g) * 0.35
    z_next = x @ a_true.T + action @ b_true.T
    model = torch.nn.Sequential(torch.nn.Linear(12, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))

    out: dict[str, dict[str, Any]] = {
        "d1_geometry": geometry_report(x, reference, k=5),
        "d2_substrate_ablation": substrate_ablation(x, y, seed=seed, bits=4, rank=4),
        "d3_difficulty_calibration": reference_separation(x, y, seed=seed),
        "d4_transfer_matrix": transfer_matrix(
            [(task.x, task.y) for task in tasks], epochs=40, lr=0.05, seed=seed
        ),
        "d5_compute_accounting": accounting(model, [12, 8, 3], steps=2, batch=16),
        "d6_rollout_gate": sysid_report(x, action, z_next, k=2, seed=seed),
        "a1_frozen_random_arm": {
            "shape": list(frozen_random_projection(x, seed=seed).shape),
            "finite": bool(torch.isfinite(frozen_random_projection(x, seed=seed)).all()),
        },
        "a2_matched_compute_arm": matched_within(4096, 4224, tol=0.10),
        "a3_buffer_compression": retention_per_byte(
            tasks, dim=12, n_classes=3, bits=(32, 4, 2), epochs=8, lr=0.05, seed=seed
        ),
        "a4_latent_robustness": degradation_curve(
            x,
            y,
            noise_levels=(0.0, 0.5, 1.5),
            dropout_levels=(0.0, 0.3, 0.7),
            bits=(32, 4, 2),
            seed=seed,
        ),
    }
    if set(out) != {
        "d1_geometry",
        "d2_substrate_ablation",
        "d3_difficulty_calibration",
        "d4_transfer_matrix",
        "d5_compute_accounting",
        "d6_rollout_gate",
        "a1_frozen_random_arm",
        "a2_matched_compute_arm",
        "a3_buffer_compression",
        "a4_latent_robustness",
    }:
        raise AssertionError("diagnostic execution surface drifted")
    if not _all_finite(out):
        raise ValueError("diagnostic execution produced non-finite values")
    return out


def execute_diagnostics(
    execution_root: Path = DEFAULT_EXECUTION_ROOT, *, seed: int = 20260709
) -> dict[str, Any]:
    parent = execution_root / "diagnostics"
    parent.mkdir(parents=True, exist_ok=True)
    existing = sorted(parent.glob("attempt_*/diagnostics.json"), reverse=True)
    for path in existing:
        try:
            payload = json.loads(path.read_text())
            if payload.get("schema") == DIAGNOSTIC_SCHEMA and set(payload.get("results", {})) == {
                "d1_geometry",
                "d2_substrate_ablation",
                "d3_difficulty_calibration",
                "d4_transfer_matrix",
                "d5_compute_accounting",
                "d6_rollout_gate",
                "a1_frozen_random_arm",
                "a2_matched_compute_arm",
                "a3_buffer_compression",
                "a4_latent_robustness",
            }:
                return {"resumed": True, "verified": True, "evidence": file_evidence(path)}
        except (OSError, json.JSONDecodeError):
            pass
    used = [int(p.name.split("_")[-1]) for p in parent.glob("attempt_[0-9][0-9][0-9]")]
    run_dir = parent / f"attempt_{(max(used, default=0) + 1):03d}"
    started = time.perf_counter()
    results = run_diagnostic_rows(seed)
    payload = {
        "schema": DIAGNOSTIC_SCHEMA,
        "seed": seed,
        "seconds": round(time.perf_counter() - started, 6),
        "results": results,
        "scope": "bounded synthetic diagnostic execution, not a project scientific verdict",
    }
    path = run_dir / "diagnostics.json"
    _atomic_json(path, payload)
    return {"resumed": False, "verified": True, "evidence": file_evidence(path)}


def _diagnostic_evidence(execution_root: Path, experiment_id: str) -> dict[str, Any] | None:
    for path in sorted((execution_root / "diagnostics").glob("attempt_*/diagnostics.json"), reverse=True):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema") == DIAGNOSTIC_SCHEMA and experiment_id in payload.get("results", {}):
            return {
                "verified": True,
                "result": payload["results"][experiment_id],
                "evidence": [file_evidence(path)],
            }
    return None


def _standalone_evidence(experiment_id: str) -> list[dict[str, Any]]:
    return [file_evidence(REPO_ROOT / rel) for rel in STANDALONE_EVIDENCE.get(experiment_id, ())]


def _surface(row: dict[str, Any]) -> str:
    experiment_id = str(row["id"])
    if experiment_id in REGISTRY:
        return "experiment-class"
    if row.get("status") == "implemented" and row.get("module"):
        return "diagnostic-module"
    if experiment_id in STANDALONE_EVIDENCE:
        return "standalone-script"
    return "registry-only"


def _row_source_evidence(row: dict[str, Any], surface: str) -> list[dict[str, Any]]:
    experiment_id = str(row["id"])
    if surface == "experiment-class":
        return class_source_evidence(experiment_id)
    if surface == "diagnostic-module" and row.get("module"):
        return [file_evidence(REPO_ROOT / str(row["module"]))]
    if surface == "standalone-script":
        return [file_evidence(REPO_ROOT / STANDALONE_SCRIPTS[experiment_id])]
    return []


def _sysctl(name: str) -> str | None:
    try:
        proc = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and value else None


def _host_profile() -> dict[str, Any]:
    disk = os.statvfs(REPO_ROOT)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
        "hardware_model": _sysctl("hw.model"),
        "physical_cpu_count": int(_sysctl("hw.physicalcpu") or 0) or None,
        "logical_cpu_count": int(_sysctl("hw.logicalcpu") or os.cpu_count() or 0),
        "memory_bytes": int(_sysctl("hw.memsize") or 0) or None,
        "free_disk_bytes_at_ledger": int(disk.f_bavail * disk.f_frsize),
    }


def _execution_campaign(execution_root: Path) -> dict[str, Any]:
    class_path = execution_root / "class_batch_receipt.json"
    smoke_path = execution_root / "blocked_class_smoke_receipt.json"
    diagnostic_paths = sorted(
        (execution_root / "diagnostics").glob("attempt_*/diagnostics.json"), reverse=True
    )
    class_payload: dict[str, Any] = {}
    smoke_payload: dict[str, Any] = {}
    if class_path.is_file():
        class_payload = json.loads(class_path.read_text())
    if smoke_path.is_file():
        smoke_payload = json.loads(smoke_path.read_text())
    attempts = class_payload.get("attempt_results", {})
    worker_reports = [
        result.get("worker_report")
        for result in attempts.values()
        if isinstance(result, dict) and isinstance(result.get("worker_report"), dict)
    ]
    rss_rows = [
        (report.get("max_rss_bytes", 0), report.get("experiment_id"))
        for report in worker_reports
        if isinstance(report, dict)
    ]
    max_rss, max_rss_id = max(rss_rows, default=(0, None))
    return {
        "reproduction_commands": [
            ".venv/bin/python scripts/project_experiment_exhaustion.py "
            "--execute-classes --max-workers 6 --timeout 300 --wall-seconds 5100",
            ".venv/bin/python scripts/project_experiment_exhaustion.py "
            "--execute-blocked-smokes --timeout 300",
            ".venv/bin/python scripts/project_experiment_exhaustion.py --execute-diagnostics --ledger",
        ],
        "class_batch": {
            "complete": class_payload.get("complete"),
            "eligible_count": class_payload.get("eligible_count"),
            "verified_count": class_payload.get("verified_count"),
            "wall_seconds": class_payload.get("wall_seconds"),
            "max_workers": class_payload.get("max_workers"),
            "timeout_count": sum(
                bool(result.get("timed_out")) for result in attempts.values() if isinstance(result, dict)
            ),
            "nonzero_return_count": sum(
                result.get("returncode") not in (0, None)
                for result in attempts.values()
                if isinstance(result, dict)
            ),
            "max_observed_worker_rss_bytes": max_rss,
            "max_observed_worker_rss_experiment": max_rss_id,
            "receipt": file_evidence(class_path),
        },
        "blocked_class_smokes": {
            "complete": smoke_payload.get("complete"),
            "verified_ids": smoke_payload.get("verified_ids", []),
            "receipt": file_evidence(smoke_path),
        },
        "diagnostics": {
            "verified": bool(diagnostic_paths),
            "receipt": file_evidence(diagnostic_paths[0]) if diagnostic_paths else None,
        },
    }


def _walk_evidence(value: Any):
    if isinstance(value, dict):
        if {"path", "exists"} <= set(value) and isinstance(value.get("path"), str):
            yield value
        for key, child in list(value.items()):
            if key != "embedded_artifacts":
                yield from _walk_evidence(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_evidence(child)


def _embed_runtime_artifacts(ledger: dict[str, Any]) -> None:
    artifacts: dict[str, dict[str, Any]] = {}
    for evidence in _walk_evidence(ledger):
        rel = str(evidence["path"])
        if not (rel.startswith("runs/") or rel in EMBEDDED_COMMITTED_RECEIPTS) or not evidence.get("exists"):
            continue
        path = REPO_ROOT / rel
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != evidence.get("sha256"):
            raise ValueError(f"runtime artifact changed while ledger was built: {rel}")
        try:
            exact_text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"runtime evidence must be small UTF-8 text, got {rel}") from exc
        evidence["embedded_artifact_sha256"] = digest
        if digest not in artifacts:
            artifacts[digest] = {
                "sha256": digest,
                "bytes": len(raw),
                "encoding": "utf-8",
                "source_paths": [rel],
                "exact_text": exact_text,
            }
        elif rel not in artifacts[digest]["source_paths"]:
            artifacts[digest]["source_paths"].append(rel)
    ledger["embedded_artifacts"] = dict(sorted(artifacts.items()))


def _embedded_text(ledger: dict[str, Any], evidence: dict[str, Any]) -> str | None:
    digest = evidence.get("embedded_artifact_sha256")
    artifact = (ledger.get("embedded_artifacts") or {}).get(digest)
    if not isinstance(artifact, dict):
        return None
    text = artifact.get("exact_text")
    return text if isinstance(text, str) else None


def verify_embedded_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    artifacts = ledger.get("embedded_artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        errors.append("embedded_artifacts missing or empty")
        artifacts = {}
    for digest, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            errors.append(f"artifact {digest}: not a mapping")
            continue
        exact_text = artifact.get("exact_text")
        if not isinstance(exact_text, str):
            errors.append(f"artifact {digest}: exact_text missing")
            continue
        raw = exact_text.encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != digest or artifact.get("sha256") != digest:
            errors.append(f"artifact {digest}: digest mismatch")
        if artifact.get("bytes") != len(raw):
            errors.append(f"artifact {digest}: byte count mismatch")

    runtime_evidence = [
        item for item in _walk_evidence(ledger) if str(item.get("path", "")).startswith("runs/")
    ]
    for evidence in runtime_evidence:
        digest = evidence.get("embedded_artifact_sha256")
        if not digest or digest != evidence.get("sha256") or digest not in artifacts:
            errors.append(f"{evidence.get('path')}: runtime evidence is not embedded")

    entries = ledger.get("entries")
    if not isinstance(entries, list):
        errors.append("entries missing")
        entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("non-mapping entry")
            continue
        experiment_id = str(entry.get("id"))
        evidence_rows = entry.get("evidence") or []
        if not isinstance(evidence_rows, list):
            errors.append(f"{experiment_id}: evidence is not a list")
            continue
        if entry.get("fresh_class_codepath_verified"):
            by_name = {Path(str(row.get("path"))).name: row for row in evidence_rows}
            manifest_text = _embedded_text(ledger, by_name.get("manifest.json", {}))
            config_text = _embedded_text(ledger, by_name.get("config.yaml", {}))
            if manifest_text is None or config_text is None:
                errors.append(f"{experiment_id}: embedded manifest/config missing")
                continue
            try:
                manifest = json.loads(manifest_text)
                cfg = yaml.safe_load(config_text)
            except (json.JSONDecodeError, yaml.YAMLError) as exc:
                errors.append(f"{experiment_id}: embedded manifest/config parse failed: {exc}")
                continue
            metrics = manifest.get("metrics")
            contract = (manifest.get("extra") or {}).get("contract") or {}
            checks = {
                "name": manifest.get("name") == experiment_id,
                "status": manifest.get("status") == "ok",
                "device": manifest.get("device") == "cpu",
                "metrics": isinstance(metrics, dict) and bool(metrics) and _all_finite(metrics),
                "contract": contract.get("id") == experiment_id,
                "config": (cfg.get("experiment") or {}).get("id") == experiment_id,
            }
            if not all(checks.values()):
                errors.append(f"{experiment_id}: embedded class checks failed {checks}")
        elif entry.get("classification") == FRESH:
            texts = [_embedded_text(ledger, row) for row in evidence_rows]
            try:
                payloads = [json.loads(text) for text in texts if text is not None]
            except json.JSONDecodeError as exc:
                errors.append(f"{experiment_id}: embedded diagnostic JSON invalid: {exc}")
                continue
            if not any(
                isinstance(payload, dict) and experiment_id in payload.get("results", {})
                for payload in payloads
            ):
                errors.append(f"{experiment_id}: embedded diagnostic result missing")
        elif entry.get("classification") == DURABLE:
            if not evidence_rows:
                errors.append(f"{experiment_id}: durable evidence missing")
            for row in evidence_rows:
                text = _embedded_text(ledger, row)
                try:
                    parsed = json.loads(text) if text is not None else None
                except json.JSONDecodeError as exc:
                    errors.append(f"{experiment_id}: durable JSON invalid: {exc}")
                    continue
                if not isinstance(parsed, (dict, list)):
                    errors.append(f"{experiment_id}: durable embedded artifact is empty or invalid")

    coverage = ledger.get("coverage") or {}
    counts = Counter(entry.get("classification") for entry in entries if isinstance(entry, dict))
    expected_counts = coverage.get("classification_counts") or {}
    if any(counts.get(category, 0) != expected_counts.get(category, 0) for category in CATEGORIES):
        errors.append("classification counts do not match entries")
    if coverage.get("registry_non_f_total") != len(entries):
        errors.append("registry_non_f_total does not match entries")
    if len({entry.get("id") for entry in entries if isinstance(entry, dict)}) != len(entries):
        errors.append("entry ids are not unique")
    return {
        "verified": not errors,
        "errors": errors,
        "embedded_artifact_count": len(artifacts),
        "runtime_evidence_reference_count": len(runtime_evidence),
        "entry_count": len(entries),
    }


def build_ledger(execution_root: Path = DEFAULT_EXECUTION_ROOT) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for row in registry_rows():
        experiment_id = str(row["id"])
        surface = _surface(row)
        classification: str
        reason: str
        execution_verified = False
        evidence: list[dict[str, Any]] = []
        observed_result: Any = None
        remaining_scientific_blocker: str | None = None

        if surface == "experiment-class":
            cls = REGISTRY[experiment_id]
            fresh = verified_class_run(execution_root, experiment_id)
            if cls.tier == "cpu-now" and fresh:
                classification = FRESH
                reason = (
                    "fresh CPU execution has a matching config, contract, nonempty finite metrics, "
                    "and hashed receipt"
                )
                execution_verified = True
                evidence = list(fresh["evidence"])
                observed_result = fresh["metrics"]
            elif cls.tier == "cpu-now":
                classification = PENDING
                reason = (
                    "runnable cpu-now class has no verified fresh attempt in the resumable execution root"
                )
            elif cls.tier == "env-later":
                classification, reason = ENV_LATER_CLASS_BLOCKERS.get(
                    experiment_id,
                    (
                        DATA_BLOCKED,
                        "the class has a bounded synthetic scaffold, but the registered experiment "
                        "requires an environment/data source",
                    ),
                )
                remaining_scientific_blocker = reason
            elif cls.tier == "2.1-only":
                runtime_ready, runtime_evidence = _vjepa21_vitb_runtime_ready()
                evidence = list(fresh["evidence"]) if fresh else []
                evidence.extend(runtime_evidence)
                if runtime_ready:
                    classification = IMPLEMENTATION_BLOCKED
                    reason = (
                        "official V-JEPA 2.1 ViT-B runtime availability is verified locally; the "
                        "registered E6 study still needs task wiring, citable referents, a dense "
                        "cache manifest, and matched scientific controls"
                    )
                    remaining_scientific_blocker = (
                        "E6 task wiring plus citable dense-token inputs and matched controls"
                    )
                else:
                    classification = MODEL_BLOCKED
                    reason = (
                        "the registered claim requires a verified official dense V-JEPA 2.1 runtime, "
                        "not its frozen-random fallback"
                    )
                    remaining_scientific_blocker = "verified official dense-token model runtime"
            else:
                classification = PENDING
                reason = f"runnable class tier {cls.tier!r} has not been freshly executed"
            if fresh and not evidence:
                evidence = list(fresh["evidence"])
            if experiment_id == "e6_relational":
                by_path = {str(item.get("path")): item for item in evidence}
                for item in [file_evidence(REPO_ROOT / path) for path in E6_INTEGRATION_EVIDENCE]:
                    by_path[str(item.get("path"))] = item
                evidence = list(by_path.values())
        elif surface == "diagnostic-module":
            diagnostic = _diagnostic_evidence(execution_root, experiment_id)
            if diagnostic:
                classification = FRESH
                reason = "fresh bounded diagnostic execution produced finite results and a hashed receipt"
                execution_verified = True
                evidence = diagnostic["evidence"]
                observed_result = diagnostic["result"]
            else:
                classification = PENDING
                reason = "implemented diagnostic module has no fresh execution receipt"
        elif surface == "standalone-script":
            evidence = _standalone_evidence(experiment_id)
            valid = bool(evidence) and all(
                item.get("exists") and item.get("json_parse_ok") and item.get("sha256") for item in evidence
            )
            if valid:
                classification = DURABLE
                reason = "prior standalone CPU result parses and is content-addressed in this ledger"
                execution_verified = True
                remaining_scientific_blocker = STANDALONE_SCIENTIFIC_BLOCKERS.get(
                    experiment_id,
                    "content hashes prove artifact integrity only; several receipts lack citable "
                    "active input manifests",
                )
            else:
                classification = PENDING
                reason = (
                    "standalone CPU row is runnable but its declared durable receipt is missing or invalid"
                )
        else:
            classification, reason = REGISTRY_ONLY_BLOCKERS.get(
                experiment_id,
                (
                    IMPLEMENTATION_BLOCKED,
                    "registry-only row has no runnable class, standalone evidence mapping, "
                    "or measured hardware receipt",
                ),
            )
            remaining_scientific_blocker = reason

        if classification == HARDWARE_BLOCKED:
            raise AssertionError(
                f"{experiment_id} was marked hardware-blocked without an experiment-specific "
                "measured boundary path"
            )
        entry = {
            "id": experiment_id,
            "name": row.get("name"),
            "kind": row.get("kind"),
            "series": row.get("series"),
            "registry_status": row.get("status"),
            "exp_tier": row.get("exp_tier"),
            "resource_tier": row.get("resource_tier"),
            "runtime_class": row.get("runtime_class"),
            "implementation_surface": surface,
            "classification": classification,
            "classification_reason": reason,
            "execution_verified": execution_verified,
            "fresh_class_codepath_verified": bool(
                surface == "experiment-class" and verified_class_run(execution_root, experiment_id)
            ),
            "scientific_claim_ready": False,
            "scientific_scope": (
                "execution/integrity verification only; no single-seed or synthetic result is "
                "promoted to a scientific verdict"
            ),
            "remaining_scientific_blocker": remaining_scientific_blocker,
            "evidence": evidence,
            "observed_result": observed_result,
            "source_evidence": _row_source_evidence(row, surface),
        }
        entries.append(entry)

    counts = Counter(entry["classification"] for entry in entries)
    surface_counts = Counter(entry["implementation_surface"] for entry in entries)
    fresh_class_codepaths = sum(entry["fresh_class_codepath_verified"] for entry in entries)
    provisional_null_signals: Counter[str] = Counter()
    for entry in entries:
        if not entry["fresh_class_codepath_verified"]:
            continue
        result = entry.get("observed_result") or {}
        if isinstance(result.get("null_supported"), bool):
            key = "null_supported" if result["null_supported"] else "null_not_supported"
        elif isinstance(result.get("null_rejected"), bool):
            key = "null_not_supported" if result["null_rejected"] else "null_supported"
        else:
            key = "not_standardized"
        provisional_null_signals[key] += 1
    queue = sorted(entry["id"] for entry in entries if entry["classification"] == PENDING)
    failures = sorted(
        entry["id"]
        for entry in entries
        if entry["classification"] in {IMPLEMENTATION_BLOCKED, DATA_BLOCKED, MODEL_BLOCKED, HARDWARE_BLOCKED}
    )
    ledger: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_unix": time.time(),
        "host": _host_profile(),
        "execution_campaign": _execution_campaign(execution_root),
        "scope": {
            "registry": "registry/experiments.yaml",
            "included": "every non-F row",
            "excluded": "F1-F20, owned by the separate Form-substrate campaign",
            "claim_boundary": (
                "fresh execution verifies runnable code and durable output only; it does not by "
                "itself verify a hypothesis"
            ),
        },
        "definitions": {
            FRESH: (
                "fresh run with matching id/config/contract, status ok, finite nonempty metrics, "
                "and SHA-256 evidence"
            ),
            DURABLE: (
                "prior JSON execution artifact exists, parses, and is content-addressed; "
                "input reproducibility is separate"
            ),
            PENDING: "a runnable surface exists but no qualifying verified execution is present",
            IMPLEMENTATION_BLOCKED: (
                "the full registered row has no runnable implementation after prerequisites are considered"
            ),
            DATA_BLOCKED: (
                "the first blocker is a rights-cleared dataset, environment, annotation, or trajectory source"
            ),
            MODEL_BLOCKED: "the first blocker is a missing citable upstream model/cache family",
            HARDWARE_BLOCKED: (
                "reserved for an experiment-specific measured local failure plus a resource "
                "projection; tier labels do not qualify"
            ),
        },
        "coverage": {
            "registry_non_f_total": len(entries),
            "classification_counts": {category: counts.get(category, 0) for category in sorted(CATEGORIES)},
            "surface_counts": dict(sorted(surface_counts.items())),
            "fresh_class_codepath_count": fresh_class_codepaths,
            "provisional_single_seed_null_signal_counts": {
                key: provisional_null_signals.get(key, 0)
                for key in ("null_supported", "null_not_supported", "not_standardized")
            },
            "accounted_exactly_once": len(entries) == len({entry["id"] for entry in entries}),
            "pending_execution_count": len(queue),
            "blocked_count": len(failures),
            "measured_hardware_blocked_count": counts.get(HARDWARE_BLOCKED, 0),
        },
        "resumable_queue": queue,
        "blocked_ids": failures,
        "hardware_boundary_conclusion": (
            "No non-F row is honestly proven M3-hardware-blocked by current experiment-specific evidence. "
            "GPU/studio tier labels are planning metadata, not measurements."
        ),
        "registry_evidence": file_evidence(REPO_ROOT / "registry" / "experiments.yaml"),
        "generator_evidence": [
            file_evidence(Path(__file__).resolve()),
            file_evidence(REPO_ROOT / "scripts" / "project_experiment_exhaustion.py"),
        ],
        "entries": entries,
    }
    if sum(ledger["coverage"]["classification_counts"].values()) != len(entries):
        raise AssertionError("classification counts do not cover the registry exactly once")
    _embed_runtime_artifacts(ledger)
    self_verification = verify_embedded_ledger(ledger)
    if not self_verification["verified"]:
        raise AssertionError(f"embedded ledger self-verification failed: {self_verification['errors']}")
    ledger["self_verification"] = self_verification
    return ledger


def write_ledger(
    path: Path = DEFAULT_LEDGER_PATH, execution_root: Path = DEFAULT_EXECUTION_ROOT
) -> dict[str, Any]:
    ledger = build_ledger(execution_root)
    _atomic_json(path, ledger)
    return ledger


def worker_execute_experiment(experiment_id: str, run_dir: Path, seed: int) -> dict[str, Any]:
    import resource

    from ..harness.runner import run_experiment

    if experiment_id not in REGISTRY:
        raise KeyError(experiment_id)
    cls = REGISTRY[experiment_id]
    allowed = cls.tier == "cpu-now" or experiment_id in BLOCKED_SMOKE_IDS
    if experiment_id.startswith("f") or not allowed:
        raise ValueError(
            f"worker only accepts non-F cpu-now classes or declared blocked smokes, "
            f"got {experiment_id} ({cls.tier})"
        )
    cfg = compose(
        [
            f"experiment={experiment_id}",
            "device=cpu",
            f"seed={seed}",
            "result_tag=project-exhaustion-provisional",
        ]
    )
    started = time.perf_counter()
    metrics = run_experiment(cfg, run_dir=run_dir)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = int(usage.ru_maxrss)
    if sys.platform != "darwin":
        max_rss *= 1024
    return {
        "experiment_id": experiment_id,
        "seconds": round(time.perf_counter() - started, 6),
        "max_rss_bytes": max_rss,
        "metric_keys": sorted(metrics),
    }
