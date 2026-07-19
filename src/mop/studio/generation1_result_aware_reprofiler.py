
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.studio.generation1_supervisor import atomic_write_json, canonical_sha256

SCHEMA = "mop-generation1-reprofile/v1"

MECHANICS_RUNG_SCHEMA = "mop-generation1-successor-mechanics-rung/v1"
MECHANICS_RESULT_SCHEMAS = ("mop-generation1-successor-mechanics-extended/v1",)

CONTINUE_DECISION = "continue_long_stress"
PRUNE_DECISION = "prune_after_canary"

WORKER_RESERVE = 4
WORKER_HARD_CEILING = 16

DEFAULT_HOST_CORES = 28
DEFAULT_MEMORY_GB = 96.0
DEFAULT_PER_CAPSULE_MEM_CAP_GB = 0.75

SECONDS_PER_HOUR = 3600.0

CLAIM_SCOPE = (
    "advisory operational telemetry only; re-derives observed throughput from sealed complete "
    "receipts to propose worker counts and de-idealized planning rates for not-yet-launched "
    "programs; changes no evidence class, seed, threshold, control, or sealed receipt"
)

_RECEIPT_ITEM_FIELDS = frozenset(
    {"index", "lane_id", "mechanism", "phase", "rung_index", "seed_start", "seed_count"}
)

_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "advisory",
        "claim_scope",
        "inputs",
        "source_results",
        "continuing_lanes",
        "pruned_lanes",
        "lane_mechanisms",
        "receipts_seen",
        "receipts_skipped",
        "observed",
        "provisional_mechanisms",
        "executed_workload",
        "recommendation",
        "complete",
        "problems",
        "activation_allowed",
        "scientific_promotion",
        "reprofile_sha256",
    }
)


class ReprofileRefused(RuntimeError):
    pass




def _seal_matches(payload: Mapping[str, Any], field: str) -> bool:
    body = {key: value for key, value in payload.items() if key != field}
    return payload.get(field) == canonical_sha256(body)


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    if resolved.is_relative_to(root):
        return str(resolved.relative_to(root))
    return str(resolved)


def _read_json_file(path: Path) -> object | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None




@dataclass(frozen=True, slots=True)
class _RungFact:
    lane_id: str
    phase: str
    mechanism: str
    rung_index: int
    seed_count: int
    mtime_ns: int


def _valid_complete_receipt(payload: object) -> dict[str, Any] | None:

    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema") != MECHANICS_RUNG_SCHEMA:
        return None
    if payload.get("complete") is not True:
        return None
    if payload.get("problems") != []:
        return None
    if payload.get("confirmation_count") != 0:
        return None
    if payload.get("activation_allowed") is not False:
        return None
    if payload.get("scientific_promotion") is not False:
        return None
    item = payload.get("item")
    if not isinstance(item, Mapping) or set(item) != set(_RECEIPT_ITEM_FIELDS):
        return None
    seed_count = item.get("seed_count")
    rung_index = item.get("rung_index")
    if not isinstance(seed_count, int) or isinstance(seed_count, bool) or seed_count <= 0:
        return None
    if not isinstance(rung_index, int) or isinstance(rung_index, bool) or rung_index < 0:
        return None
    if not isinstance(item.get("lane_id"), str) or not isinstance(item.get("mechanism"), str):
        return None
    if not isinstance(item.get("phase"), str):
        return None
    if not _seal_matches(payload, "result_sha256"):
        return None
    return dict(payload)


def _scan_receipts(runs_root: Path) -> tuple[list[_RungFact], int, int]:

    facts: list[_RungFact] = []
    seen = 0
    skipped = 0
    for path in sorted(runs_root.rglob("rung_*.json")):
        payload = _read_json_file(path)
        if not isinstance(payload, Mapping) or payload.get("schema") != MECHANICS_RUNG_SCHEMA:
            continue
        seen += 1
        receipt = _valid_complete_receipt(payload)
        if receipt is None:
            skipped += 1
            continue
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            skipped += 1
            continue
        item = receipt["item"]
        facts.append(
            _RungFact(
                lane_id=str(item["lane_id"]),
                phase=str(item["phase"]),
                mechanism=str(item["mechanism"]),
                rung_index=int(item["rung_index"]),
                seed_count=int(item["seed_count"]),
                mtime_ns=mtime_ns,
            )
        )
    return facts, seen, skipped


def _observed_from_facts(facts: Sequence[_RungFact]) -> dict[str, dict[str, Any]]:

    blocks: dict[tuple[str, str], list[_RungFact]] = {}
    receipts_by_mechanism: dict[str, int] = {}
    seeds_by_mechanism: dict[str, int] = {}
    for fact in facts:
        blocks.setdefault((fact.lane_id, fact.phase), []).append(fact)
        receipts_by_mechanism[fact.mechanism] = receipts_by_mechanism.get(fact.mechanism, 0) + 1
        seeds_by_mechanism[fact.mechanism] = seeds_by_mechanism.get(fact.mechanism, 0) + fact.seed_count

    observations: dict[str, list[float]] = {}
    for rows in blocks.values():
        rows.sort(key=lambda row: row.rung_index)
        for index in range(1, len(rows)):
            previous = rows[index - 1]
            current = rows[index]
            if current.rung_index != previous.rung_index + 1:
                continue
            delta_ns = current.mtime_ns - previous.mtime_ns
            if delta_ns <= 0:
                continue
            seconds_per_seed = delta_ns / (1e9 * current.seed_count)
            observations.setdefault(current.mechanism, []).append(seconds_per_seed)

    observed: dict[str, dict[str, Any]] = {}
    for mechanism in sorted(observations):
        values = observations[mechanism]
        observed[mechanism] = {
            "observed_seconds_per_seed": statistics.median(values),
            "sample_rungs": len(values),
            "complete_receipts": receipts_by_mechanism[mechanism],
            "total_seeds": seeds_by_mechanism[mechanism],
        }
    return observed




def _scan_results(proof_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(proof_root.glob("*.json")):
        payload = _read_json_file(path)
        if not isinstance(payload, Mapping) or payload.get("schema") not in MECHANICS_RESULT_SCHEMAS:
            continue
        if not _seal_matches(payload, "result_sha256"):
            continue
        lanes = payload.get("lanes")
        if not isinstance(lanes, Mapping):
            continue
        continuing = sorted(
            lane
            for lane, row in lanes.items()
            if isinstance(row, Mapping) and row.get("execution_decision") == CONTINUE_DECISION
        )
        pruned = sorted(
            lane
            for lane, row in lanes.items()
            if isinstance(row, Mapping) and row.get("execution_decision") == PRUNE_DECISION
        )
        lane_mechanisms = {
            lane: row.get("mechanism")
            for lane, row in sorted(lanes.items())
            if isinstance(row, Mapping) and isinstance(row.get("mechanism"), str)
        }
        results.append(
            {
                "program_id": payload.get("program_id"),
                "schema": payload.get("schema"),
                "result_sha256": payload.get("result_sha256"),
                "path": _repo_relative(path),
                "continuing_lanes": continuing,
                "pruned_lanes": pruned,
                "lane_mechanisms": lane_mechanisms,
            }
        )
    return results


def _merge_result_lanes(
    results: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str], dict[str, str]]:
    continuing: set[str] = set()
    pruned: set[str] = set()
    lane_mechanisms: dict[str, str] = {}
    for result in results:
        continuing.update(result["continuing_lanes"])
        pruned.update(result["pruned_lanes"])
        for lane, mechanism in result["lane_mechanisms"].items():
            lane_mechanisms[lane] = mechanism
    return sorted(continuing), sorted(pruned), dict(sorted(lane_mechanisms.items()))


def collect_observed_rates(runs_root: Path | str, proof_root: Path | str) -> dict[str, Any]:

    runs_root = Path(runs_root)
    proof_root = Path(proof_root)
    facts, seen, skipped = _scan_receipts(runs_root)
    observed = _observed_from_facts(facts)
    results = _scan_results(proof_root)
    continuing, pruned, lane_mechanisms = _merge_result_lanes(results)
    source_results = [
        {
            "program_id": result["program_id"],
            "schema": result["schema"],
            "result_sha256": result["result_sha256"],
            "path": result["path"],
        }
        for result in results
    ]
    return {
        "observed_rates": observed,
        "continuing_lanes": continuing,
        "pruned_lanes": pruned,
        "lane_mechanisms": lane_mechanisms,
        "source_results": source_results,
        "receipts_seen": seen,
        "receipts_skipped": skipped,
    }




def _positive(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ReprofileRefused(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ReprofileRefused(f"{label} must be a positive finite number")
    return number


def _observed_rate(value: object) -> float | None:
    if isinstance(value, Mapping):
        inner = value.get("observed_seconds_per_seed")
        return float(inner) if isinstance(inner, (int, float)) and not isinstance(inner, bool) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def recommended_worker_count(
    host_cores: int, memory_gb: float, per_capsule_mem_cap_gb: float
) -> dict[str, Any]:

    cores = _positive(host_cores, "host_cores")
    memory = _positive(memory_gb, "memory_gb")
    cap = _positive(per_capsule_mem_cap_gb, "per_capsule_mem_cap_gb")
    cores_after_reserve = int(cores) - WORKER_RESERVE
    memory_bound = math.floor(memory / cap)
    candidates = (
        ("cores_after_reserve", cores_after_reserve),
        ("memory_bound_workers", memory_bound),
        ("hard_ceiling", WORKER_HARD_CEILING),
    )
    binding_constraint, bound = min(candidates, key=lambda pair: pair[1])
    recommended = max(1, bound)
    return {
        "recommended_workers": recommended,
        "host_cores": int(cores),
        "worker_reserve": WORKER_RESERVE,
        "cores_after_reserve": cores_after_reserve,
        "memory_gb": memory,
        "per_capsule_mem_cap_gb": cap,
        "memory_bound_workers": memory_bound,
        "hard_ceiling": WORKER_HARD_CEILING,
        "binding_constraint": binding_constraint,
        "clamped_to_floor": bound < 1,
    }


def reprofile(
    planned_rates: Mapping[str, Any],
    observed: Mapping[str, Any],
    host_cores: int,
    memory_gb: float,
    per_capsule_mem_cap_gb: float,
    planned_seconds_total: float = 0.0,
) -> dict[str, Any]:

    if not isinstance(planned_rates, Mapping):
        raise ReprofileRefused("planned_rates must be a mapping")
    if not isinstance(observed, Mapping):
        raise ReprofileRefused("observed must be a mapping")
    if not isinstance(planned_seconds_total, (int, float)) or isinstance(planned_seconds_total, bool):
        raise ReprofileRefused("planned_seconds_total must be a number")
    total = float(planned_seconds_total)
    if not math.isfinite(total) or total < 0.0:
        raise ReprofileRefused("planned_seconds_total must be a nonnegative finite number")

    worker_math = recommended_worker_count(host_cores, memory_gb, per_capsule_mem_cap_gb)
    workers = worker_math["recommended_workers"]

    de_idealized: dict[str, Any] = {}
    for mechanism in sorted(set(planned_rates) | set(observed)):
        planned_value = planned_rates.get(mechanism)
        planned_rate = (
            float(planned_value)
            if isinstance(planned_value, (int, float)) and not isinstance(planned_value, bool)
            else None
        )
        observed_rate = _observed_rate(observed.get(mechanism))
        if observed_rate is not None:
            ratio = observed_rate / planned_rate if planned_rate not in (None, 0.0) else None
            de_idealized[mechanism] = {
                "planned_seconds_per_seed": planned_rate,
                "seconds_per_seed": observed_rate,
                "source": "observed",
                "observed_over_planned": ratio,
                "provisional": False,
            }
        else:
            de_idealized[mechanism] = {
                "planned_seconds_per_seed": planned_rate,
                "seconds_per_seed": planned_rate,
                "source": "planned",
                "observed_over_planned": None,
                "provisional": True,
            }

    serial_hours = total / SECONDS_PER_HOUR
    ideal_worker_hours = total / (workers * SECONDS_PER_HOUR)
    return {
        "recommended_workers": workers,
        "worker_math": worker_math,
        "de_idealized_rates": de_idealized,
        "projection": {
            "planned_seconds_total": total,
            "serial_hours": serial_hours,
            "recommended_workers": workers,
            "ideal_worker_hours": ideal_worker_hours,
        },
    }


def _normalize_observed(observed: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for mechanism in sorted(observed):
        record = observed[mechanism]
        if isinstance(record, Mapping):
            rate = _observed_rate(record)
            normalized[mechanism] = {
                "observed_seconds_per_seed": rate,
                "sample_rungs": int(record.get("sample_rungs", 0)),
                "complete_receipts": int(record.get("complete_receipts", 0)),
                "total_seeds": int(record.get("total_seeds", 0)),
            }
        else:
            normalized[mechanism] = {
                "observed_seconds_per_seed": _observed_rate(record),
                "sample_rungs": 0,
                "complete_receipts": 0,
                "total_seeds": 0,
            }
    return normalized


def _executed_workload(
    observed: Mapping[str, Any], planned_rates: Mapping[str, Any], workers: int
) -> tuple[dict[str, Any], float]:

    planned_seconds = 0.0
    de_idealized_seconds = 0.0
    for mechanism, record in observed.items():
        if not isinstance(record, Mapping):
            continue
        seeds = int(record.get("total_seeds", 0))
        observed_rate = _observed_rate(record)
        planned_value = planned_rates.get(mechanism)
        planned_rate = (
            float(planned_value)
            if isinstance(planned_value, (int, float)) and not isinstance(planned_value, bool)
            else None
        )
        if planned_rate is not None:
            planned_seconds += seeds * planned_rate
        if observed_rate is not None:
            de_idealized_seconds += seeds * observed_rate
    speedup = planned_seconds / de_idealized_seconds if de_idealized_seconds > 0.0 else None
    block = {
        "planned_seconds": planned_seconds,
        "de_idealized_seconds": de_idealized_seconds,
        "planned_serial_hours": planned_seconds / SECONDS_PER_HOUR,
        "de_idealized_serial_hours": de_idealized_seconds / SECONDS_PER_HOUR,
        "de_idealized_ideal_worker_hours": de_idealized_seconds / (workers * SECONDS_PER_HOUR),
        "planned_over_de_idealized_speedup": speedup,
    }
    return block, planned_seconds


def build_reprofile_artifact(
    observed_profile: Mapping[str, Any],
    planned_rates: Mapping[str, Any],
    *,
    host_cores: int = DEFAULT_HOST_CORES,
    memory_gb: float = DEFAULT_MEMORY_GB,
    per_capsule_mem_cap_gb: float = DEFAULT_PER_CAPSULE_MEM_CAP_GB,
    planned_seconds_total: float = 0.0,
) -> dict[str, Any]:

    if not isinstance(observed_profile, Mapping):
        raise ReprofileRefused("observed_profile must be a mapping")
    if not isinstance(planned_rates, Mapping):
        raise ReprofileRefused("planned_rates must be a mapping")

    observed = observed_profile.get("observed_rates", {})
    if not isinstance(observed, Mapping):
        raise ReprofileRefused("observed_profile.observed_rates must be a mapping")

    worker_math = recommended_worker_count(host_cores, memory_gb, per_capsule_mem_cap_gb)
    workers = worker_math["recommended_workers"]
    executed_workload, executed_planned_seconds = _executed_workload(observed, planned_rates, workers)

    total = float(planned_seconds_total)
    if total == 0.0:
        total = executed_planned_seconds

    recommendation = reprofile(
        planned_rates,
        observed,
        host_cores,
        memory_gb,
        per_capsule_mem_cap_gb,
        planned_seconds_total=total,
    )
    provisional = sorted(
        mechanism
        for mechanism, row in recommendation["de_idealized_rates"].items()
        if row["provisional"]
    )

    core = {
        "schema": SCHEMA,
        "advisory": True,
        "claim_scope": CLAIM_SCOPE,
        "inputs": {
            "host_cores": int(host_cores),
            "memory_gb": float(memory_gb),
            "per_capsule_mem_cap_gb": float(per_capsule_mem_cap_gb),
            "worker_reserve": WORKER_RESERVE,
            "worker_hard_ceiling": WORKER_HARD_CEILING,
            "planned_seconds_total": total,
        },
        "source_results": list(observed_profile.get("source_results", [])),
        "continuing_lanes": list(observed_profile.get("continuing_lanes", [])),
        "pruned_lanes": list(observed_profile.get("pruned_lanes", [])),
        "lane_mechanisms": dict(observed_profile.get("lane_mechanisms", {})),
        "receipts_seen": int(observed_profile.get("receipts_seen", 0)),
        "receipts_skipped": int(observed_profile.get("receipts_skipped", 0)),
        "observed": _normalize_observed(observed),
        "provisional_mechanisms": provisional,
        "executed_workload": executed_workload,
        "recommendation": recommendation,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "reprofile_sha256": canonical_sha256(core)}


def validate_reprofile(value: Mapping[str, Any]) -> None:

    if not isinstance(value, Mapping):
        raise ReprofileRefused("reprofile artifact must be an object")
    if set(value) != set(_ARTIFACT_FIELDS):
        missing = sorted(set(_ARTIFACT_FIELDS) - set(value))
        extra = sorted(set(value) - set(_ARTIFACT_FIELDS))
        raise ReprofileRefused(f"reprofile artifact fields drifted; missing={missing}, extra={extra}")
    if not _seal_matches(value, "reprofile_sha256"):
        raise ReprofileRefused("reprofile artifact self-seal mismatch")
    if value.get("schema") != SCHEMA:
        raise ReprofileRefused("reprofile artifact schema drifted")
    if value.get("advisory") is not True:
        raise ReprofileRefused("reprofile artifact must remain advisory")
    if value.get("complete") is not True:
        raise ReprofileRefused("reprofile artifact must be complete")
    if value.get("problems") != []:
        raise ReprofileRefused("reprofile artifact must carry no problems")
    if value.get("activation_allowed") is not False:
        raise ReprofileRefused("reprofile artifact must keep activation_allowed false")
    if value.get("scientific_promotion") is not False:
        raise ReprofileRefused("reprofile artifact must keep scientific_promotion false")




def _default_planned_rates() -> dict[str, float]:
    from mop.studies.generation1_successor_mechanics_queue import PLANNED_SECONDS_PER_SEED

    return dict(PLANNED_SECONDS_PER_SEED)


def _cmd_report(args: argparse.Namespace) -> int:
    planned_rates = _default_planned_rates()
    profile = collect_observed_rates(args.runs_root, args.proof_root)
    artifact = build_reprofile_artifact(
        profile,
        planned_rates,
        host_cores=args.host_cores,
        memory_gb=args.memory_gb,
        per_capsule_mem_cap_gb=args.mem_cap_gb,
        planned_seconds_total=args.planned_seconds_total,
    )
    validate_reprofile(artifact)
    if args.out is not None:
        atomic_write_json(Path(args.out), artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    payload = _read_json_file(Path(args.path))
    if payload is None:
        print(f"INVALID: unreadable or malformed JSON at {args.path}")
        return 1
    try:
        validate_reprofile(payload)
    except ReprofileRefused as exc:
        print(f"INVALID: {exc}")
        return 1
    print("VALID")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mop.studio.generation1_result_aware_reprofiler",
        description="Advisory result-aware auto-tuner for not-yet-launched Generation 1 programs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    report = subparsers.add_parser("report", help="build and print the advisory reprofile artifact")
    report.add_argument("--runs-root", default=str(REPO_ROOT / "runs/generation1"))
    report.add_argument("--proof-root", default=str(REPO_ROOT / "proof"))
    report.add_argument("--out", default=None, help="optional path to seal the artifact to")
    report.add_argument("--host-cores", type=int, default=DEFAULT_HOST_CORES)
    report.add_argument("--memory-gb", type=float, default=DEFAULT_MEMORY_GB)
    report.add_argument("--mem-cap-gb", type=float, default=DEFAULT_PER_CAPSULE_MEM_CAP_GB)
    report.add_argument("--planned-seconds-total", type=float, default=0.0)
    report.set_defaults(func=_cmd_report)

    validate = subparsers.add_parser("validate", help="fail-closed validate a sealed reprofile artifact")
    validate.add_argument("--path", required=True)
    validate.set_defaults(func=_cmd_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.func
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
