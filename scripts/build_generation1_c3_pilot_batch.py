#!/usr/bin/env python3
"""Consolidate the executed G1-C3 pilot lanes into one sealed batch receipt."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from mop.config import REPO_ROOT
from mop.studies.generation1_c3_communication import validate_result as validate_communication
from mop.studies.generation1_c3_construction import validate_pilot_result
from mop.studies.generation1_c3_dispatch import validate_result as validate_dispatch

SCHEMA = "mop-generation1-c3-pilot-batch/v1"
OUTPUT = REPO_ROOT / "proof/GENERATION1_C3_PILOT_BATCH.json"
LANES = {
    "G1-D1": "runs/generation1/generation1-c3-d1-learned-dispatch-pilot-v1/result.json",
    "G1-V1+M1": "proof/GENERATION1_C3_COMMUNICATION_PILOT.json",
    "G1-G1": "proof/GENERATION1_C3_CONSTRUCTION_PILOT.json",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"pilot result must be an object: {path}")
    return value


def build() -> dict[str, Any]:
    dispatch_path = REPO_ROOT / LANES["G1-D1"]
    communication_path = REPO_ROOT / LANES["G1-V1+M1"]
    construction_path = REPO_ROOT / LANES["G1-G1"]
    dispatch = _load(dispatch_path)
    communication = _load(communication_path)
    construction = _load(construction_path)
    validate_dispatch(dispatch, dispatch["config"])
    validate_communication(communication)
    validate_pilot_result(construction)
    core = {
        "schema": SCHEMA,
        "campaign_id": "generation1-c3-successor-mechanisms-pilot-v1",
        "claim_scope": "generated successor mechanics and leakage canaries only",
        "lanes": {
            "G1-D1": {
                "path": LANES["G1-D1"],
                "file_sha256": _file_sha(dispatch_path),
                "result_sha256": dispatch["result_sha256"],
                "complete": dispatch["complete"],
                "scientific_confirmation": False,
                "ready_for_confirmatory_claim": dispatch["decision"]["ready_for_confirmatory_claim"],
            },
            "G1-V1+M1": {
                "path": LANES["G1-V1+M1"],
                "file_sha256": _file_sha(communication_path),
                "result_sha256": communication["result_sha256"],
                "complete": communication["execution_complete"],
                "scientific_confirmation": communication["decision"]["scientific_confirmation"],
                "supports_real_producer_implementation": communication["decision"][
                    "supports_real_producer_implementation"
                ],
            },
            "G1-G1": {
                "path": LANES["G1-G1"],
                "file_sha256": _file_sha(construction_path),
                "result_sha256": construction["result_sha256"],
                "complete": True,
                "scientific_confirmation": False,
                "verdict": construction["decision"]["verdict"],
            },
        },
        "decision": {
            "all_pilot_lanes_executed": True,
            "c3_confirmed": False,
            "next_action": "implement and preregister confirmatory producers plus independent verifiers",
            "interpretation": (
                "D1 passed its leakage and execution canary but did not beat both static controls. "
                "V1, M1, and G1 discriminate engineered regimes only. No lane earns activation."
            ),
        },
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "batch_result_sha256": _sha(core)}


def validate(value: dict[str, Any]) -> None:
    if value.get("schema") != SCHEMA:
        raise ValueError("C3 pilot batch schema drifted")
    expected = _sha({key: item for key, item in value.items() if key != "batch_result_sha256"})
    if value.get("batch_result_sha256") != expected:
        raise ValueError("C3 pilot batch seal drifted")
    if value.get("activation_allowed") is not False or value.get("scientific_promotion") is not False:
        raise ValueError("C3 pilot batch escaped activation or promotion")
    if value.get("decision", {}).get("c3_confirmed") is not False:
        raise ValueError("C3 pilot batch cannot claim confirmation")


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    result = build()
    validate(result)
    _write(OUTPUT, result)
    print(
        json.dumps(
            {"path": str(OUTPUT), **result["decision"], "seal": result["batch_result_sha256"]}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
