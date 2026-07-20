#!/usr/bin/env python3

from __future__ import annotations

import copy
import json

from mop.config import REPO_ROOT
from mop.studio.task_policy_authority import (
    canonical_sha256,
    policy_baseline_manifest_problems,
)

SOURCE = REPO_ROOT / "proof/LOCAL_THROTTLE_POLICY_BASELINE_V1.json"
TARGETS = {
    REPO_ROOT / "proof/LOCAL_THROTTLE_POLICY_BASELINE_V0.json": (
        "73ffca97b312bdb7971bcfffb441fb4b204a2a26f8c9964a50e4e7debe00f3f7"
    ),
    REPO_ROOT / "proof/LOCAL_THROTTLE_POLICY_BASELINE_V2.json": (
        "a1a8d4e3d6ca23d50808e6657eaa6c68eadffa4d8f29d81f98da49b4eb014d40"
    ),
}


def _render(source: dict[str, object], governor_sha256: str) -> dict[str, object]:
    payload = copy.deepcopy(source)
    implementation = payload.get("governor_implementation")
    if not isinstance(implementation, dict):
        raise ValueError("baseline governor implementation binding is missing")
    implementation["sha256"] = governor_sha256
    core = dict(payload)
    core.pop("manifest_sha256", None)
    payload["manifest_sha256"] = canonical_sha256(core)
    problems = policy_baseline_manifest_problems(payload)
    if problems:
        raise ValueError(f"rendered legacy baseline is invalid: {problems}")
    return payload


def main() -> int:
    source = json.loads(SOURCE.read_text())
    problems = policy_baseline_manifest_problems(source)
    if problems:
        raise ValueError(f"source policy baseline is invalid: {problems}")
    for path, governor_sha256 in TARGETS.items():
        payload = _render(source, governor_sha256)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
