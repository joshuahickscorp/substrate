from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from mop.studio.local_throttle import load_policy
from mop.studio.profiles import get_profile
from mop.studio.task_policy_authority import (
    build_policy_safety_contract,
    policy_baseline_manifest_problems,
    task_policy_authority_problems,
)

BASELINES = {
    Path("proof/LOCAL_THROTTLE_POLICY_BASELINE_V0.json"): (
        "73ffca97b312bdb7971bcfffb441fb4b204a2a26f8c9964a50e4e7debe00f3f7"
    ),
    Path("proof/LOCAL_THROTTLE_POLICY_BASELINE_V1.json"): (
        "bd7dd790460adc7760620007c691e3c345e89d9630c303abf40de59b924fddfb"
    ),
    Path("proof/LOCAL_THROTTLE_POLICY_BASELINE_V2.json"): (
        "a1a8d4e3d6ca23d50808e6657eaa6c68eadffa4d8f29d81f98da49b4eb014d40"
    ),
}


@pytest.mark.parametrize(("baseline_path", "governor_sha256"), BASELINES.items())
def test_historical_policy_baseline_is_intrinsically_valid_and_currently_compatible(
    baseline_path: Path,
    governor_sha256: str,
) -> None:
    manifest = json.loads(baseline_path.read_text())
    policy = load_policy()
    safety = build_policy_safety_contract(
        profile=get_profile(policy.profile_name).as_dict(),
        limits=policy.limits,
        monitor=policy.monitor,
        thresholds=policy.thresholds,
    )

    assert policy_baseline_manifest_problems(manifest) == []
    assert manifest["scientific_promotion"] is False
    assert manifest["policy"]["sha256"] == "d2d113bf77daabe977515049e226d20b5333dac2597888ae756d5fd5908dd685"
    assert manifest["governor_implementation"]["sha256"] == governor_sha256
    for authority in manifest["task_authorities"]:
        task = policy.task(authority["task_id"])
        assert (
            task_policy_authority_problems(
                authority,
                policy_schema="mop-local-execution-throttle-policy/v1",
                policy_path=str(policy.path),
                full_policy_sha256=policy.sha256,
                profile_name=policy.profile_name,
                safety_contract=safety,
                foreground_markers=policy.monitor["foreground_markers"],
                known_heavy_markers=policy.monitor["known_heavy_markers"],
                task_id=task.task_id,
                task_payload=json.loads(json.dumps(asdict(task), sort_keys=True)),
            )
            == []
        )
