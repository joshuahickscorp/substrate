"""Studio-native lane manifest.

The Studio audit adds lanes the laptop could not honestly expose: predictor rollouts, hosted real
corpora, full-width perspectives, live-encoder loops, developmental PR9 streams, and the Process C
doctrine decision. This module makes those lanes machine-readable without launching science on the
laptop. Runnable lanes get concrete commands. Blocked lanes carry the release condition as a receipt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .long_run import SCHEMA as DAEMON_SCHEMA
from .profiles import get_profile

SCHEMA = "mop-studio-native-lanes/v1"


@dataclass(frozen=True)
class NativeLane:
    """One Studio-native lane or preregistered wall condition."""

    lane_id: str
    facet: str
    title: str
    priority: int
    kind: str
    command_template: tuple[str, ...] | None
    null_hypothesis: str
    success_gate: str
    wall_if_blocked: str
    required_inputs: tuple[str, ...] = ()
    profile: str = "studio-m1ultra"
    heavy: bool = False
    default_plan: bool = True
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.lane_id,
            "facet": self.facet,
            "title": self.title,
            "priority": self.priority,
            "kind": self.kind,
            "profile": self.profile,
            "heavy": self.heavy,
            "default_plan": self.default_plan,
            "required_inputs": list(self.required_inputs),
            "null_hypothesis": self.null_hypothesis,
            "success_gate": self.success_gate,
            "wall_if_blocked": self.wall_if_blocked,
            "notes": self.notes,
        }


LANES: tuple[NativeLane, ...] = (
    NativeLane(
        lane_id="dr13_predictor_fidelity_real",
        facet="12",
        title="real V-JEPA predictor rollout-fidelity gate",
        priority=1,
        kind="gate",
        heavy=True,
        required_inputs=("clip_dir",),
        command_template=(
            "python",
            "scripts/mop_dr13_predictor_fidelity.py",
            "--n-clips",
            "64",
            "--clip-dir",
            "{clip_dir}",
            "--out",
            "runs/mot/dr13_predictor_fidelity_real.json",
        ),
        null_hypothesis="real predictor rollouts fail to reach a usable contiguous horizon >= 2",
        success_gate="usable horizon >= 2 by the preregistered non-overlap and usability-fraction rule",
        wall_if_blocked="requires a real moving-video .pt clip directory from DR1 or hosted corpora",
        notes="The single Part 2 lane licensed before the spine because it gates rollout use.",
    ),
    NativeLane(
        lane_id="dr13_readout_adapter_real",
        facet="12",
        title="real rollout readout-adapter recovery test",
        priority=2,
        kind="gate",
        heavy=True,
        required_inputs=("clip_dir",),
        command_template=(
            "python",
            "scripts/mop_dr13_readout_adapter.py",
            "--n-clips",
            "64",
            "--clip-dir",
            "{clip_dir}",
            "--out",
            "runs/mot/dr13_readout_adapter_real.json",
        ),
        null_hypothesis="the adapter ties raw rollout or remains above 0.5*persistence at every horizon",
        success_gate="adapted rollout beats raw by seed CI and clears the absolute persistence fraction",
        wall_if_blocked="requires the same real moving-video .pt clip directory as DR13 fidelity",
        notes="Runs after the predictor-fidelity gate when rollout signal is directional but shifted.",
    ),
    NativeLane(
        lane_id="hosted_corpora_plan",
        facet="14",
        title="hosted real-corpora acquisition plan",
        priority=3,
        kind="plan",
        command_template=(
            "python",
            "scripts/studio_pipeline.py",
            "plan",
            "--profile",
            "studio-m1ultra",
            "--budget-gb",
            "4000",
            "--accept-license",
            "--label",
            "studio_native_corpora",
        ),
        null_hypothesis="license, source quality, or budget blockers prevent a real hosted-corpora plan",
        success_gate="plan selects licensed sources inside the studio-m1ultra disk and source caps",
        wall_if_blocked="requires the studio-m1ultra profile and current registry metadata",
        notes="Planning is safe and dry; acquisition remains a separate heavy lane.",
    ),
    NativeLane(
        lane_id="hosted_corpora_acquire",
        facet="14",
        title="hosted real-corpora acquisition execution",
        priority=4,
        kind="acquire",
        heavy=True,
        default_plan=False,
        required_inputs=("plan_path",),
        command_template=(
            "python",
            "scripts/studio_pipeline.py",
            "acquire",
            "--plan",
            "{plan_path}",
            "--profile",
            "studio-m1ultra",
            "--budget-gb",
            "4000",
            "--accept-license",
            "--execute",
        ),
        null_hypothesis="licensed sources cannot be acquired or validated inside the disk budget",
        success_gate="selected sources land with data cards and validation receipts",
        wall_if_blocked="requires manual license access plus an inspected hosted-corpora plan",
        notes="Not emitted by default because it may download large external data.",
    ),
    NativeLane(
        lane_id="live_encoder_rederive",
        facet="13",
        title="live-encoder doctrine re-derivation",
        priority=5,
        kind="doctrine-gate",
        required_inputs=("encode_schedule",),
        command_template=None,
        null_hypothesis="measured live encode remains too slow or unstable to retire the cached-only ban",
        success_gate="Wave 0 encode schedule shows a real winner with enough s/clip headroom for live arms",
        wall_if_blocked="requires Wave 0 encode_device.json and encode_schedule.json from the M1 Ultra",
        notes=(
            "This is a decision gate, not a science run. The live arm is licensed only after measured s/clip."
        ),
    ),
    NativeLane(
        lane_id="perspective_ecology_audit",
        facet="15",
        title="full-width perspective ecology audit",
        priority=6,
        kind="audit",
        command_template=None,
        null_hypothesis=(
            "new perspectives lack matched controls or A6 residualization and cannot support a claim"
        ),
        success_gate=(
            "all perspective arms align on referent ids and carry matched controls plus provenance flags"
        ),
        wall_if_blocked=(
            "requires DR1 merged cache plus language, audio, depth, flow, SAM, code, and math features"
        ),
        notes="PerspectiveAdapter is ready; extraction scripts must point it at DR1 referents first.",
    ),
    NativeLane(
        lane_id="pr9_long_stream",
        facet="16",
        title="developmental long-stream PR9 under daemon",
        priority=7,
        kind="long-run",
        heavy=True,
        required_inputs=("dr1_cache",),
        command_template=(
            "python",
            "scripts/studio/pr9_continual_backprop.py",
            "--cache",
            "{dr1_cache}",
            "--seeds",
            "0-9",
            "--out",
            "runs/mot/pr9_continual_backprop.json",
        ),
        null_hypothesis="the well-tuned plain baseline shows no certified plasticity loss, or CBP ties it",
        success_gate="certificate fires first, then CBP wins without retention cost and with no sign flip",
        wall_if_blocked="requires the DR1 merged real-latent cache and a free encoder lane",
        notes="PR9 stays second after DR1 and decides whether Process C is licensed.",
    ),
    NativeLane(
        lane_id="process_c_dense_token_decision",
        facet="17",
        title="Process C dense-token doctrine decision",
        priority=8,
        kind="decision",
        command_template=None,
        null_hypothesis="Process C remains unlicensed unless PR9 or DR1 proves the frozen-substrate wall",
        success_gate=(
            "PR9 kill-switch or DR1 representational wall licenses only the 1 to 10M dense-token pilot"
        ),
        wall_if_blocked="requires PR9 tie on a loss-inducing stream or a DR1 representational wall proof",
        notes="The dense-token module exists; a launcher is intentionally not emitted before licensing.",
    ),
)


def lane_by_id() -> dict[str, NativeLane]:
    return {lane.lane_id: lane for lane in LANES}


def build_native_lane_manifest(
    *,
    profile_name: str = "studio-m1ultra",
    include_heavy: bool = False,
    lane_ids: list[str] | None = None,
    inputs: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Evaluate native lanes under a profile without running them."""
    profile = get_profile(profile_name)
    selected = _select_lanes(lane_ids)
    variables = {k: v for k, v in (inputs or {}).items() if v}
    evaluated = [_evaluate_lane(lane, profile.name, variables, include_heavy) for lane in selected]
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": profile.as_dict(),
        "include_heavy": bool(include_heavy),
        "inputs": variables,
        "lanes": evaluated,
        "summary": {
            "total": len(evaluated),
            "ready": sum(1 for lane in evaluated if lane["status"] == "ready"),
            "blocked": sum(1 for lane in evaluated if lane["status"] == "blocked"),
        },
    }


def write_native_manifest(manifest: dict[str, Any], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, default=str) + "\n")


def write_native_daemon_plan(manifest: dict[str, Any], path: Path | str) -> dict[str, Any]:
    """Write a long-run daemon plan from ready native lanes only."""
    jobs = [
        {
            "id": lane["id"],
            "cmd": lane["command"],
            "kind": lane["kind"],
            "notes": lane["success_gate"],
        }
        for lane in manifest["lanes"]
        if lane["status"] == "ready"
    ]
    if not jobs:
        raise ValueError("no ready native lanes to write into a daemon plan")
    plan = {
        "schema": DAEMON_SCHEMA,
        "native_schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": manifest["profile"]["name"],
        "inputs": manifest.get("inputs", {}),
        "jobs": jobs,
        "blocked_lanes": [
            {
                "id": lane["id"],
                "facet": lane["facet"],
                "blocked_reason": lane["blocked_reason"],
                "wall_if_blocked": lane["wall_if_blocked"],
            }
            for lane in manifest["lanes"]
            if lane["status"] == "blocked"
        ],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan, indent=2, default=str) + "\n")
    return plan


def _select_lanes(lane_ids: list[str] | None) -> list[NativeLane]:
    if not lane_ids:
        return list(LANES)
    known = lane_by_id()
    missing = sorted(set(lane_ids).difference(known))
    if missing:
        raise ValueError(f"unknown native lane ids: {missing}; choose from {sorted(known)}")
    return [known[lane_id] for lane_id in lane_ids]


def _evaluate_lane(
    lane: NativeLane,
    profile_name: str,
    inputs: dict[str, str],
    include_heavy: bool,
) -> dict[str, Any]:
    rec = lane.as_dict()
    rec["status"] = "blocked"
    rec["command"] = None
    rec["blocked_reason"] = ""
    if profile_name != lane.profile:
        rec["blocked_reason"] = f"profile {profile_name!r} does not match required {lane.profile!r}"
        return rec
    if lane.heavy and not include_heavy:
        rec["blocked_reason"] = "heavy lane omitted unless --include-heavy is set after preregistration"
        return rec
    missing = [key for key in lane.required_inputs if not inputs.get(key)]
    if missing:
        rec["blocked_reason"] = f"missing required input(s): {', '.join(missing)}"
        return rec
    if lane.command_template is None:
        rec["blocked_reason"] = lane.wall_if_blocked
        return rec
    rec["status"] = "ready"
    rec["command"] = [part.format(**inputs) for part in lane.command_template]
    return rec
