#!/usr/bin/env python
"""Run or resume the P5 exact-versus-factorized context capability pilot.

Profiles are wall envelopes, not claim levels. ``p5smoke`` proves the plumbing end to end with one
seed and a twelve-step dense reference; ``p5pilot`` is the staged five-seed pilot (seed 0 across
all frame counts, an f64 trainability gate, then later seeds only on off-ceiling frame counts).
A stopped run is resumable from content-addressed per-arm checkpoints and per-seed receipts by
rerunning the exact same command. The proof artifact is written atomically and only when the pilot
completes; the promotion block refuses confirmatory claims by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from mop.config import REPO_ROOT, compose
from mop.substrate.custom_workbench import _atomic_json
from mop.substrate.p5_context import run_p5_pilot

PROFILES = ("p5smoke", "p5pilot")

# One heavy local lane at a time: refuse to start while any of these is alive elsewhere.
HEAVY_PROCESS_MARKERS = (
    "cache_factorized_encoder.py",
    "cache_real_encoder.py",
    "custom_substrate_workbench.py",
    "p4_capability_density.py",
    "p5_context_capability.py",
)


def assert_heavy_lane_free(markers: tuple[str, ...] = HEAVY_PROCESS_MARKERS) -> None:
    """Refuse to start if another known heavy process is alive. Own pid and parent pid are
    excluded (the launching shell's command line echoes this script's name)."""

    mine = {os.getpid(), os.getppid()}
    listing = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True)
    offenders: list[str] = []
    for line in listing.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid in mine:
            continue
        if any(marker in command for marker in markers):
            offenders.append(f"pid {pid}: {command.strip()}")
    if offenders:
        raise SystemExit(
            "heavy lane BUSY, refusing to start: " + "; ".join(offenders) + ". One heavy local "
            "process at a time; rerun after the in-flight run exits."
        )


def _config(profile: str) -> dict:
    cfg = compose(["experiment=mop_p5_context_capability"])
    value = OmegaConf.to_container(cfg.experiment, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved P5 context pilot config is not a mapping")
    profiles = value.pop("profiles", None)
    if not isinstance(profiles, dict) or profile not in profiles:
        raise SystemExit(f"profile {profile!r} is missing from the P5 context pilot config")
    value["training"] = {**value["training"], **profiles[profile]}
    value["profile"] = profile
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="p5pilot")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    assert_heavy_lane_free()
    config = _config(args.profile)
    run_dir = args.run_dir or REPO_ROOT / "runs/p5_context" / args.profile
    receipt = run_p5_pilot(config, run_dir, args.device)
    out = args.out or REPO_ROOT / "proof/P5_CONTEXT_CAPABILITY_PILOT.json"
    if receipt["complete"]:
        _atomic_json(out, receipt)
    print(
        json.dumps(
            {
                "profile": args.profile,
                "complete": receipt["complete"],
                "resumable": receipt["resumable"],
                "all_ok": receipt["all_ok"],
                "wall_seconds": receipt["resource_telemetry"]["wall_seconds_this_invocation"],
                "frames_complete": sum(1 for row in receipt["frames"].values() if row["complete"]),
                "trainability_gate_failed": receipt["trainability_gate_failed"],
                "confirmatory_promotable": receipt["promotion"]["confirmatory_promotable"],
                "proof": str(out) if receipt["complete"] else None,
            },
            indent=2,
        )
    )
    return 0 if receipt["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
