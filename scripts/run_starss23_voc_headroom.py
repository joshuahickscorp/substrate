"""Run the STARSS23 value-of-computation HEADROOM instrument on the real subset and seal its artifacts.

Additive-only: writes only proof/STARSS23_VOC_HEADROOM.json, proof/STARSS23_VOC_HEADROOM.prereg.json, and
proof/STARSS23_VOC_HEADROOM.verification.json. Edits no sealed bed module and touches no live campaign
path. House style: no em dashes and no en dashes.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/run_starss23_voc_headroom.py --timestamp 2026-07-18T00:00Z
"""

from __future__ import annotations

import argparse
import sys

from mop.beds.starss23.vochead_prereg import build_vochead_prereg, write_vochead_prereg
from mop.beds.starss23.vochead_producer import build_vochead_artifact, write_vochead_artifact
from mop.beds.starss23.vochead_verifier import verify_vochead_artifact, write_vochead_verification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seal the STARSS23 value-of-computation headroom bed.")
    parser.add_argument(
        "--timestamp",
        required=True,
        help="an explicit ISO-8601 timestamp for the preregistration (never read from a wall clock)",
    )
    args = parser.parse_args(argv)

    prereg = build_vochead_prereg(timestamp=args.timestamp)
    prereg_path = write_vochead_prereg(prereg)
    print(f"prereg sealed -> {prereg_path}")

    artifact = build_vochead_artifact()
    artifact_path = write_vochead_artifact(artifact)
    print(f"artifact sealed -> {artifact_path}")

    verification = verify_vochead_artifact(artifact)
    verification_path = write_vochead_verification(verification)
    print(f"verification sealed -> {verification_path}")

    head = artifact["headline"]
    print("\nheadline:")
    for key, value in head.items():
        print(f"  {key}: {value}")
    print(f"\nsynthetic_control_ok: {artifact['synthetic_control_ok']}")
    print("verification:")
    for key in (
        "seal_intact",
        "targets_reproduced",
        "interpretation_reproduced",
        "honesty_ok",
        "independent_referee_reproduction",
        "independent_scientific_confirmation",
    ):
        print(f"  {key}: {verification.get(key)}")
    if verification.get("mismatches"):
        print("MISMATCHES:", verification["mismatches"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
