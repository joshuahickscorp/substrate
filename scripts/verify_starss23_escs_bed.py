#!/usr/bin/env python3
"""Independent verifier entrypoint: re-score the sealed STARSS23 ESCS artifact from specification.

This drives the separately authored verifier (``mop.beds.starss23.verifier``), which imports none of the
producer's code. It reads ``proof/STARSS23_ESCS_BED.json``, re-implements the referee, the sign-flip
statistics, and the canonical seal from the written spec, and writes
``proof/STARSS23_ESCS_BED.verification.json``. On synthetic fixtures the independent referee reproduction
can pass while ``independent_scientific_confirmation`` stays false, which is the whole point: mechanics
reproduce, science does not promote.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mop.beds.starss23.verifier import verify_sealed_file, write_verification  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    artifact_path = Path(args[0]) if args else REPO_ROOT / "proof" / "STARSS23_ESCS_BED.json"
    out_path = (
        Path(args[1]) if len(args) > 1 else REPO_ROOT / "proof" / "STARSS23_ESCS_BED.verification.json"
    )
    payload = verify_sealed_file(str(artifact_path))
    write_verification(payload, str(out_path))
    print(f"verified {artifact_path}")
    print(f"wrote {out_path}")
    print(
        f"independent_referee_reproduction={payload['independent_referee_reproduction']} "
        f"independent_scientific_confirmation={payload['independent_scientific_confirmation']} "
        f"source_kind={payload['source_kind']}"
    )
    if payload["mismatches"]:
        print("mismatches:")
        for mismatch in payload["mismatches"]:
            print(f"  - {mismatch}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
