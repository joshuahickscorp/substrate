#!/usr/bin/env python3
"""Producer entrypoint: run the STARSS23 ESCS event-formation bed and emit the sealed proof artifact.

Runs the whole bed on the synthetic STARSS23 fixtures (five paired seeds, the frozen featurizer, the one
trained candidate gate, and the three controls) and writes the byte-sealed
``proof/STARSS23_ESCS_BED.json``. Synthetic data caps the verdict at mechanics-ok; the artifact hardcodes
activation_allowed=false, scientific_promotion=false, and independent_scientific_confirmation=false.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mop.beds.starss23.artifact import BedConfig, build_bed_artifact, write_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    out_path = Path(args[0]) if args else REPO_ROOT / "proof" / "STARSS23_ESCS_BED.json"
    bed = build_bed_artifact(BedConfig())
    path = write_artifact(bed.artifact, out_path)
    print(f"wrote {path}")
    print(
        f"verdict={bed.verdict} "
        f"seal={bed.seal} "
        f"dominates={bed.detail['dominates']} "
        f"one_sided_p={bed.detail['one_sided_p']} "
        f"noisy_tv_at_chance={bed.detail['noisy_tv_at_chance']}"
    )
    print(
        "flags: activation_allowed=false scientific_promotion=false "
        "independent_scientific_confirmation=false (synthetic; mechanics only)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
