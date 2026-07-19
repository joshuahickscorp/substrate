#!/usr/bin/env python3
"""Producer entrypoint: run the STARSS23 ESCS event-formation bed and emit the sealed proof artifact.

Two lanes share this entrypoint. Without ``--real`` it runs the synthetic STARSS23 fixtures (five paired
seeds, the frozen featurizer, the one trained candidate gate, and the three controls) and writes the
byte-sealed ``proof/STARSS23_ESCS_BED.json``; synthetic data caps the verdict at mechanics-ok. With
``--real`` it runs the real, MIT-licensed STARSS23 FOA subset through ``RealStarssAdapter``: it first
writes the self-sealed preregistration ``proof/STARSS23_ESCS_BED.prereg.json`` (the cost-benefit SESOI,
fixed before any test score is read), then runs the bed and writes the same sealed artifact with
``source_kind=real`` and ``rights_clean=true``. Either lane hardcodes activation_allowed=false,
scientific_promotion=false, and independent_scientific_confirmation=false.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# A fixed default preregistration timestamp so the sealed prereg body is byte-reproducible across re-runs.
# The wall clock is never read inside a sealed body; the timestamp is always passed in.
DEFAULT_PREREG_TIMESTAMP = "2026-07-17T00:00:00Z"


def _run_synthetic(out_path: Path) -> int:
    from mop.beds.starss23.artifact import BedConfig, build_bed_artifact
    from mop.substrate.events import write_canonical_json

    bed = build_bed_artifact(BedConfig())
    path = write_canonical_json(bed.artifact, out_path)
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


def _run_real(out_path: Path, foa_root: Path, metadata_root: Path, timestamp: str) -> int:
    from mop.beds.starss23.real_artifact import (
        DEFAULT_FOA_ROOT,
        DEFAULT_METADATA_ROOT,
        build_real_bed_artifact,
    )
    from mop.substrate.events import write_canonical_json

    prereg_path = REPO_ROOT / "proof" / "STARSS23_ESCS_BED.prereg.json"
    bed = build_real_bed_artifact(
        timestamp=timestamp,
        foa_root=foa_root or DEFAULT_FOA_ROOT,
        metadata_root=metadata_root or DEFAULT_METADATA_ROOT,
        prereg_path=prereg_path,
    )
    path = write_canonical_json(bed.artifact, out_path)
    print(f"wrote prereg {prereg_path}")
    print(f"wrote {path}")
    print(
        f"source_kind=real rights_clean=true verdict={bed.verdict} seal={bed.seal} "
        f"dominates={bed.detail['dominates']} mean_delta={bed.detail['mean_delta']:.6f} "
        f"one_sided_p={bed.detail['one_sided_p']} "
        f"sesoi_f1={bed.detail['sesoi_f1']} exceeds_sesoi={bed.detail['mean_delta_exceeds_sesoi']} "
        f"noisy_tv_at_chance={bed.detail['noisy_tv_at_chance']}"
    )
    print(f"per_seed_deltas={bed.detail['per_seed_deltas']}")
    print(
        "flags: activation_allowed=false scientific_promotion=false "
        "independent_scientific_confirmation=false (one real run; mechanics demonstration only)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the STARSS23 ESCS event-formation bed.")
    parser.add_argument("out_path", nargs="?", default=None, help="sealed artifact output path")
    parser.add_argument("--real", action="store_true", help="run the real STARSS23 FOA subset")
    parser.add_argument("--foa", default=None, help="FOA audio root (real lane)")
    parser.add_argument("--metadata", default=None, help="metadata root (real lane)")
    parser.add_argument("--timestamp", default=DEFAULT_PREREG_TIMESTAMP, help="fixed prereg timestamp")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    out_path = Path(args.out_path) if args.out_path else REPO_ROOT / "proof" / "STARSS23_ESCS_BED.json"
    if args.real:
        return _run_real(
            out_path,
            Path(args.foa) if args.foa else None,
            Path(args.metadata) if args.metadata else None,
            args.timestamp,
        )
    return _run_synthetic(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
