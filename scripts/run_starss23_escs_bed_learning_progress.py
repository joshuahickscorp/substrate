#!/usr/bin/env python3
"""Producer entrypoint: run the STARSS23 ESCS "learning_progress" gate variant on the cached real corpus.

This runs the E1 learning_progress gate variant through the EXISTING sealed harness, referee, controls,
and statistics on the shared frozen-featurizer cache (no re-featurization). It first writes the
self-sealed preregistration sidecar ``proof/STARSS23_ESCS_BED_learning_progress.prereg.json`` (the SESOI
and sign-flip plan imported unchanged from the sealed prereg, fixed before any test score is read), then
runs the bed and writes the sealed artifact ``proof/STARSS23_ESCS_BED_learning_progress.json`` with
``source_kind=real`` and ``rights_clean=true``. It hardcodes activation_allowed=false,
scientific_promotion=false, and independent_scientific_confirmation=false.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# A fixed default preregistration timestamp so the sealed bodies are byte-reproducible across re-runs.
DEFAULT_PREREG_TIMESTAMP = "2026-07-17T00:00:00Z"


def main(argv: list[str] | None = None) -> int:
    from mop.beds.starss23.learning_progress_producer import (
        DEFAULT_ARTIFACT_PATH,
        DEFAULT_PREREG_PATH,
        build_lp_bed_artifact,
    )
    from mop.substrate.events import write_canonical_json

    parser = argparse.ArgumentParser(
        description="Run the STARSS23 ESCS learning_progress gate variant on the cached real corpus."
    )
    parser.add_argument("out_path", nargs="?", default=None, help="sealed artifact output path")
    parser.add_argument("--cache-root", default=None, help="feature cache root")
    parser.add_argument("--foa", default=None, help="FOA audio root (for cache keying)")
    parser.add_argument("--metadata", default=None, help="metadata root (for cache keying)")
    parser.add_argument("--timestamp", default=DEFAULT_PREREG_TIMESTAMP, help="fixed prereg timestamp")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    from mop.beds.starss23.feature_cache import DEFAULT_CACHE_ROOT
    from mop.beds.starss23.real_artifact import DEFAULT_FOA_ROOT, DEFAULT_METADATA_ROOT

    out_path = Path(args.out_path) if args.out_path else REPO_ROOT / DEFAULT_ARTIFACT_PATH
    prereg_path = REPO_ROOT / DEFAULT_PREREG_PATH

    bed = build_lp_bed_artifact(
        timestamp=args.timestamp,
        cache_root=args.cache_root or DEFAULT_CACHE_ROOT,
        foa_root=args.foa or DEFAULT_FOA_ROOT,
        metadata_root=args.metadata or DEFAULT_METADATA_ROOT,
        prereg_path=prereg_path,
    )
    path = write_canonical_json(bed.artifact, out_path)
    print(f"wrote prereg {prereg_path}")
    print(f"wrote {path}")
    print(
        f"variant=learning_progress source_kind=real rights_clean=true verdict={bed.verdict} "
        f"seal={bed.seal}"
    )
    print(
        f"beats_random={bed.detail['beats_random']} dominates={bed.detail['dominates']} "
        f"mean_delta={bed.detail['mean_delta']:.6f} one_sided_p={bed.detail['one_sided_p']} "
        f"sesoi_f1={bed.detail['sesoi_f1']} exceeds_sesoi={bed.detail['mean_delta_exceeds_sesoi']} "
        f"noisy_tv_at_chance={bed.detail['noisy_tv_at_chance']}"
    )
    print(f"per_seed_deltas={bed.detail['per_seed_deltas']}")
    print(
        f"adjacency_fraction={bed.detail['pooled_adjacency_fraction']} "
        f"candidate_tp={bed.detail['pooled_candidate_distinct_onset_tp']} "
        f"rate_matched_random_tp={bed.detail['pooled_rate_matched_random_distinct_onset_tp']} "
        f"(committed baseline 204 / random 237)"
    )
    print(
        "flags: activation_allowed=false scientific_promotion=false "
        "independent_scientific_confirmation=false (one real run; exploratory variant; mechanics only)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
