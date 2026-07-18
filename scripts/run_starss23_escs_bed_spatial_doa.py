#!/usr/bin/env python3
"""Producer entrypoint: run the STARSS23 ESCS frozen featurizer "spatial_doa" and seal its proof artifact.

This runs the spatial-DOA active-intensity direction-of-arrival featurizer on the REAL, MIT-licensed
STARSS23 FOA subset, featurizing once into its own feature cache. It scores the swap through the same
sealed harness, referee, controls, and sign-flip statistics as the committed real run, reads the SESOI
from the already-sealed featurizer preregistration (never rebuilding it), and writes the sealed
``proof/STARSS23_ESCS_BED_spatial_doa.json`` with the same shape as the committed artifact
(activation_allowed=false, scientific_promotion=false, independent_scientific_confirmation=false,
source_kind=real, rights_clean=true).

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# A fixed default timestamp so the sealed body is byte-reproducible; the wall clock is never sealed.
DEFAULT_TIMESTAMP = "2026-07-17T00:00:00Z"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the STARSS23 ESCS spatial_doa frozen featurizer on the real corpus."
    )
    parser.add_argument("out_path", nargs="?", default=None, help="sealed artifact output path")
    parser.add_argument("--cache-root", default=None, help="feature-cache root (defaults to the built cache)")
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP, help="fixed sealed timestamp")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    from mop.beds.starss23.spatial_doa_producer import (
        DEFAULT_VARIANT_ARTIFACT_PATH,
        build_spatial_doa_artifact,
        write_variant_artifact,
    )

    out_path = Path(args.out_path) if args.out_path else REPO_ROOT / DEFAULT_VARIANT_ARTIFACT_PATH

    bed = build_spatial_doa_artifact(
        timestamp=args.timestamp,
        cache_root=args.cache_root,
    )
    path = write_variant_artifact(bed.artifact, out_path)

    detail = bed.detail
    spread = detail["spread"]
    cand = spread["candidate"]
    rmr = spread["rate_matched_random"]
    print(f"wrote {path}")
    print(
        f"featurizer=spatial_doa source_kind=real rights_clean=true verdict={bed.verdict} "
        f"seal={bed.seal}"
    )
    print(
        f"beats_rate_matched_random={detail['beats_random']} dominates={detail['dominates']} "
        f"mean_delta={detail['mean_delta']:.6f} one_sided_p={detail['one_sided_p']} "
        f"sesoi_f1={detail['sesoi_f1']} exceeds_sesoi={detail['mean_delta_exceeds_sesoi']} "
        f"noisy_tv_at_chance={detail['noisy_tv_at_chance']}"
    )
    print(f"per_seed_deltas={detail['per_seed_deltas']}")
    print(
        f"featurizer_flops_per_frame={detail['featurizer_flops_per_frame']} "
        f"candidate_max_lifecycle_flops={detail['candidate_max_lifecycle_flops']} "
        f"flop_ceiling={detail['flop_ceiling']} "
        f"within_ceiling={detail['candidate_max_lifecycle_flops'] <= detail['flop_ceiling']}"
    )
    print(
        f"candidate: mean_fires={cand['mean_fires']} "
        f"mean_adjacency={cand['mean_adjacency_fraction']:.4f} "
        f"mean_distinct_tp={cand['mean_distinct_onset_tp']}"
    )
    print(
        f"rate_matched_random: mean_fires={rmr['mean_fires']} "
        f"mean_adjacency={rmr['mean_adjacency_fraction']:.4f} "
        f"mean_distinct_tp={rmr['mean_distinct_onset_tp']}"
    )
    print(
        "flags: activation_allowed=false scientific_promotion=false "
        "independent_scientific_confirmation=false (one real run of one featurizer; mechanics outcome only)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
