#!/usr/bin/env python3
"""Producer entrypoint for the ``recurrence_spread`` E1 gate variant.

Runs the recurrence_spread variant on the REAL, MIT-licensed STARSS23 FOA corpus by reading the SHARED
feature cache (never re-featurizing the corpus), through the EXISTING sealed harness, referee, controls,
and sign-flip statistics. It uses the sealed E1 variants preregistration
(``proof/STARSS23_ESCS_BED_VARIANTS.prereg.json``) for the SESOI (0.05), the direction, and the sign-flip
plan, and writes the sealed ``proof/STARSS23_ESCS_BED_recurrence_spread.json`` with the same shape as the
committed artifact (flags false, source_kind real, rights_clean true).

The variant beats rate-matched-random only if it strictly dominates on mean F1 at every budget AND clears
the one-sided sign-flip AND exceeds the preregistered SESOI. A single run is a mechanics demonstration
only; it can never be scientifically confirmed.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# A fixed default timestamp so any timestamped provenance stays byte-reproducible across re-runs. The wall
# clock is never read inside a sealed body.
DEFAULT_TIMESTAMP = "2026-07-17T00:00:00Z"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the STARSS23 ESCS recurrence_spread gate variant on the cached real corpus."
    )
    parser.add_argument("out_path", nargs="?", default=None, help="sealed variant artifact output path")
    parser.add_argument("--cache-root", default=None, help="feature cache root (defaults to the built cache)")
    parser.add_argument("--foa", default=None, help="FOA audio root (for cache keying)")
    parser.add_argument("--metadata", default=None, help="metadata root (for cache keying)")
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP, help="fixed provenance timestamp")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    from mop.beds.starss23.feature_cache import DEFAULT_CACHE_ROOT
    from mop.beds.starss23.real_artifact import DEFAULT_FOA_ROOT, DEFAULT_METADATA_ROOT
    from mop.beds.starss23.recurrence_spread_artifact import (
        build_recurrence_spread_artifact,
        write_recurrence_spread_artifact,
    )

    out_path = (
        Path(args.out_path)
        if args.out_path
        else REPO_ROOT / "proof" / "STARSS23_ESCS_BED_recurrence_spread.json"
    )

    bed = build_recurrence_spread_artifact(
        timestamp=args.timestamp,
        cache_root=args.cache_root or DEFAULT_CACHE_ROOT,
        foa_root=args.foa or DEFAULT_FOA_ROOT,
        metadata_root=args.metadata or DEFAULT_METADATA_ROOT,
    )
    path = write_recurrence_spread_artifact(bed.artifact, out_path)

    diag = bed.detail["fire_spread_diagnostic"]
    print(f"wrote {path}")
    print(
        f"variant=recurrence_spread source_kind=real rights_clean=true verdict={bed.verdict} "
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
        "fire_spread: "
        f"candidate_adjacency={diag['mean_candidate_adjacency_fraction']:.4f} "
        f"random_adjacency={diag['mean_rate_matched_random_adjacency_fraction']:.4f} "
        f"candidate_distinct_tp={diag['mean_candidate_distinct_onset_tp']:.1f} "
        f"random_distinct_tp={diag['mean_rate_matched_random_distinct_onset_tp']:.1f} "
        f"(base_null_seed0=204, random_seed0=237)"
    )
    print(
        "flags: activation_allowed=false scientific_promotion=false "
        "independent_scientific_confirmation=false (one real variant run; mechanics demonstration only)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
