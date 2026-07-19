#!/usr/bin/env python3
"""Producer entrypoint: run the STARSS23 ESCS gate variant "refractory_nms" and seal its proof artifact.

This runs the refractory_nms E1 gate variant on the REAL, MIT-licensed STARSS23 FOA subset, consuming the
shared feature cache so the frozen featurizer is never recomputed. It scores the variant through the same
sealed harness, referee, controls, and sign-flip statistics as the committed real run, reads the SESOI
from the already-sealed variant preregistration (never rebuilding it), and writes the sealed
``proof/STARSS23_ESCS_BED_refractory_nms.json`` with the same shape as the committed artifact
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
        description="Run the STARSS23 ESCS refractory_nms gate variant on the cached real corpus."
    )
    parser.add_argument("out_path", nargs="?", default=None, help="sealed artifact output path")
    parser.add_argument("--cache-root", default=None, help="feature-cache root (defaults to the built cache)")
    parser.add_argument("--timestamp", default=DEFAULT_TIMESTAMP, help="fixed sealed timestamp")
    parser.add_argument("--window", type=int, default=None, help="refractory NMS window in frames")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    from mop.beds.starss23.feature_cache import DEFAULT_CACHE_ROOT
    from mop.beds.starss23.gate_refractory_nms import DEFAULT_WINDOW_FRAMES
    from mop.beds.starss23.refractory_nms_producer import (
        DEFAULT_VARIANT_ARTIFACT_PATH,
        build_refractory_nms_artifact,
    )
    from mop.substrate.events import write_canonical_json

    out_path = Path(args.out_path) if args.out_path else REPO_ROOT / DEFAULT_VARIANT_ARTIFACT_PATH
    cache_root = Path(args.cache_root) if args.cache_root else DEFAULT_CACHE_ROOT
    window = args.window if args.window is not None else DEFAULT_WINDOW_FRAMES

    bed = build_refractory_nms_artifact(
        timestamp=args.timestamp,
        cache_root=cache_root,
        window=window,
    )
    path = write_canonical_json(bed.artifact, out_path)

    detail = bed.detail
    spread = detail["spread"]
    var_c = spread["variant_candidate"]
    base_c = spread["committed_gate_candidate"]
    print(f"wrote {path}")
    print(
        f"variant=refractory_nms source_kind=real rights_clean=true verdict={bed.verdict} "
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
        f"variant_candidate: mean_fires={var_c['mean_fires']} "
        f"mean_adjacency={var_c['mean_adjacency_fraction']:.4f} "
        f"mean_distinct_tp={var_c['mean_distinct_onset_tp']}"
    )
    print(
        f"committed_gate_reference: mean_fires={base_c['mean_fires']} "
        f"mean_adjacency={base_c['mean_adjacency_fraction']:.4f} "
        f"mean_distinct_tp={base_c['mean_distinct_onset_tp']}"
    )
    print(
        "flags: activation_allowed=false scientific_promotion=false "
        "independent_scientific_confirmation=false (one real run of one variant; mechanics outcome only)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
