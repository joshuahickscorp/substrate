#!/usr/bin/env python
"""Build the serial, citable local V-JEPA scale-atlas receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from mop.config import REPO_ROOT
from mop.studies.vjepa_scale_atlas import build_local_scale_atlas, write_local_scale_atlas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default=str(REPO_ROOT / "data" / "cache"))
    parser.add_argument(
        "--random-control",
        action="append",
        default=[],
        metavar="TAG=PATH",
        help="add a citable random-architecture cache; repeat for multiple scales or seeds",
    )
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument(
        "--stimulus-identity",
        default=str(REPO_ROOT / "proof" / "FACTORIZED_STIMULUS_IDENTITY.json"),
        help="separate receipt used only after strict cache/hash/rebinding validation",
    )
    parser.add_argument("--out", default=str(REPO_ROOT / "proof" / "VJEPA_SCALE_ATLAS_LOCAL.json"))
    args = parser.parse_args(argv)
    root = Path(args.cache_root)
    caches: dict[str, Path | str] = {
        "vit_l": root / "vjepa2_vitl_local8_citable",
        "vit_h": root / "vjepa2_vith_local8_citable",
        "vit_g": root / "vjepa2_vitg_local8_citable",
    }
    for spec in args.random_control:
        tag, separator, raw_path = spec.partition("=")
        if not separator or not tag.strip() or not raw_path.strip():
            parser.error(f"--random-control must be TAG=PATH, got {spec!r}")
        if tag in caches:
            parser.error(f"duplicate cache tag {tag!r}")
        caches[tag] = Path(raw_path)
    receipt = build_local_scale_atlas(
        caches,
        permutations=args.permutations,
        stimulus_identity_path=args.stimulus_identity,
    )
    write_local_scale_atlas(receipt, args.out)
    print(f"wrote {args.out}: {len(receipt['caches'])} citable scales, promotable=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
