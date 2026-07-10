#!/usr/bin/env python
"""Build and transactionally publish the semantic MOP potential atlas bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from mop.config import REPO_ROOT
from mop.studies.potential_atlas_driver import write_atlas_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "proof/MOP_POTENTIAL_ATLAS.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "proof/MOP_POTENTIAL_ATLAS.json")
    parser.add_argument(
        "--markdown", type=Path, default=REPO_ROOT / "MOP_POTENTIAL_ATLAS_2026_07.md"
    )
    parser.add_argument(
        "--requirements", type=Path, default=REPO_ROOT / "proof/EXTENDED_COMPUTE_REQUIREMENTS.json"
    )
    parser.add_argument(
        "--validation", type=Path, default=REPO_ROOT / "proof/MOP_POTENTIAL_ATLAS_VALIDATION.json"
    )
    args = parser.parse_args(argv)
    receipt = write_atlas_bundle(
        source_path=args.source,
        atlas_path=args.out,
        markdown_path=args.markdown,
        requirements_path=args.requirements,
        validation_path=args.validation,
    )
    summary = receipt["summary"]
    print(
        f"wrote {args.out}: facets={summary['facet_count']}, "
        f"weighted={summary['weighted_score']:.3f}, category2={summary['category2_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
