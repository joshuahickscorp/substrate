#!/usr/bin/env python3
"""Independently verify the Generation 1 C1 competence atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.studies.generation1_competence_atlas_verify import verify_atlas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    result = verify_atlas(arguments.config, arguments.atlas, arguments.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verification_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
