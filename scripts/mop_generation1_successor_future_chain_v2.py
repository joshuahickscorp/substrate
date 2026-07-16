#!/usr/bin/env python3
"""Start the append-only Generation 1 successor future chain v2."""

# ruff: noqa: E402 - direct execution must bootstrap the repository before MOP imports

from __future__ import annotations

import sys
from pathlib import Path

REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
for _source_root in (REPO_BOOTSTRAP / "src", REPO_BOOTSTRAP):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from mop.studio.generation1_successor_future_chain_v2 import main

if __name__ == "__main__":
    raise SystemExit(main())
