#!/usr/bin/env python3
"""Run or verify the deterministic inert G0 formation mechanics study."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

main = importlib.import_module("mop.studies.escs_g0_formation_study").main


if __name__ == "__main__":
    raise SystemExit(main())
