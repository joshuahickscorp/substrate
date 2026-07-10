#!/usr/bin/env python3
"""Validate the committed MOP potential atlas and optionally write a durable receipt."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

main = import_module("mop.studies.potential_atlas_validation").main


if __name__ == "__main__":
    raise SystemExit(main())
