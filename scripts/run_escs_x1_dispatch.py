#!/usr/bin/env python3
"""Manifest-governed entry point for the unexecuted ESCS X1 preregistration."""

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

main = importlib.import_module("mop.studies.escs_x1_dispatch").main


if __name__ == "__main__":
    raise SystemExit(main())
