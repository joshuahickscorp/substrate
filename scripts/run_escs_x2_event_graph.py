#!/usr/bin/env python3
"""Manifest-governed entry point for the activation-disabled ESCS X2 scaffold."""

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

main = importlib.import_module("mop.studies.escs_x2_event_graph").main


if __name__ == "__main__":
    raise SystemExit(main())
