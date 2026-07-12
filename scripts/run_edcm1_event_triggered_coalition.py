#!/usr/bin/env python3
"""Entry point for the manifest-governed EDCM-1 v3 mechanics study."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mop.studies.edcm1_event_triggered_coalition import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
