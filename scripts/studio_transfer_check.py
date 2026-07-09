#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.studio._shim import forward  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return forward("transfer-check", argv)


if __name__ == "__main__":
    raise SystemExit(main())
