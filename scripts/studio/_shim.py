"""Compatibility helpers for historical Studio wrapper scripts."""

from __future__ import annotations

import sys

from .__main__ import main as studio_main


def forward(command: str, argv: list[str] | None = None) -> int:
    return studio_main([command, *(sys.argv[1:] if argv is None else argv)])
