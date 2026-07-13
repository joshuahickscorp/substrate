#!/usr/bin/env python3
"""Validate and optionally publish the quiescent ESCS substrate preflight."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from mop.config import REPO_ROOT
from mop.escs.substrate_preflight import (
    assess_substrate_preflight,
    load_substrate_preflight_manifest,
)

DEFAULT_MANIFEST = REPO_ROOT / "configs/experiment/escs_substrate_preflight.json"
DEFAULT_OUTPUT = REPO_ROOT / "proof/ESCS_SUBSTRATE_PREFLIGHT.json"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    manifest = load_substrate_preflight_manifest(arguments.manifest)
    report = assess_substrate_preflight(manifest, repository_root=REPO_ROOT)
    payload = report.payload()
    if not arguments.validate_only:
        output = arguments.out.resolve()
        protected = {arguments.manifest.resolve()}
        protected.update((REPO_ROOT / binding.path).resolve() for binding in manifest.bindings)
        if output in protected:
            raise ValueError("substrate preflight output aliases a bound input authority")
        _atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.scaffold_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
