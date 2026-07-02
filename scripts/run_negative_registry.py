#!/usr/bin/env python
"""Run Leg 11D negative-result registry: execute each experiment's cheapest predicted-null
ablation at toy scale (compose + run_experiment), read its null-related booleans, and record a
verdict + taxonomy slot per experiment. Logs progress to stderr, emits the registry dict as JSON
to stdout, and writes a markdown table to runs/negative_registry.md.

Usage: python scripts/run_negative_registry.py [--full]
"""

from __future__ import annotations

import argparse
import json
import sys

from mop.config import REPO_ROOT
from mop.studies.negative_registry import negative_registry, to_markdown


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run experiments at config-default scale (slower)")
    a = ap.parse_args(argv)

    print(f"[11D] negative-result registry (toy={not a.full})", file=sys.stderr)
    reg = negative_registry(toy=not a.full)
    for e in reg["entries"]:
        print(
            f"[11D] {e['experiment']:16s} verdict={e['verdict']:9s} cat={e['taxonomy_category']}",
            file=sys.stderr,
        )
    s = reg["summary"]
    print(
        f"[11D] summary confirmed={s['confirmed']} refuted={s['refuted']} "
        f"mixed={s['mixed']} error={s.get('error', 0)}",
        file=sys.stderr,
    )

    out_dir = REPO_ROOT / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "negative_registry.md"
    md_path.write_text(to_markdown(reg))
    print(f"[11D] wrote {md_path}", file=sys.stderr)

    print(json.dumps(reg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
