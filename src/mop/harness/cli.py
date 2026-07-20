from __future__ import annotations

import json
import sys

OPERATIONS = {"doctor", "profiles"}


def _doctor(args) -> int:
    from ..config import REPO_ROOT
    from ..evidence import atomic_write_json
    from ..studio_doctor import doctor, render_md

    report = doctor(args.profile)
    markdown = REPO_ROOT / "runs/studio_doctor.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_md(report))
    if args.out:
        out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
        atomic_write_json(out, report)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["all_ok"] else 1


def _profiles(_args) -> int:
    from ..studio.profiles import list_profiles

    print(json.dumps(list_profiles(), indent=2))
    return 0


def _operation(argv: list[str]) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="mop",
        description="MOP operations",
        epilog="Other arguments run an experiment as configuration overrides.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="run host readiness checks")
    doctor.add_argument("--profile")
    doctor.add_argument("--out", type=Path)
    doctor.set_defaults(func=_doctor)
    profiles = subparsers.add_parser("profiles", help="list resource profiles")
    profiles.set_defaults(func=_profiles)
    args = parser.parse_args(argv)
    return int(args.func(args))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in OPERATIONS | {"-h", "--help"}:
        return _operation(argv or ["--help"])
    from ..config import compose
    from .runner import run_experiment

    cfg = compose(argv)
    metrics = run_experiment(cfg)
    print(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
