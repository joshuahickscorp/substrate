from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mop.config import REPO_ROOT
from mop.studio.profiles import list_profiles
from mop.studio_doctor import doctor, render_md


def _doctor(args: argparse.Namespace) -> int:
    report = doctor(args.profile)
    markdown = REPO_ROOT / "runs" / "studio_doctor.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_md(report))
    if args.out:
        out = Path(args.out)
        out = out if out.is_absolute() else REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["all_ok"] else 1


def _profiles(_: argparse.Namespace) -> int:
    print(json.dumps(list_profiles(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MOP host operations")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor_parser = sub.add_parser("doctor", help="run host readiness checks")
    doctor_parser.add_argument("--profile", default=None)
    doctor_parser.add_argument("--out", default=None)
    doctor_parser.set_defaults(func=_doctor)
    profiles_parser = sub.add_parser("profiles", help="list resource profiles")
    profiles_parser.set_defaults(func=_profiles)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
