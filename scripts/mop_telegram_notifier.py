#!/usr/bin/env python3
"""Control MOP Telegram rung notifications using the existing Hawking bot."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_notifier() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "src/mop/studio/telegram_rung_notifier.py"
    spec = importlib.util.spec_from_file_location("mop_telegram_rung_notifier_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the MOP Telegram notifier module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    notifier = _load_notifier()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "prime", "send-test", "install", "run"))
    args = parser.parse_args()
    try:
        if args.command == "status":
            result = notifier.status()
        elif args.command == "prime":
            result = notifier.prime()
        elif args.command == "send-test":
            result = notifier.send_message(
                "✅ MOP Generation 1 notifications are connected.\n"
                "Rungs, important results, failures, and campaign completion will appear here."
            )
        elif args.command == "install":
            result = notifier.install_launch_agent()
        else:
            result = notifier.run_once()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (RuntimeError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
