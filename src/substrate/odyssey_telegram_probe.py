"""Write an immutable acknowledgement receipt for one real Odyssey Telegram probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from substrate import odyssey_authority as authority

PROGRAM = "substrate-odyssey-7d-v1"
PLAN = Path("plans/substrate/tangible_next_launch")
NOTIFIER = Path("tools/odyssey7d_telegram_notifier.py")


class Refused(RuntimeError):
    """A probe receipt cannot be bound to the currently frozen build."""


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _load_notifier(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("substrate_odyssey_telegram_notifier", path)
    if spec is None or spec.loader is None:
        raise Refused("cannot load the pinned Odyssey Telegram notifier")
    module = importlib.util.module_from_spec(spec)
    # Decorators such as ``@dataclass`` resolve their module through
    # ``sys.modules`` while a dynamically-loaded file is executing.  Register
    # the exact pinned module first, and remove it again if its initialization
    # fails so a partial notifier can never be reused.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(spec.name, None)
        raise Refused("cannot initialize the pinned Odyssey Telegram notifier") from error
    return module


def run(root: Path, output_path: Path) -> dict[str, Any]:
    """Send one acknowledged probe and bind it to the exact frozen source map."""
    root = root.expanduser().resolve()
    output_path = (root / output_path).resolve() if not output_path.is_absolute() else output_path.resolve()
    if not _inside(root, output_path):
        raise Refused("output path must stay inside the repository root")
    if output_path.exists():
        raise Refused(f"refusing to overwrite existing Telegram probe receipt: {output_path}")
    frozen_document = authority._read_json(root / PLAN / "ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    frozen_sha256 = authority._assert_sha256(frozen_document.get("sha256"), label="frozen_build_sha256")
    frozen = authority._validate_frozen_build(root, frozen_sha256)
    notifier_path = root / NOTIFIER
    expected_notifier_sha256 = frozen.get("implementation_sha256", {}).get("telegram_notifier")
    if not isinstance(expected_notifier_sha256, str) or authority.file_digest(notifier_path) != expected_notifier_sha256:
        raise Refused("Telegram notifier is absent from or has drifted from the frozen source map")
    notifier = _load_notifier(notifier_path)
    result = notifier.probe(frozen_sha256)
    message_id = result.get("message_id") if isinstance(result, dict) else None
    checks = {
        "frozen_build_bound": result.get("frozen_build_sha256") == frozen_sha256 if isinstance(result, dict) else False,
        "notifier_source_bound": True,
        "telegram_api_acknowledged": result.get("delivered") is True if isinstance(result, dict) else False,
        "probe_message_id_valid": isinstance(message_id, int) and message_id > 0,
    }
    body = {
        "schema": "SUBSTRATE_ODYSSEY_TELEGRAM_PROBE/v1",
        "program": PROGRAM,
        "activation": False,
        "external_activation": False,
        "source_commit": authority._git_head(root),
        "frozen_build_sha256": frozen_sha256,
        "notifier_source_sha256": expected_notifier_sha256,
        "delivery": {"message_id": message_id, "acknowledged": checks["telegram_api_acknowledged"]},
        "checks": checks,
        "all_pass": all(checks.values()),
    }
    body["sha256"] = authority.digest(body)
    authority._write_json(output_path, body)
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send and record one frozen Odyssey Telegram probe")
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.root, args.out)
    except (Refused, RuntimeError) as error:
        print(json.dumps({"activation": False, "refused": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
