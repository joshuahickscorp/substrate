"""Reproduce the sealed Odyssey control plane in an exact clean clone.

This is intentionally a scoped CI receipt for the Odyssey control plane.  It
does not evaluate experiments or start a worker.  Its only effect is to write
a self-digested receipt after the exact committed tree has cloned, passed the
Odyssey test and lint targets, and regenerated the frozen build identically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from substrate import odyssey_transition

PROGRAM = "substrate-odyssey-7d-v1"
PLAN = Path("plans/substrate/tangible_next_launch")
TEST_TARGETS = (
    "tests/substrate/test_odyssey7d.py",
    "tests/substrate/test_odyssey_authority.py",
    "tests/substrate/test_odyssey_machine_subjects.py",
    "tests/substrate/test_odyssey_model_canary.py",
    "tests/substrate/test_odyssey_rehearsal.py",
    "tests/substrate/test_odyssey_task_bank.py",
    "tests/substrate/test_odyssey_manifest_materializer.py",
    "tests/substrate/test_odyssey_worker.py",
    "tests/substrate/test_odyssey_arms.py",
    "tests/substrate/test_odyssey_mutations.py",
    "tests/substrate/test_r2_continuity_verifier.py",
    "tests/substrate/test_r2_provenance_verifier.py",
    "tests/substrate/test_odyssey_clean_clone.py",
    "tests/substrate/test_odyssey_telegram_probe.py",
    "tests/substrate/test_odyssey7d_telegram_notifier.py",
    "tests/substrate/test_odyssey_detachment.py",
)
LINT_TARGETS = (
    "src/substrate/odyssey7d.py",
    "src/substrate/odyssey_authority.py",
    "src/substrate/odyssey_machine_subjects.py",
    "src/substrate/odyssey_model_canary.py",
    "src/substrate/odyssey_rehearsal.py",
    "src/substrate/odyssey_clean_clone.py",
    "src/substrate/odyssey_telegram_probe.py",
    "src/substrate/odyssey_task_bank.py",
    "src/substrate/odyssey_manifest_materializer.py",
    "src/substrate/odyssey_mutations.py",
    "src/substrate/odyssey_detachment.py",
    "src/substrate/odyssey_transition.py",
    "src/substrate/odyssey_worker.py",
    "src/substrate/odyssey_arms.py",
    "src/substrate/r2_continuity_verifier.py",
    "src/substrate/r2_provenance_verifier.py",
    "tools/odyssey7d_telegram_notifier.py",
    *TEST_TARGETS,
)
REQUIRED_CHECKS = (
    "exact_commit_checkout",
    "scoped_tests",
    "ruff_check",
    "frozen_build_regeneration",
    "source_map_match",
)


class Refused(RuntimeError):
    """The clone cannot honestly certify the requested source tree."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Refused(f"{path} must contain an object")
    unsigned = dict(value)
    claimed = unsigned.pop("sha256", None)
    if not isinstance(claimed, str) or claimed != digest(unsigned):
        raise Refused(f"{path} has an invalid self-digest")
    return value


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    head = completed.stdout.strip()
    if completed.returncode != 0 or len(head) != 40:
        raise Refused("cannot resolve the current git HEAD")
    return head


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, env=env, check=False)
    except OSError as error:
        return {"ok": False, "command": command, "detail": f"could not start: {error}"}
    detail = ((completed.stdout or "") + (completed.stderr or ""))[-2000:]
    return {
        "ok": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "detail": detail,
    }


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _source_map(root: Path, paths: dict[str, Path]) -> dict[str, str]:
    if not all(path.is_file() for path in paths.values()):
        missing = sorted(name for name, path in paths.items() if not path.is_file())
        raise Refused(f"required frozen source is missing: {missing}")
    return {name: file_digest(path) for name, path in paths.items()}


def _frozen_build(root: Path) -> dict[str, Any]:
    frozen = _read_json(root / PLAN / "ODYSSEY_FROZEN_BUILD.json")
    if frozen.get("schema") != "SUBSTRATE_ODYSSEY_FROZEN_BUILD/v1":
        raise Refused("unexpected frozen-build schema")
    expected_inputs = frozen.get("input_sha256")
    expected_implementation = frozen.get("implementation_sha256")
    if not isinstance(expected_inputs, dict) or not isinstance(expected_implementation, dict):
        raise Refused("frozen build lacks source maps")
    if _source_map(root, odyssey_transition.build_inputs(root)) != expected_inputs:
        raise Refused("current frozen build inputs drift from the sealed source map")
    if _source_map(root, odyssey_transition.implementation_inputs(root)) != expected_implementation:
        raise Refused("current frozen implementation drifts from the sealed source map")
    return frozen


def _scoped_paths(root: Path) -> list[str]:
    paths = {
        str(path.relative_to(root)) for path in (*odyssey_transition.build_inputs(root).values(), *odyssey_transition.implementation_inputs(root).values())
    }
    paths.update(TEST_TARGETS)
    paths.update(LINT_TARGETS)
    return sorted(paths)


def _assert_scoped_tree_clean(root: Path) -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--", *_scoped_paths(root)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise Refused("cannot inspect scoped git status")
    if completed.stdout.strip():
        raise Refused("Odyssey sources, plans, or scoped tests are not committed")


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    if path.exists():
        raise Refused(f"refusing to overwrite existing clean-clone receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise Refused(f"refusing to replace an existing temporary receipt: {temporary}")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def run(root: Path, output_path: Path) -> dict[str, Any]:
    """Run scoped CI in a clean clone and write one immutable receipt."""
    root = root.expanduser().resolve()
    output_path = (root / output_path).resolve() if not output_path.is_absolute() else output_path.resolve()
    if not _inside(root, output_path):
        raise Refused("output path must stay inside the repository root")
    _assert_scoped_tree_clean(root)
    source_commit = _git_head(root)
    frozen = _frozen_build(root)
    frozen_sha256 = frozen["sha256"]
    environment = {**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"}
    with tempfile.TemporaryDirectory(prefix="substrate-odyssey-cleanclone-") as temporary:
        temporary_root = Path(temporary)
        clone = temporary_root / "substrate"
        clone_result = _run(
            ["git", "clone", "--quiet", "--no-local", "--no-hardlinks", str(root), str(clone)],
            cwd=temporary_root,
            env=environment,
        )
        checkout_result = {"ok": False, "detail": "clone did not complete"}
        clone_head: str | None = None
        if clone_result["ok"]:
            checkout_result = _run(["git", "checkout", "--quiet", "--detach", source_commit], cwd=clone, env=environment)
            if checkout_result["ok"]:
                clone_head = _git_head(clone)
        clone_environment = {**environment, "PYTHONPATH": str(clone / "src")}
        available = clone_result["ok"] and checkout_result["ok"] and clone_head == source_commit
        tests_result = (
            _run([sys.executable, "-m", "pytest", "-q", *TEST_TARGETS], cwd=clone, env=clone_environment)
            if available
            else {"ok": False, "detail": "exact checkout unavailable"}
        )
        lint_result = (
            _run([sys.executable, "-m", "ruff", "check", *LINT_TARGETS], cwd=clone, env=clone_environment)
            if available
            else {"ok": False, "detail": "exact checkout unavailable"}
        )
        frozen_before = _read_json(clone / PLAN / "ODYSSEY_FROZEN_BUILD.json") if available else None
        freeze_one = (
            _run(
                [sys.executable, "-m", "substrate.odyssey_transition", "freeze", "--root", str(clone)],
                cwd=clone,
                env=clone_environment,
            )
            if available
            else {"ok": False, "detail": "exact checkout unavailable"}
        )
        frozen_after_one = _read_json(clone / PLAN / "ODYSSEY_FROZEN_BUILD.json") if freeze_one["ok"] else None
        freeze_two = (
            _run(
                [sys.executable, "-m", "substrate.odyssey_transition", "freeze", "--root", str(clone)],
                cwd=clone,
                env=clone_environment,
            )
            if freeze_one["ok"]
            else {"ok": False, "detail": "first frozen-build regeneration failed"}
        )
        frozen_after_two = _read_json(clone / PLAN / "ODYSSEY_FROZEN_BUILD.json") if freeze_two["ok"] else None
        source_map = _source_map(clone, odyssey_transition.implementation_inputs(clone)) if available else {}
        input_map = _source_map(clone, odyssey_transition.build_inputs(clone)) if available else {}
        checks = {
            "exact_commit_checkout": available,
            "scoped_tests": tests_result["ok"],
            "ruff_check": lint_result["ok"],
            "frozen_build_regeneration": (
                freeze_one["ok"]
                and freeze_two["ok"]
                and frozen_before is not None
                and frozen_after_one is not None
                and frozen_after_two is not None
                and frozen_before.get("sha256") == frozen_sha256
                and frozen_after_one.get("sha256") == frozen_sha256
                and frozen_after_two.get("sha256") == frozen_sha256
            ),
            "source_map_match": (source_map == frozen["implementation_sha256"] and input_map == frozen["input_sha256"]),
        }
        body = {
            "schema": "SUBSTRATE_ODYSSEY_CLEAN_CLONE_CI/v1",
            "program": PROGRAM,
            "activation": False,
            "external_activation": False,
            "source_commit": source_commit,
            "frozen_build_sha256": frozen_sha256,
            "regenerated_frozen_build_sha256": (frozen_after_two.get("sha256") if frozen_after_two is not None else None),
            "implementation_sha256": source_map,
            "input_sha256": input_map,
            "test_targets": list(TEST_TARGETS),
            "lint_targets": list(LINT_TARGETS),
            "checks": checks,
            "all_pass": all(checks[name] for name in REQUIRED_CHECKS),
            "clone": {
                "checkout_head": clone_head,
                "clone": clone_result,
                "checkout": checkout_result,
                "tests": tests_result,
                "lint": lint_result,
                "freeze_one": freeze_one,
                "freeze_two": freeze_two,
            },
        }
    body["sha256"] = digest(body)
    _write_json(output_path, body)
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce the sealed Odyssey control plane in a clean clone")
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.root, args.out)
    except Refused as error:
        print(json.dumps({"activation": False, "refused": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
