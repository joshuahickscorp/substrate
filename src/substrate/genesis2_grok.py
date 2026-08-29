"""Launch and archive read-only Grok review cells.

Review text is advisory.  The archive records disposition, but no Grok
judgment is ever mapped to an experimental pass/fail field.
"""

from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from substrate import genesis2_config as C2
from substrate import genesis2_io as IO2

RUNNER = Path.home() / ".claude-grok" / "bin" / "grok-run"
TASK_ROOT = Path.home() / ".claude-grok" / "tasks"


def _prompt(role: str, round_name: str) -> str:
    return (
        f"Role: {role}. Round: {round_name}. You are an independent, read-only hostile reviewer "
        "for Substrate Cognitive Material Genesis II. The program compares exact and low-bit S2, "
        "equally plastic associative monoliths, and fields with exact microstores, structural "
        "consolidation, rare topology, conditional granularity, calibrated binding budgets, "
        "checkpoint continuity, mutations, and hidden composition. External activation is false "
        "and unqualified Nous is forbidden. Return a concise review with: (1) the strongest validity "
        "threat in your specialty, (2) one concrete falsifying check, (3) one fairness check for S2 "
        "or the strongest monolith, and (4) a disposition recommendation among adopted, "
        "adopted_with_changes, rejected, superseded, deferred. Opinions are not evidence."
    )


def _launch_one(role: str, round_name: str) -> dict[str, Any]:
    process = subprocess.run(
        [str(RUNNER), "consult", "--prompt", _prompt(role, round_name), "--background"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    task_id = process.stdout.strip().splitlines()[0] if process.returncode == 0 and process.stdout.strip() else ""
    return {
        "role": role,
        "round": round_name,
        "task_id": task_id,
        "launch_returncode": process.returncode,
        "launch_stderr": process.stderr.strip(),
        "launched": process.returncode == 0 and bool(task_id),
        "disposition": "deferred",
        "counts_as_evidence": False,
        "activation": False,
    }


def launch(
    *,
    round_name: str,
    roles: tuple[str, ...],
    workers: int = 8,
    publish: bool = True,
) -> dict[str, Any]:
    if round_name not in C2.REVIEW_ROUNDS:
        raise ValueError(f"unknown review round {round_name!r}")
    unknown = sorted(set(roles) - set(C2.REVIEW_CELLS))
    if unknown:
        raise ValueError(f"unknown review roles: {unknown}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(lambda role: _launch_one(role, round_name), roles))
    document = IO2.authority(
        "substrate-genesis2-grok-launch/v1",
        {
            "round": round_name,
            "reviews": rows,
            "roles": len(rows),
            "launched": sum(1 for row in rows if row["launched"]),
            "opinions_are_evidence": False,
            "all_pass": all(row["launched"] for row in rows),
        },
    )
    if publish:
        IO2.write_json(IO2.ARTIFACTS / f"grok_{round_name}_launch.json", document)
    return document


def _task_files(task_id: str) -> dict[str, Any]:
    root = TASK_ROOT / task_id
    if not root.is_dir():
        return {"status": "missing", "files": {}, "complete": False}
    files: dict[str, Any] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.stat().st_size > 2_000_000:
            files[path.name] = {
                "sha256": IO2.file_digest(path),
                "bytes": path.stat().st_size,
                "content_omitted": True,
            }
            continue
        text = path.read_text(errors="replace")
        if path.suffix == ".json":
            try:
                content: Any = json.loads(text)
            except json.JSONDecodeError:
                content = text
        else:
            content = text
        files[path.name] = {
            "sha256": IO2.file_digest(path),
            "bytes": path.stat().st_size,
            "content": content,
        }
    status = (root / "status").read_text().strip() if (root / "status").is_file() else "unknown"
    return {
        "status": status,
        "files": files,
        "complete": status == "done",
    }


def _review_disposition(task: dict[str, Any]) -> str:
    if not task["complete"]:
        return "deferred"
    report = task["files"].get("grok-report.md", {}).get("content", "")
    section = str(report).split("Disposition", 1)[-1]
    allowed = ("adopted_with_changes", "superseded", "rejected", "deferred", "adopted")
    for disposition in allowed:
        if re.search(rf"(?:`|\*\*){re.escape(disposition)}(?:`|\*\*)", section):
            return disposition
    return "adopted_with_changes"


def collect(launch_documents: list[dict[str, Any]], *, publish: bool = True) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for launch_document in launch_documents:
        for launched in launch_document.get("reviews", []):
            task = _task_files(str(launched["task_id"]))
            rows.append(
                {
                    **launched,
                    "task": task,
                    "completed": bool(task["complete"]),
                    "disposition": _review_disposition(task),
                    "counts_as_evidence": False,
                    "activation": False,
                }
            )
    roles = {str(row["role"]) for row in rows if row["completed"]}
    document = IO2.authority(
        "substrate-genesis2-grok-docs/archive/v1",
        {
            "reviews": rows,
            "distinct_completed_roles": len(roles),
            "minimum_roles": C2.GROK_MINIMUM_ROLES,
            "preferred_roles": C2.GROK_PREFERRED_ROLES,
            "minimum_met": len(roles) >= C2.GROK_MINIMUM_ROLES,
            "preferred_met": len(roles) >= C2.GROK_PREFERRED_ROLES,
            "opinions_are_evidence": False,
            "all_pass": len(roles) >= C2.GROK_MINIMUM_ROLES,
        },
    )
    if publish:
        IO2.write_json(IO2.ARTIFACTS / "grok_archive.json", document)
    return document


def demo() -> None:
    assert len(C2.REVIEW_CELLS) == len(set(C2.REVIEW_CELLS))
    assert C2.GROK_MINIMUM_ROLES <= C2.GROK_PREFERRED_ROLES <= len(C2.REVIEW_CELLS)
    assert "Opinions are not evidence" in _prompt(C2.REVIEW_CELLS[0], C2.REVIEW_ROUNDS[0])
    print("genesis2 Grok archive self-check passed")


if __name__ == "__main__":
    demo()
