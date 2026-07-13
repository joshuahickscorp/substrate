from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/run_escs_mechanics_chassis.py"
CONFIG = REPO_ROOT / "configs/experiment/escs_mechanics_chassis.json"


def _command(*extra: str) -> list[str]:
    return [sys.executable, str(SCRIPT), "--config", str(CONFIG), *extra]


def test_escs_mechanics_runner_is_deterministic_and_receipt_fails_on_tamper(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    subprocess.run(_command("--out", str(first)), cwd=REPO_ROOT, check=True)
    subprocess.run(_command("--out", str(second)), cwd=REPO_ROOT, check=True)

    first_payload = json.loads(first.read_text(encoding="utf-8"))
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    assert first_payload == second_payload
    assert first_payload["complete"] is True
    assert first_payload["all_ok"] is True
    assert first_payload["problems"] == []
    assert first_payload["claim_scope"] == "scripted-mechanics-only"
    assert first_payload["event_ledger"]["event_kinds"] == [
        "observation",
        "hypothesis",
        "commitment",
        "consequence",
    ]
    subprocess.run(
        _command("--out", str(first), "--verify-only"),
        cwd=REPO_ROOT,
        check=True,
    )

    first_payload["event_ledger"]["event_count"] += 1
    first.write_text(json.dumps(first_payload), encoding="utf-8")
    rejected = subprocess.run(
        _command("--out", str(first), "--verify-only"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 1
    assert "proof digest mismatch" in rejected.stderr


def test_escs_mechanics_runner_rejects_unknown_config_fields(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["unpriced_control_plane"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(config), encoding="utf-8")

    rejected = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(changed),
            "--out",
            str(tmp_path / "proof.json"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert rejected.returncode != 0
    assert "missing or unknown root fields" in rejected.stderr
