from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from mop.config import REPO_ROOT


def test_direct_fresh_challenge_launcher_resolves_sibling_guard(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/p5_context_fresh_challenge.py",
            "--primary",
            str(tmp_path / "missing-primary.json"),
            "--primary-run-dir",
            str(tmp_path / "missing-primary-run"),
            "--run-dir",
            str(tmp_path / "fresh-run"),
            "--out",
            str(tmp_path / "fresh-proof.json"),
            "--device",
            "cpu",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "ModuleNotFoundError" not in result.stderr
    assert "P5 fresh challenge refused:" in result.stderr
