"""Clean clone validation of the terminal commit.

A clone is made from the local repository, the exact commit is checked out, and the package is imported,
tested and verified there. Environmental skips are recorded exactly. Claiming offline success while a
required cached dependency is absent would be worse than reporting the skip.

House style: no dashes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fastforge.runs import io

PY = sys.executable
PROG = io.PROGRAM


def run(cmd, cwd, env=None, timeout=1800):
    try:
        r = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=dict(os.environ, **(env or {}))
        )
        return {
            "cmd": " ".join(cmd),
            "exit_code": r.returncode,
            "tail": (r.stdout or r.stderr).strip().splitlines()[-3:],
        }
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd), "exit_code": -1, "tail": ["timeout"]}


def main():
    t0 = time.time()
    commit = io.commit()
    tmp = Path(tempfile.mkdtemp(prefix="mop-fast-state-clone-"))
    dest = tmp / "clone"
    steps = {}
    steps["clone"] = run(["git", "clone", "--no-local", "--quiet", str(io.ROOT), str(dest)], tmp)
    steps["checkout"] = run(["git", "checkout", "--quiet", commit], dest)
    env = {"PYTHONPATH": "src", "OMP_NUM_THREADS": "1"}
    steps["import_package"] = run([PY, "-c", "import mop, fastforge; print(fastforge.__doc__)"], dest, env)
    steps["cli"] = run([PY, "-c", "from mop.harness.cli import main; print(main.__module__)"], dest, env)
    steps["acceptance"] = run([PY, "scripts/acceptance.py"], dest, env)
    steps["substrate_tests"] = run(
        [PY, "-m", "pytest", "tests/unit/test_fast_state_forge.py", "-q"], dest, env
    )
    steps["integrated_tests"] = run([PY, "-m", "pytest", "tests/unit", "-q"], dest, env)
    fabric = f"proof/substrate/{PROG}/MOP_FAST_STATE_EVIDENCE_FABRIC.json"
    index_check = (
        "import json,hashlib,pathlib;"
        f"f=json.load(open({fabric!r}));"
        "bad=[a['logical_id'] for a in f['artifacts'] if "
        "hashlib.sha256(pathlib.Path(a['original_path']).read_bytes()).hexdigest()!=a['content_hash']];"
        "print('bad',len(bad));assert not bad"
    )
    recovery_check = (
        "import json,pathlib;"
        f"f=json.load(open({fabric!r}));"
        "miss=[a['logical_id'] for a in f['artifacts'] if not pathlib.Path(a['canonical_path']).is_file()];"
        "print('missing',len(miss));assert not miss"
    )
    steps["evidence_index"] = run([PY, "-c", index_check], dest, env)
    steps["checkpoint_recovery"] = run([PY, "-c", recovery_check], dest, env)
    dirt = subprocess.run(["git", "status", "--porcelain"], cwd=dest, capture_output=True, text=True)
    steps["worktree_clean"] = {
        "cmd": "git status --porcelain",
        "exit_code": 0 if not dirt.stdout.strip() else 1,
        "tail": dirt.stdout.strip().splitlines()[:3],
    }

    from fastforge import data as D

    skips = {
        "har_dataset_present": D.HAR_ROOT.is_dir(),
        "speech_cache_present": D.SPEECH_CACHE.exists(),
        "harth_present": D.HARTH_ROOT.is_dir(),
        "note": "domain data lives outside the repository by design, so a clean clone can validate code, "
        "tests and evidence but cannot rerun training without those paths. Tests that need data "
        "skip themselves rather than pass silently.",
    }
    passed = {k: v["exit_code"] == 0 for k, v in steps.items()}
    io.seal(
        "MOP_FAST_STATE_CLEAN_CLONE.json",
        {
            "schema": "mop-fast-state-clean-clone/v1",
            "commit": commit,
            "clone_path": str(dest),
            "steps": steps,
            "passed": passed,
            "all_pass": all(passed.values()),
            "environmental_skips": skips,
            "wall_seconds": round(time.time() - t0, 1),
        },
    )
    print(json.dumps(passed), flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("CLEANCLONE_DONE", flush=True)


if __name__ == "__main__":
    main()
