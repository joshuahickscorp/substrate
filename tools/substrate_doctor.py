#!/usr/bin/env python3
"""substrate doctor — report operational facts about this checkout.

Reports.  Never mutates.  Never deletes.  Exit code is 0 when every REQUIRED
check passes; advisory checks never fail the run.

Written during the pre-Ascension closeout so a new agent can establish repository
reality in one command instead of an hour of archaeology.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SUBSTRATE_REPOSITORY_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
PLAN = ROOT / "plans/substrate/tangible_next_launch"

OK, WARN, FAIL = "ok", "warn", "FAIL"
results: list[tuple[str, str, str, str]] = []  # (severity, area, check, detail)


def record(sev: str, area: str, check: str, detail: str) -> None:
    results.append((sev, area, check, detail))


def sh(*args: str, timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - doctor must never crash
        return 127, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- git

def check_index_lock() -> None:
    """Stale-lock candidate detection.  Never deletes: ownership can be ambiguous."""
    lock = ROOT / ".git" / "index.lock"
    if not lock.exists():
        record(OK, "git", "index.lock", "absent")
        return
    age_s = time.time() - lock.stat().st_mtime
    size = lock.stat().st_size
    rc, ps = sh("pgrep", "-f", "git ")
    live = [line for line in ps.splitlines() if line.strip()] if rc == 0 else []
    if live:
        record(WARN, "git", "index.lock",
               f"present ({size} B, {age_s/3600:.1f} h old) AND git processes are running — "
               f"ownership ambiguous, DO NOT delete")
    elif age_s > 3600:
        record(WARN, "git", "index.lock",
               f"STALE CANDIDATE: {size} B, {age_s/3600:.1f} h old, no live git process. "
               f"A human should confirm before removing {lock}")
    else:
        record(WARN, "git", "index.lock", f"present ({size} B, {age_s/60:.0f} min old) — likely a live operation")


def check_git() -> None:
    rc, head = sh("git", "rev-parse", "HEAD")
    record(OK if rc == 0 else FAIL, "git", "HEAD", head[:40] if rc == 0 else head)
    rc, branch = sh("git", "branch", "--show-current")
    record(OK if rc == 0 else WARN, "git", "branch", branch or "(detached)")
    rc, dirty = sh("git", "status", "--porcelain", "--untracked-files=no")
    n = len([x for x in dirty.splitlines() if x.strip()]) if rc == 0 else -1
    record(OK if rc == 0 else FAIL, "git", "tracked-dirty", f"{n} paths")
    check_index_lock()


# ---------------------------------------------------------------- environment

def check_python() -> None:
    venv_py = ROOT / ".venv/bin/python"
    if not venv_py.exists():
        record(FAIL, "env", "venv", f"missing at {venv_py} — run: uv venv --python 3.12 .venv")
        return
    rc, ver = sh(str(venv_py), "-c", "import sys;print('.'.join(map(str,sys.version_info[:3])))")
    pin = "3.12"
    record(OK if ver.startswith(pin) else FAIL, "env", "python",
           f"{ver} (canonical pin {pin}.x — Makefile:7, portability.py:454)")
    for mod in ("substrate", "numpy", "cryptography", "pytest", "docx", "openpyxl", "pypdf", "sympy"):
        rc, _ = sh(str(venv_py), "-c", f"import {mod}")
        record(OK if rc == 0 else FAIL, "env", f"import {mod}", "ok" if rc == 0 else "MISSING")


REQUIRED_DEV = {"git": "git", "z3": "z3"}
ODYSSEY_ONLY = {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe", "ollama": "ollama",
                "aria2c": "aria2c", "git-lfs": "git-lfs"}
OPTIONAL = {"colmap": "colmap"}


def check_tools() -> None:
    for name, exe in REQUIRED_DEV.items():
        record(OK if shutil.which(exe) else FAIL, "tool", f"{name} (required-for-dev)",
               shutil.which(exe) or "ABSENT")
    for name, exe in ODYSSEY_ONLY.items():
        record(OK if shutil.which(exe) else WARN, "tool", f"{name} (odyssey-only)",
               shutil.which(exe) or "absent — Odyssey paths only")
    for name, exe in OPTIONAL.items():
        record(OK if shutil.which(exe) else WARN, "tool", f"{name} (optional)",
               shutil.which(exe) or "absent")
    elan = Path.home() / ".elan/bin/elan"
    if elan.exists():
        rc, out = sh(str(elan), "run", "leanprover/lean4:v4.33.0-rc1", "lean", "--version")
        record(OK if "4.33.0-rc1" in out else WARN, "tool", "lean (odyssey-only)",
               out.splitlines()[0] if out else "no output")
    else:
        record(WARN, "tool", "lean (odyssey-only)", f"absent — no {elan}")
    blender = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    record(OK if blender.exists() else WARN, "tool", "blender (optional)",
           "app bundle present (intentionally not on PATH)" if blender.exists() else "absent")


# ---------------------------------------------------------------- storage

def check_storage() -> None:
    record(OK, "storage", "ROOT", str(ROOT))
    data = ROOT / "data"
    if data.is_symlink():
        target = os.readlink(data)
        real = Path(data).resolve()
        record(OK if real.exists() else FAIL, "storage", "data symlink",
               f"-> {target} ({'resolves' if real.exists() else 'BROKEN'})")
        record(OK if str(real).startswith("/Volumes/") else WARN, "storage", "data volume",
               "on external volume" if str(real).startswith("/Volumes/") else f"NOT external: {real}")
    elif data.is_dir():
        record(WARN, "storage", "data", "real directory, not a symlink — bulk may be on the SSD")
    else:
        record(FAIL, "storage", "data", "missing")
    for label, path in (("ssd", ROOT), ("bulk", data)):
        try:
            u = shutil.disk_usage(path)
            record(OK if u.free > 20 * 2**30 else WARN, "storage", f"{label} free",
                   f"{u.free/2**30:.1f} GiB")
        except OSError as exc:
            record(FAIL, "storage", f"{label} free", str(exc))
    # iCloud guard: the project must not live under a synced path
    bad = ("/Desktop/", "/Documents/", "Mobile Documents")
    record(FAIL if any(b in str(ROOT) for b in bad) else OK, "storage", "not iCloud-synced",
           str(ROOT))


# ---------------------------------------------------------------- ancestor

def check_frozen_build() -> None:
    frozen_path = PLAN / "ODYSSEY_FROZEN_BUILD.json"
    if not frozen_path.exists():
        record(FAIL, "ancestor", "frozen build", f"missing {frozen_path}")
        return
    frozen = json.loads(frozen_path.read_text())
    record(OK, "ancestor", "frozen build sha256", frozen.get("sha256", "?")[:16])
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from substrate import odyssey_transition  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "ancestor", "drift check", f"cannot import odyssey_transition: {exc}")
        return
    paths = {**odyssey_transition.implementation_inputs(ROOT),
             "odyssey_worker": ROOT / "src/substrate/odyssey_worker.py",
             "odyssey_authority": ROOT / "src/substrate/odyssey_authority.py"}
    pins = frozen.get("implementation_sha256", {})
    drifted = []
    for name, expected in pins.items():
        src = paths.get(name)
        if src is None or not src.is_file():
            drifted.append(f"{name}:MISSING"); continue
        h = hashlib.sha256()
        with open(src, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        if h.hexdigest() != expected:
            drifted.append(name)
    record(OK if not drifted else WARN, "ancestor", f"frozen drift ({len(pins)} pinned)",
           "zero drift — ancestral gates can still seal" if not drifted
           else f"DRIFTED: {drifted} (expected during Ascension engineering, fatal after freeze)")


def check_activation() -> None:
    sealed = PLAN / "ODYSSEY_7D.authority.json"
    record(OK if not sealed.exists() else WARN, "ancestor", "launch authority",
           "absent — Odyssey cannot launch" if not sealed.exists() else f"PRESENT at {sealed}")
    pre = ROOT / "runs/substrate/odyssey7d/ODYSSEY_PREFLIGHT.json"
    if pre.exists():
        d = json.loads(pre.read_text())
        good = d.get("launch_allowed") is False and d.get("activation") is False
        record(OK if good else FAIL, "ancestor", "activation/launch",
               f"activation={d.get('activation')} launch_allowed={d.get('launch_allowed')} status={d.get('status')}")
    else:
        record(WARN, "ancestor", "preflight", "absent (untracked artifact)")


def main() -> int:
    check_git()
    check_python()
    check_tools()
    check_storage()
    check_frozen_build()
    check_activation()

    width = max(len(c) for _, _, c, _ in results) + 2
    area = None
    for sev, a, check, detail in results:
        if a != area:
            print(f"\n[{a}]")
            area = a
        mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[sev]
        print(f"{mark}{check:<{width}}{detail}")
    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    print(f"\n{len(results)} checks — {len(results)-len(fails)-len(warns)} ok, {len(warns)} warn, {len(fails)} FAIL")
    if fails:
        print("\nFAILING:")
        for _, a, c, d in fails:
            print(f"  {a}/{c}: {d}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
