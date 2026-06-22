"""Studio readiness doctor. One command answers: is this machine ready to run the campaign at
scale on Apple Silicon. It probes the things that actually break a real run (wrong python,
torch without MPS, no disk, no video decoder, a placeholder encoder mistaken for real, a
malformed leg) and reports each as a green/red check with a detail line. It is deliberately
non-fatal where the world is allowed to be incomplete this session: a missing video backend or
an unreachable HuggingFace are REPORTED, not failures, because the cached-latent path runs
without either and we never download weights here.

doctor() -> {checks:[{name,ok,detail}], all_ok, summary}. render_md() turns it into a table.
Nothing here trains, downloads, or runs long: every check is cheap (seconds, no network except
one short best-effort metadata ping that is caught and swallowed).
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable

from omegaconf import OmegaConf

from .config import REPO_ROOT
from .devices import apple_silicon_info

# the V-JEPA 2 default encoder, used only for a metadata reachability ping (NEVER downloaded)
_HF_PROBE_ID = "facebook/vjepa2-vitl-fpc64-256"
_HF_TIMEOUT_S = 4.0

# checks the doctor must always report (the test pins these names)
CHECK_NAMES = (
    "python",
    "torch",
    "apple_silicon",
    "disk_space",
    "video_backend",
    "huggingface",
    "encoders",
    "cache_write",
    "config_validation",
)


def _check(name: str, fn: Callable[[], tuple[bool, str]]) -> dict:
    """Run one probe. A probe returns (ok, detail); a crash inside it is itself a failed check,
    never an exception out of doctor()."""
    try:
        ok, detail = fn()
    except Exception as e:  # a probe that throws is a red check, not a crashed doctor
        ok, detail = False, f"{type(e).__name__}: {e}"
    return {"name": name, "ok": bool(ok), "detail": detail}


def _check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)  # project floor
    return ok, f"{v.major}.{v.minor}.{v.micro}"


def _check_torch() -> tuple[bool, str]:
    try:
        import torch
    except Exception as e:
        return False, f"torch not importable: {type(e).__name__}"
    avail = bool(torch.backends.mps.is_available())
    built = bool(torch.backends.mps.is_built())
    # torch present is the pass condition. MPS absence is reported in the detail, not a failure
    # (cpu and cuda are valid devices too); on Apple Silicon the absence is the thing to notice.
    return True, f"torch {torch.__version__}; mps available={avail} built={built}"


def _check_apple_silicon() -> tuple[bool, str]:
    info = apple_silicon_info()
    if not info.get("is_apple_silicon"):
        # not the native target, but not an error: cuda/cpu boxes are supported paths
        return True, "not Apple Silicon (cuda/cpu path)"
    chip = info.get("chip", "Apple Silicon")
    p, e = info.get("performance_cores"), info.get("efficiency_cores")
    mem = info.get("unified_memory_gb")
    return True, f"{chip}: {p}P/{e}E cores, {mem} GB unified, mps={info.get('mps_available')}"


def _check_disk_space() -> tuple[bool, str]:
    du = shutil.disk_usage(REPO_ROOT)
    free_gb = du.free / 1e9
    # latent caches + run artifacts are small, but real video ingestion wants headroom
    ok = free_gb >= 5.0
    return ok, f"{free_gb:.1f} GB free of {du.total / 1e9:.0f} GB at repo"


def _check_video_backend() -> tuple[bool, str]:
    """Report which decode backend is present (torchvision.io, then decord). Absence is REPORTED,
    not a failure: the cached-latent path needs no video decoder and most runs use the cache."""
    for mod, label in (("torchvision.io", "torchvision.io"), ("decord", "decord")):
        try:
            __import__(mod)
            return True, f"present: {label}"
        except Exception:
            continue
    return True, "none (cache path ok; install `.[video]` or decord for real video)"


def _check_huggingface() -> tuple[bool, str]:
    """OPTIONAL, non-fatal: a short metadata ping to confirm HF is reachable, WITHOUT downloading
    any weights. Any failure (offline, timeout, repo gated) is caught and reported as unreachable.
    Always ok=True: HF reachability never blocks the doctor (we do not download here)."""
    try:
        from huggingface_hub import HfApi
    except Exception as e:
        return True, f"huggingface_hub absent ({type(e).__name__}); skipped (non-fatal)"
    try:
        info = HfApi().model_info(_HF_PROBE_ID, timeout=_HF_TIMEOUT_S)
        sha = (getattr(info, "sha", "") or "")[:8]
        return True, f"reachable: {_HF_PROBE_ID}@{sha} (metadata only, no download)"
    except Exception as e:
        return True, f"unreachable ({type(e).__name__}); non-fatal, cache path unaffected"


def _check_encoders() -> tuple[bool, str]:
    """List configs/encoder/*.yaml with name, embed_dim, and the available flag. The check passes
    if every config is well-formed (positive embed_dim); the available flag is informational
    (some encoders, e.g. V-JEPA 2.1, are not yet published on HF and are deferred)."""
    cdir = REPO_ROOT / "configs" / "encoder"
    files = sorted(cdir.glob("*.yaml"))
    if not files:
        return False, "no encoder configs found"
    rows, ok = [], True
    for f in files:
        c = OmegaConf.load(f)
        name = str(OmegaConf.select(c, "name", default=f.stem))
        dim = int(OmegaConf.select(c, "embed_dim", default=0))
        avail = bool(OmegaConf.select(c, "available", default=True))
        if dim <= 0:
            ok = False
        rows.append(f"{name}(d={dim},{'real-ready' if avail else 'deferred'})")
    return ok, f"{len(files)} configs: " + ", ".join(rows)


def _check_cache_write() -> tuple[bool, str]:
    """Write and delete a tmp file under data/cache to prove the cache dir is writable (this is
    where every latent store lands). Cleans up after itself."""
    cache_dir = REPO_ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    probe = cache_dir / ".studio_doctor_write_test"
    probe.write_bytes(b"ok")
    wrote = probe.read_bytes() == b"ok"
    probe.unlink()
    return wrote and not probe.exists(), f"write+delete ok under {cache_dir.relative_to(REPO_ROOT)}"


def _check_config_validation() -> tuple[bool, str]:
    """Run the harness validator over every encoder/experiment config and queued leg. Clean ==
    zero problems. Lists the first few problems in the detail when not clean."""
    from .harness.validate import check_all

    problems = check_all()
    if not problems:
        return True, "0 problems (encoders, experiments, legs all valid)"
    head = "; ".join(f"{p['where']}: {p['problem']}" for p in problems[:3])
    more = f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""
    return False, f"{len(problems)} problems: {head}{more}"


_PROBES: tuple[tuple[str, Callable[[], tuple[bool, str]]], ...] = (
    ("python", _check_python),
    ("torch", _check_torch),
    ("apple_silicon", _check_apple_silicon),
    ("disk_space", _check_disk_space),
    ("video_backend", _check_video_backend),
    ("huggingface", _check_huggingface),
    ("encoders", _check_encoders),
    ("cache_write", _check_cache_write),
    ("config_validation", _check_config_validation),
)


def doctor() -> dict:
    """Run every readiness check. Returns {checks, all_ok, summary}. Never raises: a probe that
    throws becomes a failed check. all_ok is the AND over every check's ok."""
    checks = [_check(name, fn) for name, fn in _PROBES]
    passed = sum(1 for c in checks if c["ok"])
    return {
        "checks": checks,
        "all_ok": all(c["ok"] for c in checks),
        "summary": {"total": len(checks), "passed": passed, "failed": len(checks) - passed},
    }


def render_md(report: dict) -> str:
    """A readable Markdown table of the report: one row per check plus a summary line. Contains
    every check name so it is a faithful, greppable record of what the doctor saw."""
    s = report["summary"]
    head = "STUDIO READY" if report["all_ok"] else "NOT READY"
    lines = [
        "# Studio readiness doctor",
        "",
        f"**{head}** ({s['passed']}/{s['total']} checks passed, {s['failed']} failed)",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
    ]
    for c in report["checks"]:
        detail = str(c["detail"]).replace("|", "\\|")  # keep the table cell intact
        lines.append(f"| {c['name']} | {'ok' if c['ok'] else 'FAIL'} | {detail} |")
    lines.append("")
    return "\n".join(lines)
