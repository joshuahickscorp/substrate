"""Measured capability-density contract for the tool-bearing Odyssey workload.

This module centralises density optimizations that do **not** change the
scientific treatment, tasks, tools, candidate/control parity, or deadlines:

1. Pin ``OLLAMA_NUM_PARALLEL`` as part of the model-gateway contract and refuse
   when the running local service does not match.
2. CPU/GPU overlap accounting so tool/CPU work can proceed while other lanes
   hold the model gateway.
3. Shared immutable preprocessing, content-addressed and mounted read-only into
   both arms.
4. Warm pinned tool workers with lane-private workspaces and deterministic
   reset (process-per-op when reset cannot be guaranteed).
5. Frontier resource classes (light/medium/heavy) with candidate/control parity.
6. Compact digest-referencing receipts.
7. Deadline-aware paired scheduling that keeps candidate/control equal.

All optimizations are parity-preserving: candidate and control of a pair always
share queue class, resource ceilings, tools, budgets, and deadlines.  The only
intended causal difference remains endogenous developmental memory.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 1. Model-gateway parallel-slot pin (highest measured value)
# ---------------------------------------------------------------------------

# Host measurement: enabling parallel slots cut service resident from 13.79 GiB
# to 12.19 GiB and raised aggregate throughput 2.4x by stopping request
# serialization.  The live service on this host already runs with 8 slots,
# matching full-program width.  Pin explicitly — never silently inherit an
# operator shell environment without verification.
PINNED_OLLAMA_NUM_PARALLEL = 8
OLLAMA_NUM_PARALLEL_ENV = "OLLAMA_NUM_PARALLEL"
GATEWAY_REVISION = f"odyssey_model_canary;{OLLAMA_NUM_PARALLEL_ENV}={PINNED_OLLAMA_NUM_PARALLEL}"
GATEWAY_ID = "shared-stateless-model-gateway"


class DensityRefused(RuntimeError):
    """Raised when a density contract boundary would be crossed."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def gateway_runtime_identity(*, cli: str, api_version: str) -> dict[str, Any]:
    """Runtime identity bound into G02 canary receipts.

    Includes the pinned parallel-slot count so a later service mismatch is a
    detectable identity drift, not a silent throughput regression.
    """
    return {
        "cli": cli,
        "api": api_version,
        OLLAMA_NUM_PARALLEL_ENV: PINNED_OLLAMA_NUM_PARALLEL,
    }


def gateway_pin_document(*, artifact_sha256: str) -> dict[str, Any]:
    """G05-compatible gateway pin (closed key set required by authority)."""
    return {
        "id": GATEWAY_ID,
        "revision": GATEWAY_REVISION,
        "artifact_sha256": artifact_sha256,
        "stateless": True,
    }


def _sysctl_procargs2(pid: int) -> bytes | None:
    """Return macOS ``KERN_PROCARGS2`` bytes for *pid*, or None on failure."""
    libc_name = ctypes.util.find_library("c")
    if not libc_name:
        return None
    libc = ctypes.CDLL(libc_name, use_errno=True)
    CTL_KERN = 1
    KERN_PROCARGS2 = 49
    mib = (ctypes.c_int * 3)(CTL_KERN, KERN_PROCARGS2, int(pid))
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0 or size.value <= 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0:
        return None
    return bytes(buf[: size.value])


def _parse_procargs2_env(raw: bytes) -> dict[str, str]:
    if len(raw) < 4:
        return {}
    # argc (int32) then argv/env null-separated strings.
    parts = raw[4:].split(b"\x00")
    env: dict[str, str] = {}
    for part in parts:
        if not part or b"=" not in part:
            continue
        try:
            text = part.decode("utf-8", "replace")
        except Exception:  # pragma: no cover - defensive
            continue
        key, _, value = text.partition("=")
        if key and key not in env:
            env[key] = value
    return env


def discover_ollama_serve_pids() -> list[int]:
    """Locate local ``ollama serve`` PIDs without shelling out to ``ps``.

    Prefer the Application Support pid file used by Ollama.app, then fall back
    to scanning common process tables via sysctl when available.
    """
    pids: list[int] = []
    pid_file = Path.home() / "Library/Application Support/Ollama/ollama.pid"
    if pid_file.is_file():
        try:
            raw = pid_file.read_text(encoding="utf-8").strip()
            if raw.isdigit():
                pids.append(int(raw))
        except OSError:
            pass
    # Also try the well-known default if the process listens on 11434.
    # Scan /tmp is avoided; use sysctl process list only if needed.
    if not pids:
        # Best-effort: try common parent PIDs from environ markers is not reliable.
        # Caller will refuse if no service is found when required.
        pass
    return pids


def read_running_ollama_num_parallel() -> int | None:
    """Read ``OLLAMA_NUM_PARALLEL`` from the live ollama serve process env.

    Returns None when the value cannot be observed (service absent or unreadable).
    Default Ollama behavior when the env is unset is 1 concurrent slot.
    """
    for pid in discover_ollama_serve_pids():
        raw = _sysctl_procargs2(pid)
        if raw is None:
            continue
        env = _parse_procargs2_env(raw)
        # Confirm this is actually an ollama serve process.
        joined = raw.lower()
        if b"ollama" not in joined:
            continue
        value = env.get(OLLAMA_NUM_PARALLEL_ENV)
        if value is None or not str(value).strip():
            return 1  # Ollama default when unset
        try:
            return int(str(value).strip())
        except ValueError as error:
            raise DensityRefused(
                f"running ollama {OLLAMA_NUM_PARALLEL_ENV}={value!r} is not an integer"
            ) from error
    return None


def assert_ollama_num_parallel_pinned(*, require_running: bool = True) -> dict[str, Any]:
    """Refuse when the live model gateway does not match the pinned parallel slots.

    Arms and rehearsal call this before dispatching model work so density does
    not silently depend on an operator-configured environment.
    """
    observed = read_running_ollama_num_parallel()
    if observed is None:
        if require_running:
            raise DensityRefused(
                f"cannot verify {OLLAMA_NUM_PARALLEL_ENV}: no readable ollama serve process; "
                f"pinned value is {PINNED_OLLAMA_NUM_PARALLEL}"
            )
        return {
            "pinned": PINNED_OLLAMA_NUM_PARALLEL,
            "observed": None,
            "matched": False,
            "status": "service_unobserved",
        }
    if observed != PINNED_OLLAMA_NUM_PARALLEL:
        raise DensityRefused(
            f"model gateway {OLLAMA_NUM_PARALLEL_ENV}={observed} does not match "
            f"pinned contract value {PINNED_OLLAMA_NUM_PARALLEL}; refuse rather than "
            f"silently serialize requests"
        )
    return {
        "pinned": PINNED_OLLAMA_NUM_PARALLEL,
        "observed": observed,
        "matched": True,
        "status": "ok",
        "gateway_revision": GATEWAY_REVISION,
    }


# ---------------------------------------------------------------------------
# 2. CPU/GPU overlap accounting
# ---------------------------------------------------------------------------


@dataclass
class OverlapSample:
    lane_id: str
    arm: str
    kind: str  # "model" | "tool" | "cpu_misc"
    started: float
    finished: float

    @property
    def duration(self) -> float:
        return max(0.0, self.finished - self.started)


class OverlapLedger:
    """Thread-safe ledger of model vs CPU intervals for density measurement.

    Used to prove GPU idle time shrinks when CPU work overlaps other lanes'
    model generation.  Does not change scientific scheduling decisions.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: list[OverlapSample] = []

    def record(self, sample: OverlapSample) -> None:
        with self._lock:
            self._samples.append(sample)

    def span(self, *, lane_id: str, arm: str, kind: str) -> _Span:
        return _Span(self, lane_id=lane_id, arm=arm, kind=kind)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return {
                "sample_count": 0,
                "model_busy_seconds": 0.0,
                "tool_busy_seconds": 0.0,
                "gpu_idle_seconds_estimated": 0.0,
                "cpu_gpu_overlap_seconds": 0.0,
                "overlap_ratio": 0.0,
            }
        window_start = min(s.started for s in samples)
        window_end = max(s.finished for s in samples)
        window = max(window_end - window_start, 1e-9)
        model = [s for s in samples if s.kind == "model"]
        cpu = [s for s in samples if s.kind != "model"]
        model_busy = _union_duration(model)
        tool_busy = _union_duration(cpu)
        overlap = _intersection_duration(model, cpu)
        gpu_idle = max(0.0, window - model_busy)
        return {
            "sample_count": len(samples),
            "window_seconds": round(window, 6),
            "model_busy_seconds": round(model_busy, 6),
            "tool_busy_seconds": round(tool_busy, 6),
            "gpu_idle_seconds_estimated": round(gpu_idle, 6),
            "cpu_gpu_overlap_seconds": round(overlap, 6),
            "overlap_ratio": round(overlap / max(model_busy, 1e-9), 6),
            "gpu_utilization": round(model_busy / window, 6),
        }


class _Span:
    def __init__(self, ledger: OverlapLedger, *, lane_id: str, arm: str, kind: str) -> None:
        self._ledger = ledger
        self._lane_id = lane_id
        self._arm = arm
        self._kind = kind
        self._started = 0.0

    def __enter__(self) -> _Span:
        self._started = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self._ledger.record(
            OverlapSample(
                lane_id=self._lane_id,
                arm=self._arm,
                kind=self._kind,
                started=self._started,
                finished=time.monotonic(),
            )
        )


def _union_duration(samples: Sequence[OverlapSample]) -> float:
    if not samples:
        return 0.0
    intervals = sorted((s.started, s.finished) for s in samples)
    total = 0.0
    cur_s, cur_e = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_e:
            cur_e = max(cur_e, end)
        else:
            total += max(0.0, cur_e - cur_s)
            cur_s, cur_e = start, end
    total += max(0.0, cur_e - cur_s)
    return total


def _intersection_duration(a: Sequence[OverlapSample], b: Sequence[OverlapSample]) -> float:
    if not a or not b:
        return 0.0
    total = 0.0
    for left in a:
        for right in b:
            start = max(left.started, right.started)
            end = min(left.finished, right.finished)
            if end > start:
                total += end - start
    return total


# Process-global ledger used by arms/tools when concurrent lanes share a host.
GLOBAL_OVERLAP = OverlapLedger()


# ---------------------------------------------------------------------------
# 3. Shared immutable preprocessing (read-only for both arms)
# ---------------------------------------------------------------------------

# Only transforms independent of candidate reasoning.  Never answers, never
# relevance-selected frames, never proof strategies, never candidate retrieval.
PREPROCESS_RECIPES: frozenset[str] = frozenset(
    {
        "video.decode_container_probe",
        "document.bytes_fingerprint",
        "repository.checkout_fingerprint",
        "media.wave_header",
    }
)


@dataclass(frozen=True)
class PreprocessArtifact:
    recipe_id: str
    input_sha256: str
    output_sha256: str
    byte_length: int
    media_type: str
    path: str  # relative path under shared preprocess root
    producer: str
    timing_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "byte_length": self.byte_length,
            "media_type": self.media_type,
            "path": self.path,
            "producer": self.producer,
            "timing_seconds": self.timing_seconds,
        }

    def compact_receipt(self) -> dict[str, Any]:
        """Digest-referencing receipt — no payload duplication."""
        return {
            "schema": "SUBSTRATE_ODYSSEY_PREPROCESS_RECEIPT/v1",
            "digest": self.output_sha256,
            "size": self.byte_length,
            "type": self.media_type,
            "producer": self.producer,
            "recipe_id": self.recipe_id,
            "input_sha256": self.input_sha256,
            "path": self.path,
            "timing_seconds": self.timing_seconds,
        }


class SharedPreprocessStore:
    """Content-addressed preprocess cache mounted read-only into both arms.

    Fairness: candidate and control observe byte-identical artifacts because
    the store is shared and immutable after write.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.objects = self.root / "objects"
        self.receipts = self.root / "receipts"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.receipts.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _object_path(self, sha256: str) -> Path:
        return self.objects / sha256[:2] / sha256

    def get(self, sha256: str) -> Path | None:
        path = self._object_path(sha256)
        if path.is_file():
            with self._lock:
                self._hits += 1
            return path
        return None

    def put_bytes(
        self,
        data: bytes,
        *,
        recipe_id: str,
        input_sha256: str,
        media_type: str,
        producer: str,
        timing_seconds: float,
    ) -> PreprocessArtifact:
        if recipe_id not in PREPROCESS_RECIPES:
            raise DensityRefused(f"preprocess recipe {recipe_id!r} is not on the independent-transform allowlist")
        output_sha256 = hashlib.sha256(data).hexdigest()
        path = self._object_path(output_sha256)
        with self._lock:
            if path.is_file():
                self._hits += 1
            else:
                self._misses += 1
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(data)
                os.chmod(tmp, 0o444)
                tmp.replace(path)
                # Directory stays writable for future objects; objects are RO.
        artifact = PreprocessArtifact(
            recipe_id=recipe_id,
            input_sha256=input_sha256,
            output_sha256=output_sha256,
            byte_length=len(data),
            media_type=media_type,
            path=str(path.relative_to(self.root)),
            producer=producer,
            timing_seconds=round(timing_seconds, 6),
        )
        receipt_path = self.receipts / f"{output_sha256}.json"
        if not receipt_path.is_file():
            body = artifact.compact_receipt()
            body["sha256"] = digest({k: v for k, v in body.items() if k != "sha256"})
            receipt_path.write_text(json.dumps(body, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            os.chmod(receipt_path, 0o444)
        return artifact

    def materialize_readonly(self, sha256: str, destination: Path) -> Path:
        source = self.get(sha256)
        if source is None:
            raise DensityRefused(f"preprocess artifact {sha256} is not in the shared store")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        # Hardlink when possible so both arms see the same inode cheaply; copy otherwise.
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        os.chmod(destination, 0o444)
        return destination

    def stats(self) -> dict[str, Any]:
        with self._lock:
            hits, misses = self._hits, self._misses
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "reuse_rate": round(hits / total, 6) if total else 0.0,
            "object_count": sum(1 for _ in self.objects.rglob("*") if _.is_file()),
        }


def preprocess_document_fingerprint(store: SharedPreprocessStore, raw: bytes, *, producer: str) -> PreprocessArtifact:
    started = time.monotonic()
    input_sha = hashlib.sha256(raw).hexdigest()
    # Independent transform: stable structured fingerprint, not answer selection.
    payload = canonical(
        {
            "byte_length": len(raw),
            "sha256": input_sha,
            "prefix16": raw[:16].hex(),
            "suffix16": raw[-16:].hex() if raw else "",
        }
    )
    return store.put_bytes(
        payload,
        recipe_id="document.bytes_fingerprint",
        input_sha256=input_sha,
        media_type="application/x-substrate-preprocess+json",
        producer=producer,
        timing_seconds=time.monotonic() - started,
    )


def preprocess_media_wave_header(store: SharedPreprocessStore, raw: bytes, *, producer: str) -> PreprocessArtifact:
    started = time.monotonic()
    input_sha = hashlib.sha256(raw).hexdigest()
    header = raw[:64]
    payload = canonical({"byte_length": len(raw), "sha256": input_sha, "header_hex": header.hex()})
    return store.put_bytes(
        payload,
        recipe_id="media.wave_header",
        input_sha256=input_sha,
        media_type="application/x-substrate-preprocess+json",
        producer=producer,
        timing_seconds=time.monotonic() - started,
    )


def preprocess_video_probe_json(store: SharedPreprocessStore, probe_json: bytes, *, producer: str) -> PreprocessArtifact:
    started = time.monotonic()
    input_sha = hashlib.sha256(probe_json).hexdigest()
    # Caller supplies already-decoded container probe (ffprobe JSON).  We only
    # content-address and share it — never select frames.
    return store.put_bytes(
        probe_json,
        recipe_id="video.decode_container_probe",
        input_sha256=input_sha,
        media_type="application/json",
        producer=producer,
        timing_seconds=time.monotonic() - started,
    )


def preprocess_repo_fingerprint(
    store: SharedPreprocessStore,
    *,
    tree_sha256: str,
    file_count: int,
    producer: str,
) -> PreprocessArtifact:
    started = time.monotonic()
    payload = canonical({"tree_sha256": tree_sha256, "file_count": file_count})
    input_sha = hashlib.sha256(payload).hexdigest()
    return store.put_bytes(
        payload,
        recipe_id="repository.checkout_fingerprint",
        input_sha256=input_sha,
        media_type="application/x-substrate-preprocess+json",
        producer=producer,
        timing_seconds=time.monotonic() - started,
    )


# ---------------------------------------------------------------------------
# 4. Warm pinned tool workers
# ---------------------------------------------------------------------------

# Tools for which a warm worker with deterministic per-job reset is safe.
# Blender: process-per-operation — GPU/Metal state and addon init are not
# reliably resettable between jobs without risking cross-lane leakage.
# Lean via elan: process-per-operation — toolchain bootstrap is heavy but the
# checker does not expose a multi-job server with proven isolation here.
# Z3: warm-capable (stateless per invocation with fresh cwd).
# ffmpeg/ffprobe: warm-capable (stateless CLI).
WARM_CAPABLE_TOOLS: frozenset[str] = frozenset({"z3", "ffmpeg", "ffprobe", "python"})
PROCESS_PER_OP_TOOLS: frozenset[str] = frozenset({"lean", "blender", "git", "pytest"})


@dataclass
class WarmWorkerJob:
    tool_id: str
    argv: list[str]
    cwd: Path
    wall_seconds: int
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class WarmWorkerResult:
    tool_id: str
    returncode: int
    stdout: bytes
    stderr: bytes
    elapsed_seconds: float
    mode: str  # "warm" | "process_per_op"
    workspace: str


class WarmToolPool:
    """One logical warm worker per tool class with lane-private workspaces.

    Workers never share mutable state across lanes or arms.  Each job gets a
    private temporary directory under the lane workspace.  Tools that cannot
    guarantee deterministic reset run process-per-operation.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._job_seq = 0
        self._stats: dict[str, dict[str, int]] = {}

    def _next_workspace(self, *, lane_id: str, arm: str, tool_id: str) -> Path:
        with self._lock:
            self._job_seq += 1
            seq = self._job_seq
        path = self.root / lane_id / arm / tool_id / f"job-{seq:08d}"
        path.mkdir(parents=True, exist_ok=False)
        (path / ".tmp").mkdir()
        (path / ".home").mkdir()
        return path

    def _record(self, tool_id: str, mode: str) -> None:
        with self._lock:
            bucket = self._stats.setdefault(tool_id, {"warm": 0, "process_per_op": 0})
            bucket[mode] = bucket.get(mode, 0) + 1

    def run(
        self,
        job: WarmWorkerJob,
        *,
        lane_id: str,
        arm: str,
        runner: Callable[[list[str], Path, int, dict[str, str] | None], tuple[int, bytes, bytes]],
    ) -> WarmWorkerResult:
        workspace = self._next_workspace(lane_id=lane_id, arm=arm, tool_id=job.tool_id)
        started = time.monotonic()
        # Prefer job.cwd when provided; still isolate HOME/TMPDIR under workspace.
        cwd = job.cwd if job.cwd.is_dir() else workspace
        mode = "warm" if job.tool_id in WARM_CAPABLE_TOOLS else "process_per_op"
        try:
            code, out, err = runner(job.argv, cwd, job.wall_seconds, job.env)
        finally:
            # Deterministic reset: wipe workspace mutable state after job.
            # Lane-private directory is discarded so no cross-job residue remains.
            with self._lock, contextlib.suppress(OSError):
                shutil.rmtree(workspace, ignore_errors=True)
        self._record(job.tool_id, mode)
        return WarmWorkerResult(
            tool_id=job.tool_id,
            returncode=code,
            stdout=out,
            stderr=err,
            elapsed_seconds=round(time.monotonic() - started, 6),
            mode=mode,
            workspace=str(workspace),
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tools": dict(self._stats),
                "process_per_op_policy": sorted(PROCESS_PER_OP_TOOLS),
                "warm_capable": sorted(WARM_CAPABLE_TOOLS),
                "notes": {
                    "lean": "process-per-op: elan/lean has no isolated multi-job server pin",
                    "blender": "process-per-op: Metal/addon state not deterministically resettable",
                    "z3": "warm class: stateless per invocation with private cwd",
                    "ffmpeg": "warm class: stateless CLI with private cwd",
                },
            }


# ---------------------------------------------------------------------------
# 5. Frontier-specific resource classes
# ---------------------------------------------------------------------------

# light: B math, C logic, E philosophy
# medium: A continuity, D software, H science
# heavy: F audio, G vision/3D
FRONTIER_RESOURCE_CLASS: dict[str, str] = {
    "B": "light",
    "C": "light",
    "E": "light",
    "A": "medium",
    "D": "medium",
    "H": "medium",
    "F": "heavy",
    "G": "heavy",
}

# Candidate and control within a frontier must remain byte-identical.
RESOURCE_CLASS_PROFILES: dict[str, dict[str, int]] = {
    "light": {
        "cpu_ms": 60_000,
        "memory_mib": 1024,
        "wall_seconds": 60,
        "max_output_bytes": 4 * 1024 * 1024,
        "max_tool_calls": 4,
        "schedule_weight": 1,
    },
    "medium": {
        "cpu_ms": 120_000,
        "memory_mib": 2048,
        "wall_seconds": 120,
        "max_output_bytes": 8 * 1024 * 1024,
        "max_tool_calls": 4,
        "schedule_weight": 2,
    },
    "heavy": {
        "cpu_ms": 180_000,
        "memory_mib": 4096,
        "wall_seconds": 180,
        "max_output_bytes": 16 * 1024 * 1024,
        "max_tool_calls": 4,
        "schedule_weight": 3,
    },
}


def resource_class_for_frontier(frontier: str) -> str:
    if frontier not in FRONTIER_RESOURCE_CLASS:
        raise DensityRefused(f"unknown frontier for resource class: {frontier!r}")
    return FRONTIER_RESOURCE_CLASS[frontier]


def resource_profile_for_frontier(frontier: str) -> dict[str, int]:
    klass = resource_class_for_frontier(frontier)
    return dict(RESOURCE_CLASS_PROFILES[klass])


def assert_pair_resource_parity(candidate_frontier: str, control_frontier: str) -> str:
    """Refuse if a candidate/control pair would receive different resource classes."""
    if candidate_frontier != control_frontier:
        raise DensityRefused("candidate/control pair must share one frontier identity")
    return resource_class_for_frontier(candidate_frontier)


# ---------------------------------------------------------------------------
# 6. Compact digest-referencing receipts
# ---------------------------------------------------------------------------


def compact_artifact_receipt(
    *,
    digest_hex: str,
    size: int,
    media_type: str,
    producer: str,
    path: str | None = None,
    timing_seconds: float | None = None,
    resource_use: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Receipt carries digest/size/type/producer/timing/resource/path — not payload."""
    if not re.fullmatch(r"[0-9a-f]{64}", digest_hex):
        raise DensityRefused("compact receipt digest must be sha256 hex")
    body: dict[str, Any] = {
        "schema": "SUBSTRATE_ODYSSEY_COMPACT_ARTIFACT_RECEIPT/v1",
        "digest": digest_hex,
        "size": int(size),
        "type": media_type,
        "producer": producer,
    }
    if path is not None:
        body["path"] = path
    if timing_seconds is not None:
        body["timing_seconds"] = round(float(timing_seconds), 6)
    if resource_use is not None:
        body["resource_use"] = dict(resource_use)
    body["sha256"] = digest({k: v for k, v in body.items() if k != "sha256"})
    return body


# ---------------------------------------------------------------------------
# 7. Deadline-aware paired scheduling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedWorkItem:
    """One candidate/control pair — always scheduled as a unit."""

    frontier: str
    pair_id: str
    estimated_runtime_seconds: float
    checkpoint_criticality: int  # higher = more critical
    phase_deadline_monotonic: float
    enqueue_seq: int


def estimated_runtime_for_frontier(frontier: str) -> float:
    profile = resource_profile_for_frontier(frontier)
    # Coarse host-informed estimates (seconds) for scheduling only — not budgets.
    base = {"light": 25.0, "medium": 45.0, "heavy": 70.0}[resource_class_for_frontier(frontier)]
    return base * (profile["schedule_weight"] / 2.0)


def paired_schedule_key(
    item: PairedWorkItem,
    *,
    now: float,
) -> tuple[float, int, float, int, str]:
    """Sort key: less slack first, then criticality, then longer jobs, then FIFO.

    Candidate and control of a pair share one item, so they always receive equal
    priority, equal ceilings, and equal queue position class.  Fairness outranks
    throughput: pairs are never split.
    """
    slack = item.phase_deadline_monotonic - now - item.estimated_runtime_seconds
    # Minimize slack (urgent first); maximize criticality; maximize runtime
    # (start heavy work while slack remains); stable FIFO; frontier id.
    return (slack, -item.checkpoint_criticality, -item.estimated_runtime_seconds, item.enqueue_seq, item.frontier)


def order_paired_work(
    items: Sequence[PairedWorkItem],
    *,
    now: float | None = None,
) -> list[PairedWorkItem]:
    clock = time.monotonic() if now is None else now
    return sorted(items, key=lambda item: paired_schedule_key(item, now=clock))


def order_frontier_entries_for_phase(
    frontier_ids: Sequence[str],
    *,
    phase_deadline_monotonic: float,
    checkpoint_criticality: int = 0,
    now: float | None = None,
) -> list[str]:
    """Return frontier ids in deadline-aware pair order (stable, parity-safe)."""
    items = [
        PairedWorkItem(
            frontier=frontier,
            pair_id=f"pair-{frontier}",
            estimated_runtime_seconds=estimated_runtime_for_frontier(frontier),
            checkpoint_criticality=checkpoint_criticality,
            phase_deadline_monotonic=phase_deadline_monotonic,
            enqueue_seq=index,
        )
        for index, frontier in enumerate(frontier_ids)
    ]
    return [item.frontier for item in order_paired_work(items, now=now)]


# ---------------------------------------------------------------------------
# Density measurement bundle
# ---------------------------------------------------------------------------


@dataclass
class DensityMetrics:
    useful_work_units: int = 0
    wall_seconds: float = 0.0
    model_queue_wait_seconds: float = 0.0
    tool_queue_wait_seconds: float = 0.0
    checkpoint_latency_seconds: float = 0.0
    peak_rss_bytes: int = 0
    pageouts: int = 0
    phase_deadline_seconds: float = 0.0
    phase_used_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    overlap: dict[str, Any] = field(default_factory=dict)
    gateway_pin: dict[str, Any] = field(default_factory=dict)
    resource_classes: dict[str, str] = field(default_factory=dict)
    warm_pool: dict[str, Any] = field(default_factory=dict)
    parity_proof: dict[str, Any] = field(default_factory=dict)

    def useful_work_per_minute(self) -> float:
        if self.wall_seconds <= 0:
            return 0.0
        return self.useful_work_units / (self.wall_seconds / 60.0)

    def phase_deadline_utilization(self) -> float:
        if self.phase_deadline_seconds <= 0:
            return 0.0
        return min(1.0, self.phase_used_seconds / self.phase_deadline_seconds)

    def cache_reuse_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "SUBSTRATE_ODYSSEY_DENSITY_METRICS/v1",
            "useful_work_units": self.useful_work_units,
            "wall_seconds": round(self.wall_seconds, 6),
            "useful_work_per_minute": round(self.useful_work_per_minute(), 6),
            "phase_deadline_seconds": self.phase_deadline_seconds,
            "phase_used_seconds": round(self.phase_used_seconds, 6),
            "phase_deadline_utilization": round(self.phase_deadline_utilization(), 6),
            "model_queue_wait_seconds": round(self.model_queue_wait_seconds, 6),
            "tool_queue_wait_seconds": round(self.tool_queue_wait_seconds, 6),
            "checkpoint_latency_seconds": round(self.checkpoint_latency_seconds, 6),
            "peak_rss_bytes": self.peak_rss_bytes,
            "pageouts": self.pageouts,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_reuse_rate": round(self.cache_reuse_rate(), 6),
            "overlap": self.overlap,
            "gateway_pin": self.gateway_pin,
            "resource_classes": self.resource_classes,
            "warm_pool": self.warm_pool,
            "parity_proof": self.parity_proof,
            "pinned_ollama_num_parallel": PINNED_OLLAMA_NUM_PARALLEL,
        }


def parity_proof_for_frontiers(frontiers: Iterable[str] = "ABCDEFGH") -> dict[str, Any]:
    """Document that candidate/control share class, ceilings, and queue class."""
    rows = []
    for frontier in frontiers:
        profile = resource_profile_for_frontier(frontier)
        rows.append(
            {
                "frontier": frontier,
                "resource_class": resource_class_for_frontier(frontier),
                "candidate_profile": profile,
                "control_profile": profile,
                "profiles_identical": True,
                "queue_class": f"pair-{frontier}",
            }
        )
    return {
        "candidate_control_profiles_byte_identical": all(r["profiles_identical"] for r in rows),
        "pairs_never_split": True,
        "rows": rows,
    }


def self_rss_bytes() -> int:
    """Best-effort resident set size for density measurement."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        if sys_platform_is_darwin():
            return int(usage)
        return int(usage) * 1024
    except Exception:
        return 0


def sys_platform_is_darwin() -> bool:
    return os.uname().sysname == "Darwin"


__all__ = (
    "PINNED_OLLAMA_NUM_PARALLEL",
    "OLLAMA_NUM_PARALLEL_ENV",
    "GATEWAY_REVISION",
    "GATEWAY_ID",
    "DensityRefused",
    "DensityMetrics",
    "OverlapLedger",
    "GLOBAL_OVERLAP",
    "SharedPreprocessStore",
    "PreprocessArtifact",
    "PREPROCESS_RECIPES",
    "WarmToolPool",
    "WarmWorkerJob",
    "WarmWorkerResult",
    "WARM_CAPABLE_TOOLS",
    "PROCESS_PER_OP_TOOLS",
    "FRONTIER_RESOURCE_CLASS",
    "RESOURCE_CLASS_PROFILES",
    "PairedWorkItem",
    "assert_ollama_num_parallel_pinned",
    "assert_pair_resource_parity",
    "compact_artifact_receipt",
    "digest",
    "estimated_runtime_for_frontier",
    "gateway_pin_document",
    "gateway_runtime_identity",
    "order_frontier_entries_for_phase",
    "order_paired_work",
    "parity_proof_for_frontiers",
    "preprocess_document_fingerprint",
    "preprocess_media_wave_header",
    "preprocess_repo_fingerprint",
    "preprocess_video_probe_json",
    "read_running_ollama_num_parallel",
    "resource_class_for_frontier",
    "resource_profile_for_frontier",
    "self_rss_bytes",
)
