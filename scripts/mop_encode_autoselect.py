#!/usr/bin/env python
"""Encode device auto-select (Tier 4.1): microbench a tiny V-JEPA 2 encode on CPU and MPS and pick the
faster, so the Studio does not hand-choose the encode device. The M3-Pro microbench found CPU 13.7 vs
MPS 821 s/clip (MPS paged at 18 GB); the M1 Ultra with 128 GB may make MPS the winner, so this script
MEASURES on whatever box it runs on rather than assuming.

pick_encode_device() returns {winner, cpu_s_per_clip, mps} and (as a CLI) writes runs/mot/encode_device.json
so the encode step can read the winner. The Studio flow: run this once, then
`.venv/bin/python scripts/cache_real_encoder.py device=$(jq -r .winner runs/mot/encode_device.json) ...`.

Form (goal loop): no em or en dashes. MEASURE, never assume.

Usage:
  PYTHONPATH=src:scripts:. .venv/bin/python scripts/mop_encode_autoselect.py [--n-clips 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from compositional_under_nuisance import make_bound_nuisance_clip  # noqa: E402
from transformers import AutoModel  # noqa: E402

from mop.config import compose  # noqa: E402
from mop.studio.encode_scheduler import format_plan, plan_encode  # noqa: E402
from mop.studio.memory_envelope import MemorySampler  # noqa: E402

HF = "facebook/vjepa2-vitl-fpc64-256"


def _make_clips(n: int) -> list[torch.Tensor]:
    return [
        make_bound_nuisance_clip(i % 5, (i // 5) % 4, 4, torch.Generator().manual_seed(3000 + i))
        for i in range(n)
    ]


@torch.no_grad()
def _time_encode(model, clips, device: str, memory: MemorySampler | None = None) -> float:
    """Mean seconds per clip encoding on `device`. Raises on a device-specific failure (caught upstream)."""
    if memory is not None:
        memory.sample(f"{device}:before_model_to_device")
    m = model.to(device)
    if memory is not None:
        memory.sample(f"{device}:after_model_to_device")
    ts = []
    for i, c in enumerate(clips):
        t0 = time.perf_counter()
        m(pixel_values_videos=c.unsqueeze(0).to(device), skip_predictor=True)
        ts.append(time.perf_counter() - t0)
        if memory is not None:
            memory.sample(f"{device}:after_clip_{i}")
    return sum(ts) / len(ts)


def pick_encode_device(n_clips: int = 3, skip_mps: bool = False, allow_download: bool = False) -> dict:
    """Microbench CPU (always) and MPS (if available), return the faster as `winner`. MPS that errors or
    is unavailable is recorded and CPU wins by default. CUDA is reported as the winner unconditionally on
    a box that has it, since it is the intended fast path and a full timing is unnecessary to prefer it.
    skip_mps records MPS availability WITHOUT timing it (for the laptop smoke, where the 18 GB MPS path
    pages badly, about 821 s/clip; the Studio runs without skip_mps to get the real 128 GB measurement)."""
    memory = MemorySampler("mop_encode_autoselect")
    memory.sample("start")
    try:
        model = AutoModel.from_pretrained(
            HF,
            local_files_only=not allow_download,
            dtype=torch.float32,
        ).eval()
    except Exception as e:  # noqa: BLE001
        memory.sample(f"model_load_failed:{type(e).__name__}")
        return {
            "winner": "blocked",
            "cpu_s_per_clip": None,
            "mps": "not-tested (model load failed)",
            "n_clips": n_clips,
            "error": {
                "stage": "model_load",
                "type": type(e).__name__,
                "detail": str(e).splitlines()[0],
                "allow_download": bool(allow_download),
            },
            "memory_envelope": memory.summary(),
        }
    for p in model.parameters():
        p.requires_grad_(False)
    memory.sample("model_loaded_cpu")
    clips = _make_clips(n_clips)
    memory.sample("clips_ready")

    if torch.cuda.is_available():
        memory.sample("cuda_present_not_timed")
        return {
            "winner": "cuda",
            "cpu_s_per_clip": None,
            "mps": "not-tested (cuda present)",
            "n_clips": n_clips,
            "memory_envelope": memory.summary(),
        }

    cpu_s = round(_time_encode(model, clips, "cpu", memory), 3)
    mps: str | float
    if not torch.backends.mps.is_available():
        mps = "unavailable"
        winner = "cpu"
        memory.sample("mps_unavailable")
    elif skip_mps:
        mps = "available-not-timed (skip_mps; the Studio re-runs to time MPS at 128 GB)"
        winner = "cpu"
        memory.sample("mps_available_not_timed")
    else:
        try:
            mps_s = round(
                _time_encode(model, clips[:1], "mps", memory), 3
            )  # one clip; MPS failure/paging surfaces here
            mps = mps_s
            winner = "mps" if mps_s < cpu_s else "cpu"
        except Exception as e:  # noqa: BLE001
            mps = f"failed:{type(e).__name__}"
            winner = "cpu"
            memory.sample(f"mps_failed:{type(e).__name__}")
    memory.sample("finished")
    return {
        "winner": winner,
        "cpu_s_per_clip": cpu_s,
        "mps": mps,
        "n_clips": n_clips,
        "memory_envelope": memory.summary(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument(
        "--skip-mps", action="store_true", help="record MPS availability without timing it (laptop smoke)"
    )
    ap.add_argument("--out", type=str, default=str(_ROOT / "runs" / "mot" / "encode_device.json"))
    ap.add_argument("--schedule-out", type=str, default=str(_ROOT / "runs" / "mot" / "encode_schedule.json"))
    ap.add_argument("--profile", default="m3pro-local-max")
    ap.add_argument("--planned-clips", type=int, default=64)
    ap.add_argument("--encoder", default="vjepa2_vitl_fpc64_256")
    ap.add_argument("--dense", action="store_true", help="plan dense-token cache footprint")
    ap.add_argument("--cpu-workers", type=int, default=None, help="override profile CPU worker default")
    ap.add_argument(
        "--allow-download",
        action="store_true",
        help="allow transformers to fetch missing model files; default is local cache only",
    )
    args = ap.parse_args()
    result = pick_encode_device(args.n_clips, skip_mps=args.skip_mps, allow_download=args.allow_download)
    enc_cfg = compose([f"encoder={args.encoder}", "device=cpu"]).encoder
    enc_dict = {
        "name": str(enc_cfg.name),
        "embed_dim": int(enc_cfg.embed_dim),
        "dense": bool(getattr(enc_cfg, "dense", False)),
    }
    schedule = plan_encode(
        profile_name=args.profile,
        benchmark=result,
        encoder_config=enc_dict,
        requested_clips=args.planned_clips,
        dense=bool(args.dense),
        cpu_workers=args.cpu_workers,
    )
    result["schedule"] = {
        "ok_to_launch": schedule["ok_to_launch"],
        "winner": schedule["winner"],
        "effective_clips": schedule["effective_clips"],
        "cache_estimate": schedule["cache_estimate"],
        "blocked_reasons": schedule["blocked_reasons"],
        "next_command": schedule["next_command"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    sched_out = Path(args.schedule_out)
    sched_out.parent.mkdir(parents=True, exist_ok=True)
    sched_out.write_text(json.dumps(schedule, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nWINNER: {result['winner']} (wrote {out})")
    print(f"SCHEDULE: {format_plan(schedule)} (wrote {sched_out})")
    return 0 if schedule["ok_to_launch"] and result["winner"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
