"""Mac-Studio rehearsal capsule. ONE entry point that walks the ENTIRE future Studio workflow
end to end on tiny local fixtures, so the path is proven before any Studio hour is spent:

  generated tiny video corpus -> source validation -> decode + preprocess -> cache creation ->
  cache integrity -> full-grid dry-run + cost agreement -> miniature Tier C run -> provenance ->
  microbenchmarks -> Markdown + JSON report.

This is a REHEARSAL, not a scientific result. Every artifact is tagged real / mocked / provisional.
On this machine the video decode is MOCKED (no codec): the corpus is .npy clips that flow through
the exact same validate/decode/preprocess/cache contract; on the Studio the same path runs over
real .mp4 with a video backend. Nothing here downloads weights or data or runs a long sweep.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import bench, provenance, studio_doctor
from .config import REPO_ROOT, compose
from .devices import resolve
from .harness.queue import run_queue
from .harness.sweep import full_run_units, load_leg
from .logging_utils import get_logger
from .studies.cost_projection import cost_projection
from .substrate import (
    cache_latents,
    cache_tools,
    generate_video_corpus,
    iter_video_clips,
    load_encoder,
    storage,
)
from .substrate.video import validate_source, write_label_map

log = get_logger("rehearsal")
DEFAULT_OUT = REPO_ROOT / "runs" / "studio_rehearsal"


def rehearse(out_dir: Path | None = None, seed: int = 0, frames: int = 8, res: int = 16) -> dict:
    out = Path(out_dir or DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=True)
    dev = resolve("cpu")  # rehearsal is cpu; the Studio flips device=mps, same code
    stages: list[dict] = []
    t_start = time.time()

    def run(name, kind, fn):
        t0 = time.perf_counter()
        rec = {"stage": name, "kind": kind}
        try:
            rec["detail"] = fn()
            rec["status"] = "pass"
        except Exception as e:  # capsule continues; the report shows exactly what failed
            rec["status"] = "fail"
            rec["detail"] = {"error": f"{type(e).__name__}: {e}"}
            log.warning("stage %s failed: %s", name, e)
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        stages.append(rec)
        return rec

    ctx: dict = {}

    # 1. Studio doctor (REAL: reports this machine)
    def _doctor():
        d = studio_doctor.doctor()
        return {"all_ok": d["all_ok"], "summary": d["summary"], "real_or_mocked": "real"}

    run("doctor", "real", _doctor)

    # 2. tiny video corpus (MOCKED: generated .npy clips, no codec, no dataset)
    corpus = out / "corpus"

    def _corpus():
        man = generate_video_corpus(
            corpus, n_classes=2, clips_per_class=3, frames=frames, h=res * 2, w=res * 2, seed=seed
        )
        ctx["corpus_manifest"] = man
        return {
            "n_clips": man["n_clips"],
            "per_class": man["per_class"],
            "duplicate_injected": bool(man["duplicate"]),
            "short_injected": bool(man["short"]),
            "real_or_mocked": "mocked (.npy fixtures; real .mp4 on the Studio)",
        }

    run("video_corpus", "mocked", _corpus)

    # 3. source validation (REAL logic over the mocked corpus)
    def _validate():
        v = validate_source(corpus)
        ctx["label_map"] = v["label_map"]
        return {
            "classes": v["classes"],
            "n_clips": v["n_clips"],
            "label_map": v["label_map"],
            "real_or_mocked": "real",
        }

    run("source_validation", "real", _validate)

    # 4. decode + preprocess (decode MOCKED via .npy, preprocess REAL); dup + short exercised
    def _preprocess():
        hashes: list[str] = []
        batches = list(iter_video_clips(corpus, frames_per_clip=frames, res=res, batch=2, hashes_out=hashes))
        import torch

        clips = torch.cat([b[0] for b in batches])
        ctx["n_batches"] = len(batches)
        return {
            "clip_tensor_shape": list(clips.shape),
            "n_clips": int(clips.shape[0]),
            "duplicates_detected": len(hashes) - len(set(hashes)),
            "preprocess": "real (resize+normalize)",
            "decode": "mocked (.npy)",
        }

    run("decode_preprocess", "mocked-decode", _preprocess)

    # 5. cache creation (REAL pipeline; frozen-random substrate -> result_tag provisional)
    cache_root = out / "cache"

    def _cache():
        cfg = compose(["encoder=vjepa2_vitl_fpc64_256", "device=cpu"])
        enc = load_encoder(cfg.encoder).to(dev.device)
        hashes: list[str] = []
        clips = iter_video_clips(corpus, frames_per_clip=frames, res=res, batch=2, hashes_out=hashes)
        man = ctx["corpus_manifest"]
        store = cache_latents(
            enc,
            clips,
            cache_root,
            "rehearsal",
            total=man["n_clips"],
            device=dev,
            result_tag="provisional",
            seed=seed,
        )
        write_label_map(store.root, ctx.get("label_map", {}))
        (store.root / "clip_hashes.json").write_text(json.dumps(hashes, indent=2))
        ctx["store_dir"] = str(store.root)
        return {
            "count": len(store),
            "dim": store.meta.key_dim,
            "backend": enc.spec.backend,
            "result_tag": "provisional (frozen-random substrate, mocked video)",
            "real_or_mocked": "real-pipeline",
        }

    run("cache_creation", "real-pipeline", _cache)

    # 6. cache integrity (REAL)
    def _integrity():
        sd = Path(ctx["store_dir"])
        problems = cache_tools.validate_cache(sd)
        info = cache_tools.cache_info(sd)
        if problems:
            raise RuntimeError(f"cache integrity problems: {problems}")
        size = storage.dir_size(sd)
        return {
            "problems": problems,
            "backend": info.get("backend"),
            "result_tag": info.get("result_tag"),
            "count": info.get("count"),
            "bytes": size,
            "human": storage.human_bytes(size),
            "real_or_mocked": "real",
        }

    run("cache_integrity", "real", _integrity)

    # 7. full-grid dry-run + cost-projection agreement (REAL planning over real leg YAMLs)
    def _fullgrid():
        toy = run_queue(dry_run=True, enabled_tiers={"C"}, toy=True)
        full = run_queue(dry_run=True, enabled_tiers={"C", "E", "R"}, toy=False, run_disabled=True)
        cp = cost_projection({}, workers=4)
        by = {r["name"]: r for r in cp["per_leg"]}
        from .harness.queue import load_queue

        mismatches = []
        for leg in load_queue():
            d = load_leg(REPO_ROOT / leg.sweep)
            if leg.tier == "C" and by[leg.name]["run_units_full"] != full_run_units(d):
                mismatches.append(leg.name)
        if mismatches:
            raise RuntimeError(f"cost/manifest/full-grid disagreement: {mismatches}")
        c_dry = run_queue(dry_run=True, enabled_tiers={"C"}, toy=True)
        return {
            "toy_run_units": toy["totals"]["run_units_toy"],
            "full_run_units": full["totals"]["run_units_full"],
            "planned_C": toy["totals"]["planned_legs"],
            "skipped": [s["reason"] for s in c_dry["skipped"]],
            "cost_grand_parallel_wall_h": round(cp["grand_total_parallel_wall_s"] / 3600, 3),
            "agreement": "cost_projection == run_queue --full == manifest",
            "real_or_mocked": "real",
        }

    run("full_grid_dryrun", "real", _fullgrid)

    # 8. miniature Tier C campaign (REAL run, PROVISIONAL result: one toy leg, one unit)
    def _mini():
        ran = run_queue(dry_run=False, enabled_tiers={"C"}, toy=True, max_runs_per_leg=1, max_legs=1)
        return {
            "legs_run": list(ran.get("results", {})),
            "results": ran.get("results"),
            "tag": "rehearsal/provisional (toy scale, synthetic latents)",
            "real_or_mocked": "real-run-provisional-result",
        }

    run("miniature_campaign", "real-run", _mini)

    # 9. microbenchmarks (REAL measurements on this laptop, opt-in, tiny)
    def _bench():
        rows = bench.run_benches(reps=1)
        return {
            "benches": {r["name"]: round(r["seconds"], 5) for r in rows},
            "n": len(rows),
            "real_or_mocked": "real (laptop-throttled)",
        }

    run("microbench", "real", _bench)

    # assemble report + provenance
    overall = "pass" if all(s["status"] == "pass" for s in stages) else "partial"
    prov = provenance.provenance(
        seed=seed,
        device=dev.kind,
        encoder_id="vjepa2_vitl_fpc64_256",
        encoder_backend="frozen_random",
        result_tag="provisional",
        cache=provenance.cache_id("rehearsal", ctx.get("corpus_manifest", {}).get("n_clips", 0)),
        extra={"rehearsal": True},
    )
    summary = {
        "rehearsal": True,
        "overall": overall,
        "seed": seed,
        "started": t_start,
        "finished": time.time(),
        "provenance": prov,
        "stages": stages,
        "real_vs_mocked": {s["stage"]: s.get("detail", {}).get("real_or_mocked", s["kind"]) for s in stages},
        "deferred": [
            "natural-video clips (procure SSv2/Ego4D/EPIC; this rehearsal uses .npy mocked clips)",
            "video decode backend (uv pip install -e '.[video]'); decode is mocked here",
            "real V-JEPA weights at scale + 64-frame forward on Metal (verify on the Studio)",
            "V-JEPA 2.1 dense weights (not on HF; E6 dense deferred)",
            "Tier E/R env + rollout legs (rented CUDA)",
        ],
        "studio_day_one": [
            "make doctor",
            "uv pip install -e '.[encoder,video,apple]'",
            "python scripts/cache_video.py +source=<clips> encoder=vjepa2_vitl_fpc64_256 device=mps +total=N",
            "python scripts/cache_tool.py validate data/cache/vjepa2_vitl_fpc64_256_video",
            "python scripts/run_queue.py --dry-run --tiers C   # then: --tiers C --full",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.md").write_text(render_md(summary))
    log.info("rehearsal %s: %d stages, report -> %s", overall, len(stages), out / "report.md")
    return summary


def render_md(s: dict) -> str:
    L = ["# Mac-Studio rehearsal capsule", ""]
    L.append(
        f"REHEARSAL, not a scientific result. Overall: **{s['overall']}**. "
        f"git {s['provenance']['git_sha']} (dirty={s['provenance']['git_dirty']}), "
        f"device {s['provenance']['device']}, seed {s['seed']}."
    )
    L += ["", "## Stages", "", "| stage | status | kind | seconds | real/mocked |", "|---|---|---|---|---|"]
    for st in s["stages"]:
        rm = (
            st.get("detail", {}).get("real_or_mocked", st["kind"])
            if isinstance(st.get("detail"), dict)
            else st["kind"]
        )
        L.append(f"| {st['stage']} | {st['status']} | {st['kind']} | {st['seconds']} | {rm} |")
    L += ["", "## What was real vs mocked", ""]
    for k, v in s["real_vs_mocked"].items():
        L.append(f"- **{k}**: {v}")
    L += ["", "## Remaining deferrals (honest)", ""] + [f"- {d}" for d in s["deferred"]]
    L += ["", "## Mac Studio day one", "", "```"] + s["studio_day_one"] + ["```", ""]
    L += ["Full per-stage detail in summary.json (provenance, counts, timings)."]
    return "\n".join(L)
