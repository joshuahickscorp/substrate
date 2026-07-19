
from __future__ import annotations

import json
import time
from pathlib import Path

from .. import bench, provenance, studio_doctor
from ..config import REPO_ROOT
from ..logging_utils import get_logger
from . import controls, datacards, downloader, planner, registry
from .profiles import Profile, get_profile

log = get_logger("studio_pipeline")
RUNS_ROOT = REPO_ROOT / "runs" / "studio_pipeline"


def _run_dir(label: str | None, out_root: Path | None = None) -> Path:
    root = Path(out_root or RUNS_ROOT)
    name = label or time.strftime("%Y%m%d_%H%M%S", time.localtime())
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _set_latest(run_dir: Path) -> None:
    latest = run_dir.parent / "latest"
    latest_txt = run_dir.parent / "latest.txt"
    try:
        if latest.is_symlink() or latest.exists():
            if latest.is_symlink() or latest.is_file():
                latest.unlink()
            else:
                import shutil

                shutil.rmtree(latest)
        latest.symlink_to(run_dir.name, target_is_directory=True)
        latest_txt.unlink(missing_ok=True)  # symlink is authoritative: drop any stale text pointer
    except OSError:  # symlinks unavailable: leave a text pointer the loader also understands
        try:
            if latest.is_symlink():  # drop a stale symlink so it cannot win in _resolve_latest
                latest.unlink()
        except OSError:
            pass
        latest_txt.write_text(run_dir.name)


def _resolve_latest(p: Path) -> Path:
    p = Path(p)
    if p.exists():
        return p
    parts = list(p.parts)
    if "latest" in parts:
        i = parts.index("latest")
        ptr = Path(*parts[:i]) / "latest.txt"
        if ptr.exists():
            real = Path(*parts[:i]) / ptr.read_text().strip() / Path(*parts[i + 1 :])
            return real
    return p


def _write(run_dir: Path, name: str, obj: dict) -> Path:
    p = run_dir / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p


def _write_text(run_dir: Path, name: str, text: str) -> Path:
    p = run_dir / name
    p.write_text(text)
    return p


def _prov(profile: Profile, **extra) -> dict:
    return provenance.provenance(
        seed=extra.pop("seed", 0),
        device=extra.pop("device", "cpu"),
        result_tag=extra.pop("result_tag", "provisional"),
        extra={"profile": profile.name, **extra},
    )


def cmd_plan(
    profile_name: str = "studio-1tb",
    budget_gb: float | None = None,
    accept_license: bool = False,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    label: str | None = None,
    out_root: Path | None = None,
) -> dict:
    profile = get_profile(profile_name)
    reg_problems = registry.validate_registry()
    if reg_problems:
        log.warning("registry has %d problems (plan still written): %s", len(reg_problems), reg_problems[:3])
    p = planner.plan(
        profile, budget_gb=budget_gb, accept_license=accept_license, include=include, exclude=exclude
    )
    p["registry_problems"] = reg_problems
    p["provenance"] = _prov(profile)
    run_dir = _run_dir(label, out_root)
    _write(run_dir, "plan.json", p)
    _write_text(run_dir, "plan.md", planner.render_md(p))
    cards = datacards.write_data_cards(run_dir, plan=p)
    _set_latest(run_dir)
    log.info(
        "plan(%s): %d selected, %d GB raw + %d GB cache, %d blockers -> %s",
        profile.name,
        p["totals"]["selected_sources"],
        round(p["totals"]["raw_gb"]),
        round(p["totals"]["cache_gb"]),
        len(p["license_blockers"]),
        run_dir,
    )
    return {"run_dir": str(run_dir), "plan": p, "datacards": cards}


def _control_gen_fn(seed: int, profile: Profile):

    def gen(sel: dict, dest: Path) -> dict:
        clips = profile.clamp_clips(sel.get("subset_clips") or 32)
        per_class = max(1, clips // (2 * len(controls.DEFAULT_FAMILIES)))
        out = controls.generate_controls(
            dest,
            clips_per_class=per_class,
            frames=8,
            h=32,
            w=32,
            seed=seed,
            budget_gb=profile.fixture_budget_gb,
        )
        n_clips = sum(f["n_clips"] for f in out["families"])
        return {"bytes": out["total_bytes"], "n_clips": n_clips, "families": out["families"]}

    return gen


def cmd_acquire(
    plan_path: str | Path,
    execute: bool = False,
    budget_gb: float | None = None,
    accept_license: bool = False,
    profile_name: str | None = None,
    seed: int = 0,
) -> dict:
    plan_path = _resolve_latest(Path(plan_path))
    plan = json.loads(plan_path.read_text())
    profile = get_profile(profile_name or plan.get("profile", {}).get("name", "studio-1tb"))
    if budget_gb is None:
        budget_gb = plan.get("totals", {}).get("budget_gb")
    run_dir = plan_path.parent
    manifest = downloader.acquire(
        plan,
        profile,
        out_dir=run_dir,
        execute=execute,
        budget_gb=budget_gb,
        accept_license=accept_license,
        gen_fn=_control_gen_fn(seed, profile),
        manifest_path=run_dir / "acquire_manifest.json",
    )
    datacards.write_data_cards(run_dir, plan=plan, manifest=manifest)
    return {"run_dir": str(run_dir), "manifest": manifest}


def cmd_validate(plan_path: str | Path) -> dict:
    from ..substrate.video import validate_source

    plan_path = _resolve_latest(Path(plan_path))
    plan = json.loads(plan_path.read_text())
    run_dir = plan_path.parent
    man_path = run_dir / "acquire_manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"sources": []}
    by_slug = {s["slug"]: s for s in manifest.get("sources", [])}

    results = []
    for sel in plan.get("selected", []):
        slug = sel["slug"]
        acq = by_slug.get(slug, {})
        dest = acq.get("dest")
        rec = {
            "slug": slug,
            "method": sel.get("download_method"),
            "status": acq.get("status", "not-acquired"),
        }
        if sel.get("download_method") == "generate" and dest:
            family_reports = []
            for fam_dir in sorted(Path(dest).iterdir()) if Path(dest).exists() else []:
                if fam_dir.is_dir():
                    try:
                        v = validate_source(fam_dir)
                        family_reports.append({"family": fam_dir.name, "ok": True, **_v_summary(v)})
                    except Exception as e:
                        family_reports.append({"family": fam_dir.name, "ok": False, "error": str(e)})
            rec["validation"] = {"families": family_reports, "ok": all(f["ok"] for f in family_reports)}
        elif dest and Path(dest).exists():
            try:
                v = validate_source(dest)
                rec["validation"] = {"ok": True, **_v_summary(v)}
            except Exception as e:
                rec["validation"] = {"ok": False, "error": str(e)}
        else:
            rec["validation"] = {"ok": None, "note": "not acquired or metadata-only; nothing to validate yet"}
        results.append(rec)

    out = {
        "plan": str(plan_path),
        "results": results,
        "provenance": provenance.provenance(seed=0, device="cpu"),
    }
    _write(run_dir, "validate.json", out)
    _write_text(run_dir, "validate.md", _render_validate_md(out))
    return out


def _v_summary(v: dict) -> dict:
    return {"classes": v["classes"], "n_clips": v["n_clips"], "label_map": v["label_map"]}


def _render_validate_md(out: dict) -> str:
    L = ["# Source validation", "", "| slug | status | result |", "| --- | --- | --- |"]
    for r in out["results"]:
        val = r.get("validation", {})
        if "families" in val:
            res = f"{sum(1 for f in val['families'] if f['ok'])}/{len(val['families'])} families valid"
        elif val.get("ok") is True:
            res = f"{len(val.get('classes', []))} classes, {val.get('n_clips')} clips"
        elif val.get("ok") is False:
            res = f"FAIL: {val.get('error')}"
        else:
            res = val.get("note", "pending")
        L.append(f"| {r['slug']} | {r['status']} | {res} |")
    return "\n".join(L) + "\n"


def cmd_cache(
    plan_path: str | Path,
    execute: bool = False,
    encoder: str = "vjepa2_vitl_fpc64_256",
    profile_name: str | None = None,
    seed: int = 0,
) -> dict:
    from ..config import compose
    from ..substrate import storage

    plan_path = _resolve_latest(Path(plan_path))
    plan = json.loads(plan_path.read_text())
    profile = get_profile(profile_name or plan.get("profile", {}).get("name", "studio-1tb"))
    run_dir = plan_path.parent
    man_path = run_dir / "acquire_manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"sources": []}
    by_slug = {s["slug"]: s for s in manifest.get("sources", [])}

    free_ok, free_gb = profile.free_disk_ok()
    if execute and not free_ok:
        out = {
            "plan": str(plan_path),
            "encoder": encoder,
            "results": [],
            "error": f"free-disk kill switch: {free_gb:.0f} GB free below "
            f"{profile.min_free_disk_gb:.0f} GB floor",
            "provenance": _prov(profile),
        }
        _write(run_dir, "cache.json", out)
        log.warning("cmd_cache refused: %s", out["error"])
        return out

    enc_cfg = compose([f"encoder={encoder}", "device=cpu"]).encoder
    enc_dict = {"name": encoder, "embed_dim": int(enc_cfg.embed_dim)}
    results = []
    for sel in plan.get("selected", []):
        slug = sel["slug"]
        if sel.get("kind") == "metadata-only" or sel.get("status") == "metadata-only":
            results.append({"slug": slug, "action": "skip", "reason": "metadata-only (no latents to cache)"})
            continue
        n = profile.clamp_clips(sel.get("subset_clips") or 64)
        est = storage.estimate_for_encoder(enc_dict, n_clips=n)
        rec = {"slug": slug, "n_clips_planned": n, "estimate": est}
        acq = by_slug.get(slug, {})
        if execute and acq.get("status") == "complete" and acq.get("dest") and Path(acq["dest"]).exists():
            try:
                rec["cache"] = _build_tiny_cache(
                    Path(acq["dest"]), run_dir / "cache" / slug, encoder, profile, seed
                )
            except Exception as e:
                rec["cache"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        else:
            rec["cache"] = {
                "ok": None,
                "note": "dry-run estimate only; --execute on a landed corpus to build",
            }
        results.append(rec)

    out = {"plan": str(plan_path), "encoder": encoder, "results": results, "provenance": _prov(profile)}
    _write(run_dir, "cache.json", out)
    return out


def _first_class_folder(root: Path):
    from ..substrate.video import validate_source

    try:
        validate_source(root)
        return root
    except Exception:  # not a class-folder (empty, wrong layout, anything); try one level down
        pass
    for child in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        try:
            validate_source(child)
            return child
        except Exception:
            continue
    raise ValueError(f"no class-folder source found under {root}")


def _build_tiny_cache(source: Path, cache_root: Path, encoder: str, profile: Profile, seed: int) -> dict:
    import numpy as np

    from ..config import compose
    from ..devices import resolve
    from ..substrate import cache_latents, cache_tools, iter_video_clips, load_encoder, storage
    from ..substrate.video import validate_source, write_label_map

    src = _first_class_folder(source)
    v = validate_source(src)
    cap = profile.clamp_clips(v["n_clips"])
    dev = resolve("cpu")
    cfg = compose([f"encoder={encoder}", "device=cpu"])
    enc = load_encoder(cfg.encoder).to(dev.device)
    hashes: list[str] = []
    clips = iter_video_clips(
        src, frames_per_clip=8, res=16, batch=2, limit=cap, hashes_out=hashes, stratified=True
    )
    store = cache_latents(
        enc, clips, cache_root, "studio_cache", total=cap, device=dev, result_tag="provisional", seed=seed
    )
    # persist ONLY the classes actually cached (honest coverage), not the full source label_map
    lab_t = store.labels() if len(store) else None
    labels = np.asarray(lab_t[: len(store)]) if lab_t is not None else np.asarray([], dtype="int64")
    present = sorted({int(x) for x in labels.tolist()})
    inv = {i: c for c, i in v["label_map"].items()}
    present_map = {inv[i]: i for i in present if i in inv}
    write_label_map(store.root, present_map)
    dropped = [inv[i] for i in range(len(v["label_map"])) if i not in present and i in inv]
    if dropped:
        log.warning(
            "cache %s dropped %d classes under the clip cap: %s", store.root.name, len(dropped), dropped
        )
    problems = cache_tools.validate_cache(store.root)
    return {
        "ok": not problems,
        "store_dir": str(store.root),
        "count": len(store),
        "backend": enc.spec.backend,
        "problems": problems,
        "bytes": storage.dir_size(store.root),
        "n_classes_declared": len(v["label_map"]),
        "n_classes_present": len(present),
        "classes_dropped": dropped,
    }


def cmd_run(
    gated: bool = True,
    tiers: set[str] | None = None,
    full: bool = False,
    profile_name: str = "m3pro-local-max",
    label: str | None = None,
    out_root: Path | None = None,
    max_legs: int | None = None,
    max_runs_per_leg: int = 1,
) -> dict:
    from ..harness.queue import run_queue

    profile = get_profile(profile_name)
    tiers = tiers or {"C"}
    run_dir = _run_dir(label, out_root) if label or out_root else _latest_or_new()
    gates = _conveyor_gates(profile, tiers, full)
    passed = all(g["ok"] for g in gates)
    out: dict = {"profile": profile.name, "tiers": sorted(tiers), "full": full, "gates": gates, "ran": False}
    if not passed:
        out["reason"] = "; ".join(g["detail"] for g in gates if not g["ok"])
        _write(run_dir, "run.json", out)
        log.warning("conveyor STOPPED at a gate: %s", out["reason"])
        return out
    enabled = {t for t in tiers if profile.tier_allowed(t)}
    summary = run_queue(
        dry_run=False,
        enabled_tiers=enabled,
        toy=not full,
        max_runs_per_leg=max_runs_per_leg,
        max_legs=max_legs,
    )
    out["ran"] = True
    out["queue"] = {k: summary.get(k) for k in ("planned", "totals", "results")}
    out["provenance"] = _prov(profile)
    _write(run_dir, "run.json", out)
    return out


def _conveyor_gates(profile: Profile, tiers: set[str], full: bool) -> list[dict]:
    from ..harness.queue import run_queue

    gates: list[dict] = []
    reg = registry.validate_registry()
    gates.append({"name": "registry_valid", "ok": not reg, "detail": f"{len(reg)} registry problems"})
    free_ok, free_gb = profile.free_disk_ok()
    gates.append(
        {
            "name": "free_disk",
            "ok": free_ok,
            "detail": f"{free_gb:.0f} GB free (min {profile.min_free_disk_gb:.0f})",
        }
    )
    bad_tiers = [t for t in tiers if not profile.tier_allowed(t)]
    gates.append(
        {
            "name": "tier_allowed",
            "ok": not bad_tiers,
            "detail": f"tiers {sorted(tiers)} vs allowed {sorted(profile.allowed_tiers)}"
            + (f"; blocked {bad_tiers}" if bad_tiers else ""),
        }
    )
    try:
        dry = run_queue(
            dry_run=True, enabled_tiers={t for t in tiers if profile.tier_allowed(t)} or {"C"}, toy=not full
        )
        units = dry["totals"]["run_units_full" if full else "run_units_toy"]
    except Exception as e:
        units = -1
        gates.append({"name": "queue_resolves", "ok": False, "detail": f"queue dry-run failed: {e}"})
    gates.append(
        {
            "name": "run_count_cap",
            "ok": 0 <= units <= profile.max_run_count,
            "detail": f"{units} run-units vs cap {profile.max_run_count} (full={full})",
        }
    )
    return gates


def _latest_or_new() -> Path:
    latest = RUNS_ROOT / "latest"
    real = _resolve_latest(latest / "plan.json")
    if real.exists():
        return real.parent
    return _run_dir(None)


def cmd_optimize(cache: str | None = None, profile_name: str = "m3pro-local-max", reps: int = 1) -> dict:
    profile = get_profile(profile_name)
    run_dir = _latest_or_new()
    rows = bench.run_benches(reps=reps)
    out = {
        "cache": cache,
        "profile": profile.name,
        "benches": {r["name"]: round(r["seconds"], 6) for r in rows},
        "note": "optimization is not science; speeds are laptop-throttled and provisional",
        "next": [
            "python scripts/studio_pipeline.py optimize --cache <cache_id>  # re-measure after a change",
            "uv pip install -e '.[apple]'  # MLX throughput experiment (optional, see APPLE_SILICON.md)",
        ],
        "provenance": _prov(profile, result_tag="provisional"),
    }
    _write(run_dir, "optimize.json", out)
    _write_text(run_dir, "optimize.md", _render_optimize_md(out))
    return out


def _render_optimize_md(out: dict) -> str:
    L = [
        "# Optimization lane (provisional, not science)",
        "",
        out["note"],
        "",
        "| bench | seconds |",
        "| --- | --- |",
    ]
    for k, v in out["benches"].items():
        L.append(f"| {k} | {v} |")
    L += ["", "## Next", "", "```"] + out["next"] + ["```", ""]
    return "\n".join(L)


def cmd_report(run_dir: Path | None = None) -> dict:
    run_dir = Path(run_dir) if run_dir is not None else _latest_or_new()
    artifacts = {}
    for name in ("plan", "acquire_manifest", "validate", "cache", "run", "optimize"):
        p = run_dir / f"{name}.json"
        if p.exists():
            try:
                artifacts[name] = json.loads(p.read_text())
            except Exception:
                artifacts[name] = {"error": "unreadable"}
    summary = {
        "run_dir": str(run_dir),
        "stages_present": sorted(artifacts),
        "plan_totals": artifacts.get("plan", {}).get("totals"),
        "license_blockers": artifacts.get("plan", {}).get("license_blockers"),
        "acquire_totals": artifacts.get("acquire_manifest", {}).get("totals"),
        "run_gates": artifacts.get("run", {}).get("gates"),
        "provenance": provenance.provenance(seed=0, device="cpu"),
    }
    _write(run_dir, "summary.json", summary)
    _write_text(run_dir, "report.md", _render_report_md(summary, artifacts))
    return summary


def _render_report_md(summary: dict, artifacts: dict) -> str:
    L = [
        "# Studio pipeline report",
        "",
        f"Run: `{summary['run_dir']}`",
        "",
        f"Stages present: {summary['stages_present']}",
        "",
    ]
    pt = summary.get("plan_totals")
    if pt:
        L += [
            "## Plan",
            "",
            f"- selected {pt['selected_sources']} sources, {pt['raw_gb']:.1f} GB raw + "
            f"{pt['cache_gb']:.1f} GB cache ({pt['headroom_gb']:.1f} GB headroom under "
            f"{pt['budget_gb']:.0f} GB budget)",
            f"- breadth: modalities {pt['modalities_covered']}, domains {pt['domains_covered']}",
            "",
        ]
    if summary.get("license_blockers"):
        L += (
            ["## License blockers", ""]
            + [f"- {b['slug']} ({b['status']}): {b['reason']}" for b in summary["license_blockers"]]
            + [""]
        )
    at = summary.get("acquire_totals")
    if at:
        L += [
            "## Acquire",
            "",
            f"- {at['n_sources']} sources, {at['human_spent']} of {at['budget_human']} budget, "
            f"by status {at['by_status']}",
            "",
        ]
    if summary.get("run_gates"):
        L += ["## Conveyor gates", "", "| gate | ok | detail |", "| --- | --- | --- |"]
        for g in summary["run_gates"]:
            L.append(f"| {g['name']} | {'ok' if g['ok'] else 'FAIL'} | {g['detail']} |")
        L.append("")
    L += ["## Next", "", "```", "python scripts/studio_pipeline.py report", "```", ""]
    return "\n".join(L)


def cmd_local_max(
    download_gb: float = 10.0,
    time_min: int = 90,
    cache_clips: int = 64,
    seed: int = 0,
    label: str | None = None,
    out_root: Path | None = None,
) -> dict:
    profile = get_profile("m3pro-local-max")
    eff_download = profile.effective_budget_gb(download_gb)
    eff_clips = profile.clamp_clips(cache_clips)
    eff_time = min(int(time_min), profile.max_wall_min)
    run_dir = _run_dir(label or ("local_max_" + time.strftime("%Y%m%d_%H%M%S", time.localtime())), out_root)
    t_start = time.time()
    stages: list[dict] = []

    def stage(name: str, kind: str, fn):
        if (time.time() - t_start) / 60.0 > eff_time:  # wall-time kill switch
            stages.append(
                {
                    "stage": name,
                    "kind": kind,
                    "status": "skipped",
                    "detail": {"reason": "time budget reached"},
                }
            )
            return None
        t0 = time.perf_counter()
        rec: dict = {"stage": name, "kind": kind}
        try:
            rec["detail"] = fn()
            rec["status"] = "pass"
        except Exception as e:
            rec["status"] = "fail"
            rec["detail"] = {"error": f"{type(e).__name__}: {e}"}
            log.warning("local-max stage %s failed: %s", name, e)
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        stages.append(rec)
        return rec.get("detail")

    ctx: dict = {}

    free_ok, free_gb = profile.free_disk_ok()
    stages.append(
        {
            "stage": "free_disk_killswitch",
            "kind": "real",
            "status": "pass" if free_ok else "fail",
            "detail": {"free_gb": round(free_gb, 1), "min_free_gb": profile.min_free_disk_gb, "ok": free_ok},
        }
    )

    stage("doctor", "real", lambda: {k: studio_doctor.doctor()[k] for k in ("all_ok", "summary")})
    stage("registry_validate", "real", lambda: {"problems": registry.validate_registry()})

    def _plan():
        p = planner.plan(profile, budget_gb=eff_download)
        ctx["plan"] = p
        _write(run_dir, "plan.json", {**p, "provenance": _prov(profile)})
        _write_text(run_dir, "plan.md", planner.render_md(p))
        return {
            "selected": p["totals"]["selected_sources"],
            "blockers": len(p["license_blockers"]),
            "real_or_mocked": "real",
        }

    stage("plan", "real", _plan)

    def _acquire_dry():
        m = downloader.acquire(ctx["plan"], profile, out_dir=run_dir, execute=False, budget_gb=eff_download)
        ctx["acquire_dry"] = m
        return {"mode": m["mode"], "by_status": m["totals"]["by_status"], "real_or_mocked": "real (dry-run)"}

    stage("acquire_dryrun", "real", _acquire_dry)

    corpus = run_dir / "controls"

    def _generate():
        out = controls.generate_controls(
            corpus,
            clips_per_class=max(1, eff_clips // (2 * len(controls.DEFAULT_FAMILIES))),
            frames=8,
            h=32,
            w=32,
            seed=seed,
            budget_gb=profile.fixture_budget_gb,
        )
        ctx["controls"] = out
        return {
            "families": [f["name"] for f in out["families"]],
            "n_clips": sum(f["n_clips"] for f in out["families"]),
            "bytes": out["total_bytes"],
            "truncated": out["truncated"],
            "real_or_mocked": "real (generated structured-synthetic; mocked decode via .npy)",
        }

    stage("generate_controls", "real-generated", _generate)

    def _validate_src():
        from ..substrate.video import validate_source

        fam0 = ctx["controls"]["families"][0]["root"]
        v = validate_source(fam0)
        ctx["src_validated"] = fam0
        ctx["label_map"] = v["label_map"]
        return {
            "family": Path(fam0).name,
            "classes": v["classes"],
            "n_clips": v["n_clips"],
            "real_or_mocked": "real",
        }

    stage("validate_source", "real", _validate_src)

    def _cache():
        out = _build_tiny_cache(
            Path(ctx["src_validated"]).parent,
            run_dir / "cache" / "local",
            "vjepa2_vitl_fpc64_256",
            profile,
            seed,
        )
        ctx["cache"] = out
        return {**out, "real_or_mocked": "real-pipeline (frozen-random substrate -> provisional tag)"}

    stage("build_cache", "real-pipeline", _cache)

    def _audit():
        from ..harness.queue import load_queue, run_queue
        from ..harness.sweep import full_run_units, load_leg
        from ..studies.cost_projection import cost_projection

        toy = run_queue(dry_run=True, enabled_tiers={"C"}, toy=True)
        full = run_queue(dry_run=True, enabled_tiers={"C", "E", "R"}, toy=False, run_disabled=True)
        cp = cost_projection({}, workers=4)
        by = {r["name"]: r for r in cp["per_leg"]}
        mismatches = [
            leg.name
            for leg in load_queue()
            if leg.tier == "C"
            and by[leg.name]["run_units_full"] != full_run_units(load_leg(REPO_ROOT / leg.sweep))
        ]
        if mismatches:
            raise RuntimeError(f"queue/cost/full-grid disagreement: {mismatches}")
        return {
            "toy_run_units": toy["totals"]["run_units_toy"],
            "full_run_units": full["totals"]["run_units_full"],
            "agreement": "run_queue --full == cost_projection == manifest",
            "real_or_mocked": "real",
        }

    stage("queue_cost_audit", "real", _audit)
    stage(
        "microbench",
        "real",
        lambda: {
            "benches": {r["name"]: round(r["seconds"], 5) for r in bench.run_benches(reps=1)},
            "real_or_mocked": "real (laptop-throttled)",
        },
    )

    def _gated_run():
        out = cmd_run(
            gated=True,
            tiers={"C"},
            full=False,
            profile_name="m3pro-local-max",
            label=run_dir.name,
            out_root=run_dir.parent,
            max_legs=1,  # minimal gated run: one toy leg, one unit (the rehearsal contract)
            max_runs_per_leg=1,
        )
        return {
            "ran": out.get("ran"),
            "gates_ok": all(g["ok"] for g in out["gates"]),
            "real_or_mocked": "real-run-provisional-result",
        }

    stage("gated_run", "real-run", _gated_run)

    def _cards():
        c = datacards.write_data_cards(run_dir, plan=ctx.get("plan"), manifest=ctx.get("acquire_dry"))
        return {**c, "real_or_mocked": "real"}

    stage("datacards_ledger", "real", _cards)

    overall = "pass" if all(s["status"] in ("pass", "skipped") for s in stages) else "partial"
    summary = {
        "local_max": True,
        "overall": overall,
        "profile": profile.as_dict(),
        "requested": {"download_gb": download_gb, "time_min": time_min, "cache_clips": cache_clips},
        "effective": {"download_gb": eff_download, "time_min": eff_time, "cache_clips": eff_clips},
        "elapsed_min": round((time.time() - t_start) / 60.0, 2),
        "seed": seed,
        "stages": stages,
        "real_vs_mocked": {
            s["stage"]: (s.get("detail") or {}).get("real_or_mocked", s["kind"])
            if isinstance(s.get("detail"), dict)
            else s["kind"]
            for s in stages
        },
        "provenance": _prov(profile, seed=seed),
        "deferred": [
            "real natural-video downloads (SSv2/EPIC/Ego4D subsets): planned + dry-run only here",
            "real V-JEPA weights at scale (single canonical weight smoke is opt-in, not run by default)",
            "full Tier C natural-video campaign + Tier E/R: Studio / rented CUDA only",
        ],
        "studio_day_one": _studio_day_one(),
    }
    _write(run_dir, "summary.json", summary)
    _write_text(run_dir, "report.md", _render_local_max_md(summary))
    _set_latest(run_dir)
    log.info(
        "local-max %s: %d stages, %.1f min, report -> %s",
        overall,
        len(stages),
        summary["elapsed_min"],
        run_dir / "report.md",
    )
    return summary


def _studio_day_one() -> list[str]:
    return [
        "make doctor",
        "uv pip install -e '.[encoder,video,apple]'",
        "python scripts/studio_pipeline.py plan --profile studio-1tb --budget-gb 900",
        "python scripts/studio_pipeline.py acquire --plan runs/studio_pipeline/latest/plan.json "
        "--execute --budget-gb 900 --accept-license",
        "python scripts/studio_pipeline.py validate --plan runs/studio_pipeline/latest/plan.json",
        "python scripts/studio_pipeline.py cache    --plan runs/studio_pipeline/latest/plan.json "
        "--execute --profile studio-1tb",
        "python scripts/studio_pipeline.py run --gated --tiers C --full --profile studio-1tb",
        "python scripts/studio_pipeline.py optimize --cache <cache_id> --profile studio-1tb",
        "python scripts/studio_pipeline.py report",
    ]


def _render_local_max_md(s: dict) -> str:
    L = ["# Current-device local-max rehearsal", ""]
    L.append(
        f"Profile **{s['profile']['name']}**. Overall: **{s['overall']}**. "
        f"Elapsed {s['elapsed_min']:.1f} min "
        f"(budget {s['effective']['time_min']} min). git {s['provenance']['git_sha']} "
        f"(dirty={s['provenance']['git_dirty']}), seed {s['seed']}."
    )
    L += [
        "",
        f"Kill switches enforced: download {s['effective']['download_gb']:g} GB "
        f"(req {s['requested']['download_gb']:g}), "
        f"clips {s['effective']['cache_clips']} (req {s['requested']['cache_clips']}), "
        f"time {s['effective']['time_min']} min, min free disk {s['profile']['min_free_disk_gb']:g} GB.",
        "",
        "## Stages",
        "",
        "| stage | status | kind | seconds | real/mocked |",
        "|---|---|---|---|---|",
    ]
    for st in s["stages"]:
        rm = (
            (st.get("detail") or {}).get("real_or_mocked", st["kind"])
            if isinstance(st.get("detail"), dict)
            else st["kind"]
        )
        L.append(f"| {st['stage']} | {st['status']} | {st['kind']} | {st.get('seconds', 0)} | {rm} |")
    L += ["", "## Deferred (honest)", ""] + [f"- {d}" for d in s["deferred"]]
    L += ["", "## Mac Studio day one", "", "```"] + s["studio_day_one"] + ["```", ""]
    return "\n".join(L)
