"""Downloader orchestrator: dry-run writes NO bytes, the budget is a hard stop, manual sources need
a license ack, unsafe archive members are refused, resume skips completed sources, and duplicate
content is detected. The safety invariants the whole acquisition lane rests on."""

import json

from devsys.studio import downloader
from devsys.studio.profiles import M3PRO_LOCAL_MAX, Profile


def _plan(*sources):
    return {"selected": list(sources)}


def _src(slug, method="generate", raw_gb=0.0, cache_gb=0.0, status="available"):
    return {"slug": slug, "download_method": method, "raw_gb": raw_gb, "cache_gb": cache_gb, "status": status}


def test_dry_run_writes_no_bytes(tmp_path):
    plan = _plan(
        _src("synthetic_controls"),
        _src("epic", method="epic-download-script", raw_gb=4.0, status="available"),
    )
    m = downloader.acquire(plan, M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=False)
    assert m["mode"] == "dry-run"
    assert m["totals"]["bytes_spent"] == 0
    # all planned, nothing written to a data dir
    assert all(s["status"] == "planned" for s in m["sources"])
    assert not (tmp_path / "data").exists()
    # manifest checkpoint exists
    assert (tmp_path / "acquire_manifest.json").exists()


def test_unsafe_archive_members_detected():
    names = ["ok/a.mp4", "../escape.sh", "/etc/passwd", "sub/../../oops", "fine/b.npy"]
    bad = downloader.unsafe_archive_members(names)
    assert "../escape.sh" in bad
    assert "/etc/passwd" in bad
    assert "sub/../../oops" in bad
    assert "ok/a.mp4" not in bad and "fine/b.npy" not in bad


def test_generate_method_executes_and_counts_bytes(tmp_path):
    def gen(sel, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "clip.bin").write_bytes(b"x" * 1000)
        return {"bytes": 1000, "n_clips": 1}

    plan = _plan(_src("synthetic_controls", method="generate"))
    m = downloader.acquire(plan, M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=True, gen_fn=gen)
    s = m["sources"][0]
    assert s["status"] == "complete"
    assert s["bytes"] == 1000
    assert m["totals"]["bytes_spent"] == 1000


def test_budget_is_a_hard_stop(tmp_path):
    # a tiny budget profile: the second source would exceed it and must be skipped, not downloaded
    prof = Profile(
        name="tiny",
        disk_total_gb=10,
        reserve_gb=0,
        download_budget_gb=0.000001,  # ~1 KB budget
        download_hard_cap_gb=0.000001,
        fixture_budget_gb=1,
        raw_smoke_gb=1,
        max_cache_clips=10,
        min_free_disk_gb=0,
        max_run_count=10,
        max_wall_min=10,
        max_source_count=10,
        max_per_source_gb=10,
    )

    def gen(sel, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "c.bin").write_bytes(b"x" * 100)
        return {"bytes": 100, "n_clips": 1}

    # both sources declare 0 planned bytes so the first passes the pre-check; make them real-sized
    plan = _plan(
        {"slug": "a", "download_method": "generate", "raw_gb": 0.0, "cache_gb": 0.0, "status": "available"},
        {"slug": "b", "download_method": "generate", "raw_gb": 1.0, "cache_gb": 0.0, "status": "available"},
    )
    m = downloader.acquire(plan, prof, out_dir=tmp_path, execute=True, gen_fn=gen, budget_gb=0.000001)
    by = {s["slug"]: s for s in m["sources"]}
    # 'b' has planned_bytes ~1 GB which exceeds the ~1 KB budget -> skipped-budget (the kill switch)
    assert by["b"]["status"] == "skipped-budget"


def test_manual_source_needs_license(tmp_path):
    plan = _plan(_src("ssv2", method="http-token", raw_gb=1.0, status="manual"))
    # without acknowledgement, dry-run records needs-license
    m = downloader.acquire(plan, M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=False, accept_license=False)
    assert m["sources"][0]["status"] == "needs-license"


def test_remote_without_fetcher_blocks_cleanly(tmp_path):
    plan = _plan(_src("epic", method="epic-download-script", raw_gb=1.0, status="available"))
    m = downloader.acquire(plan, M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=True, budget_gb=10)
    # no fetcher/credentials on this device -> clean BLOCKED, never a crash or a partial
    assert m["sources"][0]["status"] == "blocked"


def test_local_path_without_source_is_needs_source_not_error(tmp_path):
    # the local-import lane with no user-supplied path is a clean needs-source, never an 'error'
    plan = _plan(_src("local_import", method="local-path", status="available"))
    m = downloader.acquire(plan, M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=True, budget_gb=10)
    assert m["sources"][0]["status"] == "needs-source"


def test_local_path_with_source_ingests_and_hashes(tmp_path):
    # a real local class-folder dir is validated + hashed in place (nothing downloaded)
    src = tmp_path / "userdata" / "classA"
    src.mkdir(parents=True)
    (src / "a.bin").write_bytes(b"clipbytes")
    sel = {
        "slug": "local_import",
        "download_method": "local-path",
        "raw_gb": 0.0,
        "cache_gb": 0.0,
        "status": "available",
        "local_path": str(tmp_path / "userdata"),
    }
    m = downloader.acquire(_plan(sel), M3PRO_LOCAL_MAX, out_dir=tmp_path / "out", execute=True, budget_gb=10)
    s = m["sources"][0]
    assert s["status"] == "complete"
    assert s["bytes"] == len(b"clipbytes")


def test_resume_skips_completed_source(tmp_path):
    calls = {"n": 0}

    def gen(sel, dest):
        calls["n"] += 1
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "c.bin").write_bytes(b"y" * 500)
        return {"bytes": 500, "n_clips": 1}

    plan = _plan(_src("synthetic_controls", method="generate"))
    m1 = downloader.acquire(plan, M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=True, gen_fn=gen)
    assert m1["sources"][0]["status"] == "complete" and calls["n"] == 1
    # second run resumes: the generator must NOT be called again
    m2 = downloader.acquire(plan, M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=True, gen_fn=gen)
    assert calls["n"] == 1  # not recomputed
    assert m2["sources"][0].get("resumed") is True


def test_duplicate_content_detected(tmp_path):
    def gen(sel, dest):
        dest.mkdir(parents=True, exist_ok=True)
        # two identical files -> one duplicate by content hash
        (dest / "a.bin").write_bytes(b"same")
        (dest / "b.bin").write_bytes(b"same")
        return {"bytes": 8, "n_clips": 2}

    plan = _plan(_src("synthetic_controls", method="generate"))
    m = downloader.acquire(plan, M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=True, gen_fn=gen)
    assert m["sources"][0]["n_duplicates"] == 1


def test_manifest_is_valid_json_checkpoint(tmp_path):
    plan = _plan(_src("synthetic_controls"))
    downloader.acquire(plan, M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=False)
    data = json.loads((tmp_path / "acquire_manifest.json").read_text())
    assert "sources" in data and "totals" in data and data["mode"] == "dry-run"


def test_manifest_has_provenance(tmp_path):
    downloader.acquire(_plan(_src("synthetic_controls")), M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=False)
    data = json.loads((tmp_path / "acquire_manifest.json").read_text())
    assert "provenance" in data and data["provenance"]["git_sha"]
    assert data["provenance"]["result_tag"] in (
        "provisional",
        "structured-synthetic",
        "real-encoder",
        "natural-video",
    )


def test_post_fetch_overrun_is_flagged_and_stops(tmp_path):
    # a misbehaving fetcher returns MORE than planned: the bytes are on disk, so flag over-budget + stop
    def fat(sel, dest, remaining):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "big.bin").write_bytes(b"x" * 10)
        return {"bytes": 5_000_000_000, "n_clips": 1}  # 5 GB reported, far over a 1 GB budget

    plan = _plan(
        _src("a", method="http", raw_gb=0.0, status="available"),
        _src("b", method="http", raw_gb=0.0, status="available"),
    )
    m = downloader.acquire(plan, M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=True, budget_gb=1.0, fetch_fn=fat)
    by = {s["slug"]: s for s in m["sources"]}
    assert by["a"]["status"] == "over-budget"
    assert "b" not in by  # acquisition stopped after the overrun


def test_cumulative_budget_across_resume(tmp_path):
    # first run completes source 'a' (spends ~0.6 GB); a resumed run must count that toward the
    # budget so 'b' (0.6 GB) is refused, total on-disk never exceeds the 1 GB budget.
    def gen(sel, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "c.bin").write_bytes(b"z" * 100)
        return {"bytes": 600_000_000, "n_clips": 1}

    a = _src("a", method="generate", raw_gb=0.0, status="available")
    b = _src("b", method="generate", raw_gb=0.6, status="available")
    m1 = downloader.acquire(
        _plan(a), M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=True, budget_gb=1.0, gen_fn=gen
    )
    assert m1["sources"][0]["status"] == "complete"
    # resume with both: 'a' resumes (counts 0.6 GB), 'b' would push to 1.2 GB > 1 GB -> skipped
    m2 = downloader.acquire(
        _plan(a, b), M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=True, budget_gb=1.0, gen_fn=gen
    )
    by = {s["slug"]: s for s in m2["sources"]}
    assert by["a"].get("resumed") is True
    assert by["b"]["status"] == "skipped-budget"


def test_partial_removed_on_fetcher_error(tmp_path):
    # a fetcher that writes bytes then raises must not leave orphaned, untracked data on disk
    def boom(sel, dest, remaining):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "partial.bin").write_bytes(b"halfway")
        raise RuntimeError("connection dropped")

    plan = _plan(_src("x", method="http", raw_gb=0.0, status="available"))
    m = downloader.acquire(plan, M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=True, budget_gb=10, fetch_fn=boom)
    assert m["sources"][0]["status"] == "error"
    assert not (tmp_path / "data" / "x").exists()  # partial cleaned up


def test_resume_refetches_when_data_missing(tmp_path):
    calls = {"n": 0}

    def gen(sel, dest):
        calls["n"] += 1
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "c.bin").write_bytes(b"q" * 50)
        return {"bytes": 50, "n_clips": 1}

    plan = _plan(_src("synthetic_controls", method="generate"))
    downloader.acquire(plan, M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=True, gen_fn=gen)
    assert calls["n"] == 1
    # delete the landed data: a resume must NOT trust the manifest's 'complete', it re-fetches
    import shutil

    shutil.rmtree(tmp_path / "data" / "synthetic_controls")
    m2 = downloader.acquire(plan, M3PRO_LOCAL_MAX, out_dir=tmp_path, execute=True, gen_fn=gen)
    assert calls["n"] == 2  # re-fetched the phantom source
    assert m2["sources"][0]["status"] == "complete"


def test_safe_extract_raises_on_unsafe_member(tmp_path):
    import pytest

    downloader.safe_extract(["ok/a.mp4", "ok/b.npy"], tmp_path)  # safe set returns fine
    with pytest.raises(RuntimeError):
        downloader.safe_extract(["ok/a.mp4", "../escape.sh"], tmp_path)


def test_unsafe_archive_members_windows_paths():
    bad = downloader.unsafe_archive_members(["C:\\Windows\\evil.dll", "share\\..\\..\\x", "ok/fine.mp4"])
    assert "C:\\Windows\\evil.dll" in bad  # windows drive-letter absolute
    assert "share\\..\\..\\x" in bad  # backslash traversal
    assert "ok/fine.mp4" not in bad
