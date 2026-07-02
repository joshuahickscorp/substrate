"""The Mac-Studio rehearsal capsule runs the whole future Studio workflow end to end on tiny
local fixtures, tags real vs mocked honestly, and writes a report + JSON summary."""

import json

from mop import studio_rehearsal
from mop.provenance import RESULT_TAGS

STAGES = {
    "doctor",
    "video_corpus",
    "source_validation",
    "decode_preprocess",
    "cache_creation",
    "cache_integrity",
    "full_grid_dryrun",
    "miniature_campaign",
    "microbench",
}


def test_capsule_runs_all_stages(tmp_path):
    s = studio_rehearsal.rehearse(out_dir=tmp_path, frames=4, res=8)
    assert s["overall"] == "pass", [st for st in s["stages"] if st["status"] != "pass"]
    assert {st["stage"] for st in s["stages"]} == STAGES
    assert all(st["status"] == "pass" for st in s["stages"])


def test_capsule_writes_artifacts(tmp_path):
    studio_rehearsal.rehearse(out_dir=tmp_path, frames=4, res=8)
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "summary.json").exists()
    summ = json.loads((tmp_path / "summary.json").read_text())
    assert summ["rehearsal"] is True
    # provenance everywhere
    p = summ["provenance"]
    assert p["git_sha"] and "torch" in p["packages"] and p["result_tag"] in RESULT_TAGS
    assert summ["deferred"] and summ["studio_day_one"]


def test_real_vs_mocked_tags_honest(tmp_path):
    s = studio_rehearsal.rehearse(out_dir=tmp_path, frames=4, res=8)
    rvm = s["real_vs_mocked"]
    assert "mocked" in rvm["video_corpus"]
    assert "mocked" in rvm["decode_preprocess"]
    assert rvm["source_validation"] == "real"
    assert rvm["cache_integrity"] == "real"


def test_full_grid_agreement_reported(tmp_path):
    s = studio_rehearsal.rehearse(out_dir=tmp_path, frames=4, res=8)
    fg = next(st for st in s["stages"] if st["stage"] == "full_grid_dryrun")["detail"]
    assert fg["full_run_units"] > fg["toy_run_units"]  # full grid is bigger than toy
    assert "agreement" in fg


def test_cache_artifacts_have_provenance(tmp_path):
    studio_rehearsal.rehearse(out_dir=tmp_path, frames=4, res=8)
    store = tmp_path / "cache" / "rehearsal"
    assert (store / "provenance.json").exists()
    assert (store / "label_map.json").exists()
    assert (store / "clip_hashes.json").exists()
    prov = json.loads((store / "provenance.json").read_text())
    assert prov["result_tag"] in RESULT_TAGS and prov["encoder_backend"] == "frozen_random"
