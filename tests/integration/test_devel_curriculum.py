"""Curriculum engine over REAL probes on generated control corpora: it rejects the aleatoric noisy-TV
(undecodable labels), never chooses it, prefers a learnable candidate, and respects the clip budget.
This is the on-device demonstration of learning-progress data selection (rung 6 / Frontier 26/33)."""

from pathlib import Path

from devsys.devel import curriculum
from devsys.studio import controls
from devsys.studio.profiles import M3PRO_LOCAL_MAX


def _sources(tmp_path, families):
    gen = controls.generate_controls(
        tmp_path, families=families, clips_per_class=24, frames=8, h=32, w=32, seed=0
    )
    return [{"slug": f["name"], "root": f["root"], "capacity": "abstraction"} for f in gen["families"]]


def test_curriculum_rejects_noisy_tv_and_picks_learnable(tmp_path):
    sources = _sources(tmp_path, ["moving_object", "hard_motion", "aleatoric_tv"])
    m = curriculum.next_lesson(sources, profile=M3PRO_LOCAL_MAX, eval_clips=48, next_clips=64, seed=0)
    # the aleatoric noisy-TV (undecodable labels) is rejected and never chosen
    assert "aleatoric_tv" in m["rejected_noise"]
    assert m["chosen"] != "aleatoric_tv"
    assert m["chosen"] is not None  # a learnable candidate was found


def test_curriculum_respects_clip_budget(tmp_path):
    sources = _sources(tmp_path, ["moving_object", "hard_motion"])
    m = curriculum.next_lesson(sources, profile=M3PRO_LOCAL_MAX, eval_clips=10_000, next_clips=10_000, seed=0)
    # eval/next clip counts are clamped to the profile cap (kill switch)
    assert m["expected_cost_clips"] <= M3PRO_LOCAL_MAX.max_cache_clips
    for s in m["ranked"]:
        if "n_eval" in s:
            assert s["n_eval"] <= M3PRO_LOCAL_MAX.max_cache_clips


def test_curriculum_manifest_has_required_fields(tmp_path):
    sources = _sources(tmp_path, ["moving_object", "hard_motion", "aleatoric_tv"])
    m = curriculum.next_lesson(sources, profile=M3PRO_LOCAL_MAX, eval_clips=48, seed=0)
    for field in (
        "chosen",
        "reason",
        "expected_capacity",
        "expected_cost_clips",
        "stopping_condition",
        "fallback",
    ):
        assert field in m
    md = curriculum.render_md(m)
    assert "Next-lesson manifest" in md


def test_learning_progress_flags_noise_via_permutation(tmp_path):
    # direct check of the permutation-test noise detector on a pure-noise corpus
    sources = _sources(tmp_path, ["aleatoric_tv"])
    from devsys.devel.curriculum import _encode_source, learning_progress

    x, y = _encode_source(Path(sources[0]["root"]), "vjepa2_vitl_fpc64_256", 48, 0)
    lp = learning_progress(x, y, seed=0)
    assert lp["is_noise"] is True  # real labels no more decodable than shuffled
    assert lp["signal"] < curriculum.SIGNAL_MARGIN
