"""E10 scaffold: the minimal open-ended JEPA capstone runs its ablation ladder at toy scale,
reports per-arm diversity/reuse/collapse, and answers its explicit null (single_agent_plateaus)
with a saved plot. Imported directly (env-later scaffold, not via the registry)."""

from devsys import config, devices
from devsys.experiments.e10_openended import ARMS, E10


def test_e10_ladder_runs_and_answers_null(tmp_path):
    cfg = config.compose(
        [
            "experiment=e10_openended",
            "device=cpu",
            "experiment.n_actions=60",  # toy override: keep it tiny (seconds)
            "experiment.dim=24",
            "experiment.horizon=3",
        ]
    )
    out = E10().run(cfg, devices.resolve("cpu"), tmp_path / "e10_openended")

    # the ablation ladder ran every arm and reported each metric
    assert set(out["arms"]) == set(ARMS)
    for arm in ARMS:
        row = out["arms"][arm]
        for k in ("distinct_skills", "skill_reuse", "archive_diversity", "collapse"):
            assert k in row
        assert row["distinct_skills"] >= 1
        assert 0.0 <= row["skill_reuse"] <= 1.0
        assert isinstance(row["collapse"], bool)
        assert row["steps"] == 60

    # per-arm diversity is reported and the ladder has one value per rung
    assert set(out["per_arm_diversity"]) == set(ARMS)
    assert len(out["diversity_ladder"]) == len(ARMS)

    # the null check key exists, is a real boolean, and the comparison actually ran
    assert "single_agent_plateaus" in out
    assert isinstance(out["single_agent_plateaus"], bool)
    assert "last_rung_gain" in out and "best_rung_gain" in out
    assert out["scaffold_only"] is True
    assert out["tier"] == "env-later"

    # plot saved
    assert (tmp_path / "e10_openended" / "e10_ladder.png").exists()
