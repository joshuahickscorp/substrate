from mop.devel import registries as R
from mop.experiments import REGISTRY


def test_shipped_bank_validates_clean():
    assert R.validate_experiments() == []


def test_every_row_has_the_full_contract():
    for e in R.load_experiments():
        for field in (
            "null_hypothesis",
            "falsifier",
            "metrics",
            "taxonomy_slot",
            "proof",
            "status",
            "exp_tier",
        ):
            assert e.get(field) not in (None, "", []), f"{e['id']} missing {field}"
        assert e["taxonomy_slot"] in range(1, 11)
        assert e["exp_tier"] in R.EXP_RUN_TIERS
        assert e["resource_tier"] in R.EXP_RESOURCE_TIERS
        assert e["status"] in R.EXP_STATUS


def test_implemented_experiments_map_to_registry():
    by_id = {e["id"]: e for e in R.load_experiments()}
    for e in by_id.values():
        if e["status"] == "implemented" and e["kind"] in ("experiment", "cross-cutting"):
            assert e["id"] in REGISTRY, f"{e['id']} implemented but not registered"


def test_every_registry_id_is_catalogued():
    catalogued = {e["id"] for e in R.load_experiments()}
    for rid in REGISTRY:
        assert rid in catalogued, f"REGISTRY id {rid} not in experiments.yaml"


def test_implemented_diagnostics_name_existing_modules():
    from mop.config import REPO_ROOT

    for e in R.load_experiments():
        if e["status"] == "implemented" and e["kind"] in ("diagnostic", "ablation"):
            assert e.get("module") and (REPO_ROOT / e["module"]).exists(), f"{e['id']} module missing"


def test_active_custom_substrate_rows_present():
    ids = {e["id"] for e in R.load_experiments()}
    assert ids == {"mop_cm7_min_objective_probe", "mop_cm8_custom_jepa_pilot"}


def test_moonshots_are_not_runnable():
    for e in R.load_experiments():
        if e["resource_tier"] == "moonshot":
            assert e["status"] != "implemented"


def test_validator_rejects_bad_row():
    bad = dict(R.load_experiments()[0])
    bad["taxonomy_slot"] = 99
    bad["exp_tier"] = "warp-drive"
    probs = R.validate_experiment(bad, registry_ids=set(REGISTRY))
    assert any("taxonomy_slot" in p for p in probs)
    assert any("exp_tier" in p for p in probs)


def test_validator_blocks_unbacked_implemented_row():
    bad = dict(R.load_experiments()[0])
    bad["id"] = "ex_ghost"
    bad["kind"] = "experiment"
    bad["status"] = "implemented"
    probs = R.validate_experiment(bad, registry_ids=set(REGISTRY))
    assert any("not in experiments.REGISTRY" in p for p in probs)


def test_experiments_md_renders_from_registry():
    md = R.render_experiments_md()
    assert "GENERATED from registry/experiments.yaml" in md
    assert "mop_cm7_min_objective_probe" in md and "FAILURE_TAXONOMY" in md
    for e in R.load_experiments():
        assert e["id"] in md
