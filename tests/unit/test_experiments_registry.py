import yaml

from mop.config import REPO_ROOT
from mop.experiments import REGISTRY


def _rows():
    payload = yaml.safe_load((REPO_ROOT / "registry/experiments.yaml").read_text())
    return payload["experiments"]


def test_active_registry_is_exact_and_runnable_rows_are_bound():
    rows = _rows()
    ids = {row["id"] for row in rows}
    assert ids == {"mop_cm7_min_objective_probe", "mop_cm8_custom_jepa_pilot"}
    assert set(REGISTRY) == ids
    implemented = {row["id"] for row in rows if row["status"] == "implemented"}
    assert implemented <= set(REGISTRY)


def test_every_row_has_the_scientific_contract():
    required = {
        "id",
        "name",
        "question",
        "null_hypothesis",
        "falsifier",
        "metrics",
        "controls",
        "source",
        "split",
        "unit",
        "treatments",
        "sesoi",
        "multiplicity",
        "budget",
        "stop",
        "claim_ceiling",
        "provider",
        "verifier",
        "program",
        "resource_tier",
        "taxonomy_slot",
        "status",
        "proof",
    }
    for row in _rows():
        assert required <= set(row)
        assert all(row[field] not in (None, "", []) for field in required)
        assert row["taxonomy_slot"] in range(1, 11)


def test_registry_ids_match_runtime_objects():
    for experiment_id, experiment in REGISTRY.items():
        assert experiment.id == experiment_id
        assert experiment.null_hypothesis
        row = next(row for row in _rows() if row["id"] == experiment_id)
        assert experiment.declaration == row
