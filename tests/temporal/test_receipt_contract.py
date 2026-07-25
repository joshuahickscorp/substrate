import copy
import json

import pytest

from mop.temporal import io
from mop.temporal import receipt_contract as RC
from mop.temporal.runs import supervisor


CELL = "histmlp|large|linear|none|h1"
GRID = [400, 800]
SEEDS = [0, 1, 2]


def seal_run(payload):
    document = dict(payload, program=io.PROGRAM, source_commit="a" * 40,
                    source_tree_oid="b" * 40, result_hash_version="canonical_json_v2")
    document["result_sha256"] = io.sha_obj(document)
    return document


def reseal_run(document):
    document["result_sha256"] = io.sha_obj(
        {key: value for key, value in document.items() if key != "result_sha256"})
    return document


def seal_proof(payload):
    document = dict(payload, program=io.PROGRAM, source_commit="a" * 40,
                    source_tree_oid="b" * 40, sha256_version="canonical_json_v2")
    document["sha256"] = io.sha_obj(document)
    return document


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


def correction_fixture(tmp_path, monkeypatch, *, mode="replacement"):
    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(io, "RUNS", tmp_path / "runs")
    parent = io.RUNS / "e2_converge_extended" / "xshard_har_stream_25.json"
    write(parent, seal_run({"cell": CELL}))
    scores = {budget: [0.4, 0.5, 0.6] for budget in GRID}
    records = {
        budget: [
            {
                "bed": "har_stream",
                "cell": CELL,
                "seed": seed,
                "updates": budget,
                "score": score,
                "checkpoint_sha": f"{seed + 1:x}" * 64,
                **(
                    {"parent_score": score - 0.1, "delta": 0.1, "corrected_score": score}
                    if mode == "delta" else {}
                ),
            }
            for seed, score in zip(SEEDS, scores[budget], strict=True)
        ]
        for budget in GRID
    }
    projection = {
        "schema": RC.PROJECTION_VERSION,
        "mode": mode,
        "cell": CELL,
        "grid": GRID,
        "seeds": SEEDS,
        "parent_receipts": [{
            "path": parent.relative_to(io.ROOT).as_posix(),
            "sha256": io.sha_file(parent),
            "cell": CELL,
        }],
        "arm_records": records,
    }
    document = seal_run({
        "receipt_contract": RC.declaration(RC.CORRECTION, f"{mode}_arm_grid"),
        "correction_projection": projection,
        "bed": "har_stream",
        "cell": CELL,
        "curve": {budget: 0.5 for budget in GRID},
        "seed_scores": scores,
    })
    path = io.RUNS / "e2_converge_corrections" / "convergence_har_stream.json"
    write(path, document)
    return path, document, parent


def test_valid_correction_receipt_and_aggregate_adapter(monkeypatch, tmp_path):
    path, document, _ = correction_fixture(tmp_path, monkeypatch)
    RC.validate(RC.CORRECTION, document, path=path)
    original = copy.deepcopy(document)
    normalized = RC.adapt_for_aggregation(document)
    assert normalized["arm_records"] == document["correction_projection"]["arm_records"]
    assert normalized["curve"] == document["curve"]
    assert document == original and "arm_records" not in document


def test_correction_missing_fields_fail_closed(monkeypatch, tmp_path):
    path, document, _ = correction_fixture(tmp_path, monkeypatch)
    document.pop("correction_projection")
    reseal_run(document)
    with pytest.raises(RC.ContractError, match="projection missing"):
        RC.validate(RC.CORRECTION, document, path=path)


def test_stale_parent_receipt_is_rejected(monkeypatch, tmp_path):
    path, document, parent = correction_fixture(tmp_path, monkeypatch)
    write(parent, seal_run({"cell": CELL, "drift": True}))
    with pytest.raises(RC.ContractError, match="stale parent"):
        RC.validate(RC.CORRECTION, document, path=path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_arm", "arm identity"),
        ("duplicate_arm", "duplicate correction arm identity"),
        ("wrong_seed", "arm identity"),
        ("wrong_cell", "mismatched factorial cell"),
    ],
)
def test_correction_identity_mutations_are_rejected(monkeypatch, tmp_path, mutation, message):
    path, document, _ = correction_fixture(tmp_path, monkeypatch)
    projection = document["correction_projection"]
    if mutation == "wrong_arm":
        projection["arm_records"][400][0]["bed"] = "speech_stream"
    elif mutation == "duplicate_arm":
        projection["arm_records"][400][1]["seed"] = 0
    elif mutation == "wrong_seed":
        projection["arm_records"][400][2]["seed"] = 9
    else:
        projection["cell"] = "gru|large|linear|none|h1"
    reseal_run(document)
    with pytest.raises(RC.ContractError, match=message):
        RC.validate(RC.CORRECTION, document, path=path)


def test_incorrect_correction_delta_is_rejected(monkeypatch, tmp_path):
    path, document, _ = correction_fixture(tmp_path, monkeypatch, mode="delta")
    document["correction_projection"]["arm_records"][400][0]["delta"] = 0.2
    reseal_run(document)
    with pytest.raises(RC.ContractError, match="incorrect correction delta"):
        RC.validate(RC.CORRECTION, document, path=path)


def test_seal_drift_is_rejected(monkeypatch, tmp_path):
    path, document, _ = correction_fixture(tmp_path, monkeypatch)
    document["curve"][400] = 0.9
    with pytest.raises(RC.ContractError, match="seal drift"):
        RC.validate(RC.CORRECTION, document, path=path)


def test_old_schema_migration_is_append_only_and_exact(monkeypatch, tmp_path):
    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(io, "RUNS", tmp_path / "runs")
    path = io.RUNS / "e2_converge" / "cshard_har_stream_0.json"
    old = seal_run({"cell": CELL, "arm_records": {400: []}})
    write(path, old)
    source_bytes = path.read_bytes()
    sidecar = RC.migrate_path(path, RC.BASE_CONVERGENCE, "raw_arm_grid")
    assert path.read_bytes() == source_bytes
    assert sidecar.is_file() and sidecar != path
    assert RC.validate(RC.BASE_CONVERGENCE, old, path=path)["version"] == RC.VERSION
    path.write_text(json.dumps({**old, "drift": True}))
    with pytest.raises(RC.ContractError, match="no exact migration"):
        RC.validate(RC.BASE_CONVERGENCE, json.loads(path.read_text()), path=path)


def test_principal_and_verification_use_the_same_versioned_contract():
    principal = seal_run({
        "receipt_contract": RC.declaration(RC.PRINCIPAL, "factorial_runs"),
        "runs": [{"cell": CELL}],
    })
    verification = seal_proof({
        "receipt_contract": RC.declaration(RC.VERIFICATION, "independent_role_checks"),
        "role_b": {"all_pass": True},
        "role_c": {"all_pass": True},
    })
    RC.validate(RC.PRINCIPAL, principal)
    RC.validate(RC.VERIFICATION, verification)


def test_two_strike_hold_requires_implementation_change(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor.io, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path / "runs")
    authority = {"source_commit": "1" * 40, "source_tree_oid": "2" * 40}
    monkeypatch.setattr(supervisor, "implementation_authority", lambda: authority)
    detail = {"stage": "e2_converge_corrections", "identity": "convergence_har_stream",
              "error_class": "ContractError", "error": "missing arm records"}

    first = supervisor.record_deterministic_failure(detail, "3" * 64)
    assert first["state"] == "diagnosis_required"
    assert supervisor.held(detail["stage"], detail["identity"])
    supervisor.diagnose_for_single_retry(
        detail["stage"], detail["identity"], first["error_fingerprint"], "omitted projection located")
    assert not supervisor.held(detail["stage"], detail["identity"])

    second = supervisor.record_deterministic_failure(detail, "4" * 64)
    assert second["state"] == "implementation_change_required"
    assert supervisor.held(detail["stage"], detail["identity"])
    authority = {"source_commit": "5" * 40, "source_tree_oid": "6" * 40}
    assert not supervisor.held(detail["stage"], detail["identity"])
