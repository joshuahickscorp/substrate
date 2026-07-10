"""Semantic and fail-closed gates for the six formerly registry-only F-series lanes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mop import config, devices
from mop.diagnostics.performance_density import DENSITY_SCHEMA
from mop.experiments import REGISTRY, get_experiment
from mop.experiments import form_rewrite_engine as rewrite_engine
from mop.experiments.f_form_substrate_missing import ScientificExecutionRefused

CPU_IDS = (
    "f6_sensorimotor_form_closure",
    "f7_developmental_form_growth",
    "f11_form_dream_replay",
    "f15_embodied_affordance_form",
)
GATED_IDS = ("f8_plastic_substrate_rewrite", "f16_perfect_slate_null")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _build_rewrite_fixture(root: Path, eid: str) -> dict[str, str]:
    """Build a complete, content-hashed fixture package for the real scientific execution path."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7301)
    classes, per_class, input_dim = 3, 30, 6
    labels = np.repeat(np.arange(classes, dtype=np.int64), per_class)
    centers = rng.normal(0.0, 1.8, size=(classes, input_dim)).astype(np.float32)
    inputs = centers[labels] + rng.normal(0.0, 0.45, size=(len(labels), input_dim)).astype(np.float32)
    view_a = inputs + rng.normal(0.0, 0.08, size=inputs.shape).astype(np.float32)
    view_b = inputs + rng.normal(0.0, 0.08, size=inputs.shape).astype(np.float32)
    split = np.empty(len(labels), dtype=np.int64)
    for cls in range(classes):
        indices = np.flatnonzero(labels == cls)
        split[indices[:15]] = 0
        split[indices[15:20]] = 1
        split[indices[20:]] = 2
    domains = split.copy()
    referents = np.asarray([f"fixture-referent:{index:04d}" for index in range(len(labels))])
    dataset = root / "rewrite_dataset.npz"
    np.savez(
        dataset,
        inputs=inputs,
        view_a=view_a,
        view_b=view_b,
        factor_labels=labels,
        transfer_labels=labels.copy(),
        domain_labels=domains,
        split=split,
        referent_ids=referents,
        view_a_referent_ids=referents.copy(),
        view_b_referent_ids=referents.copy(),
    )

    dims = [input_dim, 8, 6]
    weight0 = rng.normal(0.0, 0.42, size=(dims[1], dims[0])).astype(np.float32)
    bias0 = rng.normal(0.0, 0.05, size=(dims[1],)).astype(np.float32)
    weight1 = rng.normal(0.0, 0.42, size=(dims[2], dims[1])).astype(np.float32)
    bias1 = rng.normal(0.0, 0.05, size=(dims[2],)).astype(np.float32)
    weights = root / "fixture_encoder_weights.npz"
    np.savez(weights, weight_0=weight0, bias_0=bias0, weight_1=weight1, bias_1=bias1)
    inherited = np.maximum(inputs @ weight0.T + bias0, 0.0) @ weight1.T + bias1
    features = root / "fixture_inherited_features.npy"
    np.save(features, inherited.astype(np.float32), allow_pickle=False)

    rights = _write_json(
        root / "data_rights.json",
        {
            "schema": "mop-rewrite-data-rights/v1",
            "dataset_schema": "mop-rewrite-dataset/v1",
            "artifact_class": "fixture",
            "rights_granted": True,
            "fixture_only": True,
            "natural_data": False,
            "split_frozen": True,
            "dataset_path": dataset.name,
            "dataset_sha256": _sha256(dataset),
            "license": "CC0 generated test fixture",
            "source": "tests/integration/test_f_missing_lanes.py deterministic generator",
            "referent_scheme": "fixture-global-referent-id",
        },
    )
    encoder = _write_json(
        root / "encoder_receipt.json",
        {
            "schema": "mop-rewrite-encoder-receipt/v1",
            "artifact_class": "fixture",
            "weights_real": False,
            "feature_cache_real": False,
            "model_id": "fixture-safe-mlp",
            "weights_format": "mop-mlp-npz/v1",
            "weights_path": weights.name,
            "weights_sha256": _sha256(weights),
            "inherited_features_path": features.name,
            "inherited_features_sha256": _sha256(features),
            "architecture_dims": dims,
            "activation": "relu",
            "feature_tolerance": 1.0e-5,
        },
    )
    arms = (
        ["plastic_rewrite", "frozen_inherited", "larger_frozen_shell", "random_init_same_arch"]
        if eid == "f8_plastic_substrate_rewrite"
        else ["blank_slate", "frozen_inherited", "larger_frozen_shell", "random_init_same_arch"]
    )
    budget = 500_000
    compute = _write_json(
        root / "compute_receipt.json",
        {
            "schema": "mop-matched-compute-plan/v1",
            "artifact_class": "fixture",
            "matched_compute": True,
            "budget_flops": budget,
            "arm_flops": {arm: budget for arm in arms},
            "tolerance": 0.02,
            "dataset_sha256": _sha256(dataset),
            "weights_sha256": _sha256(weights),
        },
    )
    evidence = {
        "data_rights_manifest": str(rights),
        "real_encoder_manifest": str(encoder),
        "matched_compute_receipt": str(compute),
    }
    evidence["seed_plan_receipt"] = str(
        _write_json(
            root / "seed_plan.json",
            {
                "schema": "mop-seed-plan/v1",
                "artifact_class": "fixture",
                "experiment_id": eid,
                "seed_count": 5,
                "seeds": [0, 1, 2, 3, 4],
                "heldout_split_frozen": True,
                "dataset_sha256": _sha256(dataset),
                "margin": 0.02,
                "compute_tolerance": 0.02,
                "budget_flops": budget,
            },
        )
    )
    if eid == "f8_plastic_substrate_rewrite":
        source_receipt = _write_json(
            root / "shell_control_source.json",
            {
                "schema": "mop-shell-control-result/v1",
                "experiment_id": eid,
                "shell_controls_exhausted": True,
                "dataset_sha256": _sha256(dataset),
                "weights_sha256": _sha256(weights),
            },
        )
        evidence["shell_failure_receipt"] = str(
            _write_json(
                root / "shell_failure.json",
                {
                    "schema": "mop-shell-failure-receipt/v1",
                    "artifact_class": "fixture",
                    "shell_controls_exhausted": True,
                    "receipt_path": source_receipt.name,
                    "receipt_sha256": _sha256(source_receipt),
                    "dataset_sha256": _sha256(dataset),
                    "weights_sha256": _sha256(weights),
                },
            )
        )
    else:
        source_receipt = _write_json(
            root / "inherited_baseline_source.json",
            {
                "schema": "mop-inherited-baseline-result/v1",
                "experiment_id": eid,
                "baseline_complete": True,
                "dataset_sha256": _sha256(dataset),
                "weights_sha256": _sha256(weights),
            },
        )
        evidence["inherited_baseline_receipt"] = str(
            _write_json(
                root / "inherited_baseline.json",
                {
                    "schema": "mop-inherited-baseline-receipt/v1",
                    "artifact_class": "fixture",
                    "baseline_complete": True,
                    "receipt_path": source_receipt.name,
                    "receipt_sha256": _sha256(source_receipt),
                    "dataset_sha256": _sha256(dataset),
                    "weights_sha256": _sha256(weights),
                },
            )
        )
    return evidence


def _scientific_overrides(evidence: dict[str, str]) -> tuple[str, ...]:
    return (
        "experiment.execution_mode=scientific",
        *(f"experiment.evidence.{name}={path}" for name, path in evidence.items()),
        "experiment.scientific.batch_size=12",
        "experiment.scientific.larger_shell_width=12",
        "experiment.scientific.max_compute_flops_per_arm_seed=1000000",
        "experiment.scientific.max_package_bytes=10000000",
        "experiment.scientific.max_resident_bytes=10000000",
        "experiment.scientific.max_trainable_params=100000",
        "experiment.scientific.max_seed_count=5",
        "experiment.scientific.max_total_flops=20000000",
    )


def _rebind_dataset_hash(evidence: dict[str, str]) -> None:
    rights_path = Path(evidence["data_rights_manifest"])
    rights = json.loads(rights_path.read_text())
    dataset_path = rights_path.parent / rights["dataset_path"]
    dataset_hash = _sha256(dataset_path)
    rights["dataset_sha256"] = dataset_hash
    _write_json(rights_path, rights)
    for name in ("matched_compute_receipt", "seed_plan_receipt"):
        path = Path(evidence[name])
        document = json.loads(path.read_text())
        document["dataset_sha256"] = dataset_hash
        _write_json(path, document)
    prerequisite_name = (
        "shell_failure_receipt" if "shell_failure_receipt" in evidence else "inherited_baseline_receipt"
    )
    prerequisite_path = Path(evidence[prerequisite_name])
    prerequisite = json.loads(prerequisite_path.read_text())
    source_path = prerequisite_path.parent / prerequisite["receipt_path"]
    source = json.loads(source_path.read_text())
    source["dataset_sha256"] = dataset_hash
    _write_json(source_path, source)
    prerequisite["dataset_sha256"] = dataset_hash
    prerequisite["receipt_sha256"] = _sha256(source_path)
    _write_json(prerequisite_path, prerequisite)


def _run(eid, tmp_path, overrides=()):
    cfg = config.compose([f"experiment={eid}", "device=cpu", *overrides])
    return get_experiment(eid).run(cfg, devices.resolve("cpu"), tmp_path / eid)


def test_missing_f_classes_are_registered_and_contracts_match_configs():
    for eid in (*CPU_IDS, *GATED_IDS):
        assert eid in REGISTRY
        exp = get_experiment(eid)
        cfg = config.compose([f"experiment={eid}", "device=cpu"])
        assert tuple(cfg.experiment.metric) == exp.metric
        assert str(cfg.experiment.null_hypothesis) == exp.null_hypothesis
        assert str(cfg.experiment.tier) == exp.tier
    for eid in CPU_IDS:
        assert len(config.compose([f"experiment={eid}"]).experiment.seeds) >= 5


@pytest.mark.parametrize(
    ("eid", "overrides"),
    [
        ("f6_sensorimotor_form_closure", ("experiment.epochs=20", "experiment.goal_trials=8")),
        (
            "f7_developmental_form_growth",
            (
                "experiment.tasks=2",
                "experiment.samples_per_task=80",
                "experiment.epochs_per_task=8",
                "experiment.w_final=12",
                "experiment.grow_add=4",
            ),
        ),
        (
            "f11_form_dream_replay",
            (
                "experiment.samples=80",
                "experiment.epochs_task0=8",
                "experiment.epochs_task1=8",
                "experiment.replay_samples=24",
            ),
        ),
        ("f15_embodied_affordance_form", ("experiment.epochs=15",)),
    ],
)
def test_cpu_lanes_run_with_multiseed_evidence(eid, overrides, tmp_path):
    out = _run(eid, tmp_path, overrides)
    assert isinstance(out["null_supported"], bool)
    assert len(out["seeds"]) >= 5
    assert out["seed_ci"]["n"] >= 5
    assert out["sign_flip"]["n"] >= 5
    assert out["density"]["schema"] == DENSITY_SCHEMA
    assert out["density"]["cost"] and out["density"]["density"]


def test_f6_action_is_a_first_class_form_and_controls_are_matched(tmp_path):
    out = _run("f6_sensorimotor_form_closure", tmp_path, ("experiment.epochs=20", "experiment.goal_trials=8"))
    assert out["matched_architecture"] and out["matched_updates"]
    assert "control" in out["form_audit"]["kinds"]
    assert "action_trace" in out["form_audit"]["tags"]
    assert set(out["rollout_r2_by_arm"]) == {"true", "blind", "shuffled"}
    assert out["environment_contract"]["verified"] is True
    assert out["environment_contract"]["natural_embodiment_claim"] is False


def test_f7_is_final_capacity_and_update_matched(tmp_path):
    out = _run(
        "f7_developmental_form_growth",
        tmp_path,
        (
            "experiment.tasks=2",
            "experiment.samples_per_task=80",
            "experiment.epochs_per_task=8",
            "experiment.w_final=12",
            "experiment.grow_add=4",
        ),
    )
    assert out["capacity_matched"] and out["updates_matched"]
    assert set(out["frontier_auc_by_arm"]) == {"developmental", "fixed_final", "random"}
    assert out["final_params"]["developmental"] == out["final_params"]["fixed_final"]
    assert out["final_params"]["developmental"] == out["final_params"]["random"]


def test_f11_prices_actual_bytes_and_contains_generator_collapse_controls(tmp_path):
    out = _run(
        "f11_form_dream_replay",
        tmp_path,
        (
            "experiment.samples=80",
            "experiment.epochs_task0=8",
            "experiment.epochs_task1=8",
            "experiment.replay_samples=24",
        ),
    )
    assert out["replay_samples_matched"]
    assert set(out["retention_by_arm"]) == {
        "stored_form",
        "raw_exemplar",
        "generated",
        "random_generator",
        "no_replay",
    }
    assert set(out["bytes_by_arm"]) == {
        "stored_form",
        "raw_exemplar",
        "generated",
        "random_generator",
    }
    assert all(value > 0 for value in out["bytes_by_arm"].values())


def test_f15_uses_consequence_form_and_exact_head_controls(tmp_path):
    out = _run("f15_embodied_affordance_form", tmp_path, ("experiment.epochs=15",))
    assert out["matched_architecture"] and out["matched_updates"]
    assert "action_consequence" in out["form_audit"]["tags"]
    assert set(out["accuracy_by_arm"]) == {"intervention", "passive", "action_shuffle"}
    assert out["environment_contract"]["verified"] is True
    assert out["environment_contract"]["counterfactuals"] > 0


@pytest.mark.parametrize("eid", GATED_IDS)
def test_gated_defaults_are_nonpromotable_smokes_with_no_claim_metrics(eid, tmp_path):
    out = _run(eid, tmp_path)
    exp = get_experiment(eid)
    assert out["execution_status"] == "smoke-only"
    assert out["scientific_result"] is False
    assert out["promotion_eligible"] is False
    assert out["null_evaluated"] is False
    assert out["null_supported"] is None
    assert all(out[name] is None for name in exp.metric)
    receipt = json.loads((tmp_path / eid / "preflight_receipt.json").read_text())
    assert receipt["scientific_result"] is False
    assert receipt["promotion_eligible"] is False
    assert receipt["evidence_eligible"] is False


@pytest.mark.parametrize("eid", GATED_IDS)
def test_gated_scientific_mode_refuses_without_evidence_and_writes_receipt(eid, tmp_path):
    cfg = config.compose([f"experiment={eid}", "device=cpu", "experiment.execution_mode=scientific"])
    run_dir = tmp_path / eid
    with pytest.raises(ScientificExecutionRefused, match="scientific execution refused"):
        get_experiment(eid).run(cfg, devices.resolve("cpu"), run_dir)
    receipt = json.loads((run_dir / "preflight_receipt.json").read_text())
    assert receipt["requested_mode"] == "scientific"
    assert receipt["evidence_eligible"] is False
    assert any(check["status"] != "valid" for check in receipt["checks"].values())
    attempt = json.loads((run_dir / "attempt_receipt.json").read_text())
    projection = json.loads((run_dir / "resource_projection.json").read_text())
    assert attempt["status"] == "refused-preflight"
    assert projection["measured_hardware_wall"] is False


@pytest.mark.parametrize("eid", GATED_IDS)
def test_valid_fixture_reaches_full_scientific_engine_but_can_never_promote(eid, tmp_path):
    evidence = _build_rewrite_fixture(tmp_path / f"{eid}_package", eid)
    out = _run(eid, tmp_path / "run", _scientific_overrides(evidence))
    exp = get_experiment(eid)
    assert out["execution_status"] == "scientific-engine-complete"
    assert out["scientific_result"] is True
    assert out["null_evaluated"] is True
    assert isinstance(out["null_supported"], bool)
    assert all(isinstance(out[name], float) for name in exp.metric)
    assert out["evidence_scope"] == "fixture"
    assert out["fixture_taint_irreversible"] is True
    assert out["natural_claim_eligible"] is False
    assert out["promotion_eligible"] is False
    assert out["compute"]["matched"] is True
    assert out["compute"]["schema"] == "mop-matched-compute-estimate/v2"
    assert out["compute"]["actual_hardware_instruction_count_measured"] is False
    assert out["compute"]["hardware_time_and_energy_matched"] is False
    assert out["compute"]["relative_spread"] <= out["compute"]["tolerance"]
    assert out["heldout_domains"]["train_test_disjoint"] is True
    assert out["seed_ci"]["n"] == 5
    assert out["sign_flip"]["n"] == 5
    assert out["density"]["schema"] == DENSITY_SCHEMA
    assert out["density"]["cost"]["flops"] > 0
    assert out["density"]["cost"]["seconds"] > 0
    preflight = json.loads(Path(out["preflight_receipt"]).read_text())
    attempt = json.loads(Path(out["attempt_receipt"]).read_text())
    projection = json.loads(Path(out["resource_projection"]).read_text())
    assert preflight["evidence_eligible"] is True
    assert preflight["evidence_scope"] == "fixture"
    assert attempt["status"] == "completed"
    assert attempt["promotion_eligible"] is False
    assert projection["measured_hardware_wall"] is False
    assert set(projection["profiles"]) == {
        "m3pro-local-max",
        "studio-1tb",
        "studio-m1ultra",
    }
    assert projection["actual_attempt"]["completed"] is True
    assert out["resource_measurement"]["window_specific"] is True
    assert out["resource_measurement"]["rss_peak_sampled_bytes"] > 0
    if eid == "f8_plastic_substrate_rewrite":
        assert out["representation_rewrite_delta"] == out["representation_cosine_shift"]
    else:
        assert out["control_design"]["blank_and_random_control_share_initial_weights"] is True
        assert out["control_design"]["f16_random_control_encoder_frozen"] is True
        fingerprints = out["compute"]["initialization_fingerprints_by_arm_seed"]
        assert fingerprints["blank_slate"] == fingerprints["random_init_same_arch"]


def test_scientific_engine_refuses_tampered_dataset_before_training(tmp_path):
    eid = "f8_plastic_substrate_rewrite"
    package_root = tmp_path / "tampered_package"
    evidence = _build_rewrite_fixture(package_root, eid)
    dataset = package_root / "rewrite_dataset.npz"
    dataset.write_bytes(dataset.read_bytes() + b"tamper")
    cfg = config.compose([f"experiment={eid}", "device=cpu", *_scientific_overrides(evidence)])
    run_dir = tmp_path / "tampered_run"
    with pytest.raises(ScientificExecutionRefused, match="dataset sha256"):
        get_experiment(eid).run(cfg, devices.resolve("cpu"), run_dir)
    preflight = json.loads((run_dir / "preflight_receipt.json").read_text())
    attempt = json.loads((run_dir / "attempt_receipt.json").read_text())
    assert preflight["evidence_eligible"] is False
    assert preflight["checks"]["scientific_package"]["status"] == "invalid"
    assert attempt["status"] == "refused-evidence-integrity"
    assert attempt["promotion_eligible"] is False


def test_fixture_cannot_self_declare_natural_scope(tmp_path):
    eid = "f16_perfect_slate_null"
    package_root = tmp_path / "scope_laundering_package"
    evidence = _build_rewrite_fixture(package_root, eid)
    rights_path = Path(evidence["data_rights_manifest"])
    rights = json.loads(rights_path.read_text())
    rights["natural_data"] = True
    _write_json(rights_path, rights)
    cfg = config.compose([f"experiment={eid}", "device=cpu", *_scientific_overrides(evidence)])
    with pytest.raises(ScientificExecutionRefused, match="fixture data"):
        get_experiment(eid).run(cfg, devices.resolve("cpu"), tmp_path / "scope_laundering_run")


def test_coordinated_self_authored_natural_claim_still_cannot_promote(tmp_path):
    eid = "f8_plastic_substrate_rewrite"
    evidence = _build_rewrite_fixture(tmp_path / "coherent_laundering_package", eid)
    for name, path_text in evidence.items():
        path = Path(path_text)
        document = json.loads(path.read_text())
        document["artifact_class"] = "natural"
        if name == "data_rights_manifest":
            document["fixture_only"] = False
            document["natural_data"] = True
        if name == "real_encoder_manifest":
            document["weights_real"] = True
            document["feature_cache_real"] = True
            document["training_provenance"] = "self-authored and therefore not a trust authority"
        _write_json(path, document)
    out = _run(eid, tmp_path / "coherent_laundering_run", _scientific_overrides(evidence))
    assert out["natural_evidence_declared"] is True
    assert out["external_provenance_authority_required"] is True
    assert out["natural_claim_eligible"] is False
    assert out["promotion_eligible"] is False


def test_fixture_scientific_seed_plan_is_reproducible(tmp_path):
    eid = "f8_plastic_substrate_rewrite"
    evidence = _build_rewrite_fixture(tmp_path / "reproducible_package", eid)
    overrides = _scientific_overrides(evidence)
    first = _run(eid, tmp_path / "first", overrides)
    second = _run(eid, tmp_path / "second", overrides)
    for field in (*get_experiment(eid).metric, "accuracy_by_arm", "seed_ci", "sign_flip", "compute"):
        assert first[field] == second[field]


def test_scientific_engine_recomputes_inherited_features_not_just_their_hash(tmp_path):
    eid = "f8_plastic_substrate_rewrite"
    package_root = tmp_path / "false_feature_package"
    evidence = _build_rewrite_fixture(package_root, eid)
    features_path = package_root / "fixture_inherited_features.npy"
    features = np.load(features_path, allow_pickle=False)
    features[0, 0] += 0.25
    np.save(features_path, features, allow_pickle=False)
    encoder_path = Path(evidence["real_encoder_manifest"])
    encoder = json.loads(encoder_path.read_text())
    encoder["inherited_features_sha256"] = _sha256(features_path)
    _write_json(encoder_path, encoder)
    cfg = config.compose([f"experiment={eid}", "device=cpu", *_scientific_overrides(evidence)])
    with pytest.raises(ScientificExecutionRefused, match="not reproducible from the checkpoint"):
        get_experiment(eid).run(cfg, devices.resolve("cpu"), tmp_path / "false_feature_run")


def test_scientific_engine_refuses_package_above_explicit_resident_cap(tmp_path):
    eid = "f16_perfect_slate_null"
    evidence = _build_rewrite_fixture(tmp_path / "resident_cap_package", eid)
    cfg = config.compose(
        [
            f"experiment={eid}",
            "device=cpu",
            *_scientific_overrides(evidence),
            "experiment.scientific.max_resident_bytes=1",
        ]
    )
    run_dir = tmp_path / "resident_cap_run"
    with pytest.raises(ScientificExecutionRefused, match="resident-memory safety cap"):
        get_experiment(eid).run(cfg, devices.resolve("cpu"), run_dir)
    projection = json.loads((run_dir / "resource_projection.json").read_text())
    assert projection["measured_hardware_wall"] is False


def test_prerequisite_wrapper_must_hash_a_real_source_receipt(tmp_path):
    eid = "f8_plastic_substrate_rewrite"
    evidence = _build_rewrite_fixture(tmp_path / "tampered_prerequisite_package", eid)
    wrapper_path = Path(evidence["shell_failure_receipt"])
    wrapper = json.loads(wrapper_path.read_text())
    source_path = wrapper_path.parent / wrapper["receipt_path"]
    source_path.write_text(source_path.read_text() + " ")
    cfg = config.compose([f"experiment={eid}", "device=cpu", *_scientific_overrides(evidence)])
    with pytest.raises(ScientificExecutionRefused, match="prerequisite_receipt sha256"):
        get_experiment(eid).run(cfg, devices.resolve("cpu"), tmp_path / "tampered_prerequisite_run")


def test_compute_and_prerequisite_receipts_are_bound_to_dataset_and_weights(tmp_path):
    eid = "f16_perfect_slate_null"
    evidence = _build_rewrite_fixture(tmp_path / "unrelated_receipt_package", eid)
    compute_path = Path(evidence["matched_compute_receipt"])
    compute = json.loads(compute_path.read_text())
    compute["dataset_sha256"] = "a" * 64
    _write_json(compute_path, compute)
    cfg = config.compose([f"experiment={eid}", "device=cpu", *_scientific_overrides(evidence)])
    with pytest.raises(ScientificExecutionRefused, match="not bound to the rights-manifested dataset"):
        get_experiment(eid).run(cfg, devices.resolve("cpu"), tmp_path / "unrelated_receipt_run")


def test_hashed_prerequisite_source_is_parsed_not_blindly_trusted(tmp_path):
    eid = "f8_plastic_substrate_rewrite"
    evidence = _build_rewrite_fixture(tmp_path / "false_source_package", eid)
    wrapper_path = Path(evidence["shell_failure_receipt"])
    wrapper = json.loads(wrapper_path.read_text())
    source_path = wrapper_path.parent / wrapper["receipt_path"]
    source = json.loads(source_path.read_text())
    source["shell_controls_exhausted"] = False
    _write_json(source_path, source)
    wrapper["receipt_sha256"] = _sha256(source_path)
    _write_json(wrapper_path, wrapper)
    cfg = config.compose([f"experiment={eid}", "device=cpu", *_scientific_overrides(evidence)])
    with pytest.raises(ScientificExecutionRefused, match="source shell_controls_exhausted must be true"):
        get_experiment(eid).run(cfg, devices.resolve("cpu"), tmp_path / "false_source_run")


def test_f8_seed_margin_plan_is_immutable_and_package_bound(tmp_path):
    eid = "f8_plastic_substrate_rewrite"
    evidence = _build_rewrite_fixture(tmp_path / "seed_plan_package", eid)
    cfg = config.compose(
        [
            f"experiment={eid}",
            "device=cpu",
            *_scientific_overrides(evidence),
            "experiment.scientific.margin=0.03",
        ]
    )
    with pytest.raises(ScientificExecutionRefused, match="preregistered margin"):
        get_experiment(eid).run(cfg, devices.resolve("cpu"), tmp_path / "seed_plan_run")


@pytest.mark.parametrize(
    ("mutation", "problem"),
    [
        ("view_referent", "view_a_referent_ids must exactly equal"),
        ("fractional_split", "split values must be exact integers"),
        ("duplicate_cross_split", "duplicate input payloads cannot cross"),
    ],
)
def test_referent_and_split_leakage_guards_refuse_adversarial_packages(mutation, problem, tmp_path):
    eid = "f16_perfect_slate_null"
    package_root = tmp_path / f"{mutation}_package"
    evidence = _build_rewrite_fixture(package_root, eid)
    dataset_path = package_root / "rewrite_dataset.npz"
    with np.load(dataset_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if mutation == "view_referent":
        arrays["view_a_referent_ids"][0] = "wrong-referent"
    elif mutation == "fractional_split":
        arrays["split"] = arrays["split"].astype(np.float32)
        arrays["split"][0] = 0.5
    else:
        train_index = int(np.flatnonzero(arrays["split"] == 0)[0])
        test_index = int(np.flatnonzero(arrays["split"] == 2)[0])
        arrays["inputs"][test_index] = arrays["inputs"][train_index]
    np.savez(dataset_path, **arrays)
    _rebind_dataset_hash(evidence)
    cfg = config.compose([f"experiment={eid}", "device=cpu", *_scientific_overrides(evidence)])
    with pytest.raises(ScientificExecutionRefused, match=problem):
        get_experiment(eid).run(cfg, devices.resolve("cpu"), tmp_path / f"{mutation}_run")


def test_midrun_failure_preserves_consumed_compute_and_attempt_resource_window(monkeypatch, tmp_path):
    eid = "f8_plastic_substrate_rewrite"
    evidence = _build_rewrite_fixture(tmp_path / "partial_failure_package", eid)

    def fail_after_candidate(**_kwargs):
        raise RuntimeError("injected control-arm failure")

    monkeypatch.setattr(rewrite_engine, "_cached_feature_head", fail_after_candidate)
    cfg = config.compose([f"experiment={eid}", "device=cpu", *_scientific_overrides(evidence)])
    run_dir = tmp_path / "partial_failure_run"
    with pytest.raises(RuntimeError, match="injected control-arm failure"):
        get_experiment(eid).run(cfg, devices.resolve("cpu"), run_dir)
    progress = json.loads((run_dir / "scientific_progress.json").read_text())
    attempt = json.loads((run_dir / "attempt_receipt.json").read_text())
    projection = json.loads((run_dir / "resource_projection.json").read_text())
    assert progress["status"] == "failed"
    assert len(progress["completed_arms"]) == 1
    assert progress["completed_arms"][0]["arm"] == "plastic_rewrite"
    assert progress["consumed_estimated_flops"] > 0
    assert attempt["completed_arm_count"] == 1
    assert attempt["wall_seconds"] > 0
    assert attempt["resource_sample"]["window_specific"] is True
    assert projection["actual_attempt"]["completed"] is False
    assert projection["actual_attempt"]["consumed_estimated_flops"] > 0


def test_margin_decision_uses_per_seed_strongest_control_and_margin_adjusted_signs():
    decision = rewrite_engine._margin_null_decision(
        [0.03, 0.01, 0.03, 0.01, 0.03], margin=0.02, compute_matched=True
    )
    assert decision["rejects"] is False
    assert decision["sign_flip"]["any_flip"] is True
    assert decision["margin_adjusted_deltas"] == pytest.approx([0.01, -0.01, 0.01, -0.01, 0.01])
