from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

import mop.studies.edcm1_event_triggered_coalition as edcm


def _observation(
    tick: int = 0,
    *,
    world_id: str = "unit-world",
    relative_goal: tuple[int, int] = (2, 1),
    novelty: tuple[int, ...] = (0, 0, 0, 0),
) -> edcm.VisibleObservation:
    return edcm.VisibleObservation(
        world_id=world_id,
        event_id=edcm.canonical_sha256({"world_id": world_id, "tick": tick}),
        tick=tick,
        local_blocked=(0, 0, 0, 1),
        relative_goal=relative_goal,
        previous_action=0,
        previous_reward=-0.01,
        novelty_channels=novelty,
    )


def _transition(
    before: edcm.VisibleObservation,
    action: int,
    after: edcm.VisibleObservation | None,
    *,
    blocked: bool = False,
    reward: float = -0.01,
) -> edcm.VisibleTransition:
    feedback = edcm.PublicFeedback(before.event_id, before.tick, action, reward, blocked, False)
    return edcm.VisibleTransition(before, action, feedback, after, after is None)


def _proposal(
    config: dict,
    observation: edcm.VisibleObservation,
    kind: str,
    action: int,
) -> edcm.ProposalMessage:
    work = edcm.AbstractWork()
    return edcm.ProposalMessage.create(
        observation=observation,
        specialist_id=kind,
        specialist_kind=kind,
        action=action,
        confidence=0.7,
        expected_progress=0.5,
        evidence=("unit",),
        state_payload={"kind": kind, "state": 1},
        weights=config["abstract_work"]["weights"],
        max_age=int(config["messages"]["delayed_max_age"]),
        work=work,
    )


def _gate_row_stub(seed: int) -> dict:
    return {
        "schema": "mop-edcm1-gate-row/v3",
        "seed": seed,
        "tune": {},
        "gate": {},
        "oracle_headroom": 0.0,
        "oracle_values": [],
        "unique_win_counts": {},
        "unique_win_rates": {},
        "niche_advantages": {},
        "best_gate_kind": "reactive_spatial",
        "best_gate_success_rate": 0.0,
        "tune_event_reference": {},
        "recurrent_tuning_budget": 0,
        "recurrent_candidates": {},
        "verifier": {},
    }


def test_official_authority_is_finalized_and_loads() -> None:
    config = edcm.load_config()
    assert edcm.OFFICIAL_AUTHORITY_SHA256 != "__AUTHORITY_SHA256__"
    assert edcm.canonical_sha256(config) == edcm.OFFICIAL_AUTHORITY_SHA256
    assert config["claim_scope"] == edcm.CLAIM_SCOPE
    assert config["verdict"]["scientific_promotion"] == "blocked"


def test_changed_payload_and_external_official_path_fail_closed(tmp_path: Path) -> None:
    envelope = json.loads(edcm.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    unchanged = tmp_path / "unchanged.json"
    unchanged.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="repository config"):
        edcm.load_config(unchanged)
    envelope["payload"]["criteria"]["min_work_saving_vs_always_on"] = "0.01"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        edcm.load_config(changed, exploratory=True)


def test_world_is_connected_and_noise_never_overlaps_change_windows() -> None:
    config = edcm.load_config()
    world = edcm.PartialChangePointWorld(config, config["seeds"][0], "tune", 0)
    distance = world._shortest_path(world.start, world.goal, world.walls)
    assert distance is not None
    assert distance <= config["world"]["max_shortest_path"]
    radius = config["world"]["change_window_radius"]
    change_window = {
        tick
        for point in config["world"]["change_points"]
        for tick in range(point - radius, point + radius + 1)
    }
    assert not change_window.intersection(config["world"]["noise_ticks"])
    while not world.terminal:
        observation = world.observe()
        if observation.tick in config["world"]["noise_ticks"]:
            assert any(observation.novelty_channels)
        world.step(0)


def test_visible_transition_contains_no_hidden_evaluator_fields() -> None:
    visible = {field.name for field in dataclasses.fields(edcm.VisibleTransition)}
    evaluator = {field.name for field in dataclasses.fields(edcm.EvaluatorTransition)}
    assert "hidden_change" not in visible
    assert "action_rotation" not in visible
    assert {"hidden_change", "action_rotation", "physical_action", "niche_label"}.issubset(evaluator)


def test_abstract_work_is_deterministic_and_timing_is_nonverdict() -> None:
    config = edcm.load_config()
    work = edcm.AbstractWork(scalar_ops=2, comparisons=3, nonlinearities=1, table_reads=2)
    weights = config["abstract_work"]["weights"]
    assert work.total(weights) == 2 + 3 + 4 + 2 * 3
    assert config["abstract_work"]["empirical_timing"] == "post-v3-nonverdict-benchmark-only"


def test_planner_learns_actual_visible_successors_and_rolls_them_out() -> None:
    config = edcm.load_config()
    planner = edcm.ShortHorizonPlannerProposer(history_capacity=8, horizon=2, discount=0.8)
    before = _observation(relative_goal=(2, 0))
    after = _observation(1, relative_goal=(1, 0))
    planner.update(_transition(before, 2, after))
    assert planner.history[0].before_key == before.state_key()
    assert planner.history[0].next_key == after.state_key()
    message, work = planner.propose(
        before,
        config["abstract_work"]["weights"],
        config["messages"]["delayed_max_age"],
    )
    assert "learned-visible-transition-model" in message.evidence
    assert message.integrity_valid()
    assert work.table_reads > 0 and work.table_writes > 0


def test_verifier_is_relational_and_abstains_when_alone() -> None:
    config = edcm.load_config()
    observation = _observation()
    verifier = edcm.ContradictionVerifier()
    one = _proposal(config, observation, "reactive_spatial", 0)
    abstention, _ = verifier.verify(observation, [one])
    assert abstention.abstained
    assert not hasattr(abstention, "proposed_action")
    two = _proposal(config, observation, "short_horizon_planner", 2)
    decision, _ = verifier.verify(observation, [one, two])
    assert not decision.abstained
    assert decision.endorsed_message_id in {one.message_id, two.message_id}


def test_hard_dispatch_calls_and_updates_only_active_proposers() -> None:
    config = edcm.load_config()
    controller = edcm.CoalitionController(config, "fixed")
    world = edcm.PartialChangePointWorld(config, config["seeds"][0], "tune", 1)
    before = {
        kind: (proposer.telemetry.propose_calls, proposer.telemetry.update_calls)
        for kind, proposer in controller.proposers.items()
    }
    prepared = controller.prepare(world.observe(), edcm.ActivationRecord(("reactive_spatial",)))
    resolution = controller.resolve(prepared)
    transition = world.step(resolution.action)
    controller.update(transition.visible)
    for kind, proposer in controller.proposers.items():
        expected = int(kind == "reactive_spatial")
        assert proposer.telemetry.propose_calls - before[kind][0] == expected
        assert proposer.telemetry.update_calls - before[kind][1] == expected
    assert controller.hard_dispatch_violations == 0


def test_verifier_lesion_prevents_both_verify_and_update_calls() -> None:
    config = edcm.load_config()
    controller = edcm.CoalitionController(config, "fixed")
    world = edcm.PartialChangePointWorld(config, config["seeds"][0], "tune", 4)
    record = edcm.ActivationRecord(
        ("reactive_spatial", "episodic_retrieval"),
        (edcm.VERIFIER_ID,),
    )
    prepared = controller.prepare(world.observe(), record)
    resolution = controller.resolve(prepared, "verifier_lesion")
    transition = world.step(resolution.action)
    controller.update(transition.visible)
    assert controller.verifier.verify_calls == 0
    assert controller.verifier.update_calls == 0
    assert controller.hard_dispatch_violations == 0


def test_channel_delay_preserves_referent_and_is_consumed_at_age_one() -> None:
    config = edcm.load_config()
    tick_zero = _observation(0)
    tick_one = _observation(1)
    delayed_planner = _proposal(config, tick_zero, "short_horizon_planner", 2)
    controller = edcm.CoalitionController(config, "fixed")
    prepared = controller.prepare(tick_one, edcm.ActivationRecord(("reactive_spatial",)))
    resolution = controller.resolve(prepared, "planner_channel_delay", delayed_planner)
    delivered_planner = [
        message for message in resolution.delivered if message.specialist_kind == "short_horizon_planner"
    ]
    assert len(delivered_planner) == 1
    assert delivered_planner[0].age(tick_one) == 1
    assert delivered_planner[0].referent_valid(tick_one)
    assert delivered_planner[0].usable(tick_one, 1)


def test_proposer_link_lesions_drop_only_the_named_message() -> None:
    config = edcm.load_config()
    observation = _observation()
    proposals = tuple(
        _proposal(config, observation, kind, action)
        for kind, action in zip(edcm.PROPOSER_ORDER, (0, 1, 2), strict=True)
    )
    prepared = edcm.PreparedDecision(
        observation,
        edcm.ActivationRecord(edcm.PROPOSER_ORDER),
        proposals,
        edcm.PROPOSER_ORDER,
        edcm.AbstractWork(),
        "unit",
    )
    conditions = {
        "reactive_link_lesion": "reactive_spatial",
        "episodic_link_lesion": "episodic_retrieval",
        "planner_link_lesion": "short_horizon_planner",
    }
    for condition, target in conditions.items():
        delivered, _, _ = edcm.apply_message_condition(config, prepared, condition, None)
        assert {message.specialist_kind for message in delivered} == set(edcm.PROPOSER_ORDER) - {target}


def test_round_matched_schedules_preserve_initial_and_extra_counts() -> None:
    records = [
        edcm.ActivationRecord(("reactive_spatial",)),
        edcm.ActivationRecord(("reactive_spatial", "episodic_retrieval"), (edcm.VERIFIER_ID,)),
        edcm.ActivationRecord(("reactive_spatial", "short_horizon_planner")),
        edcm.ActivationRecord(("reactive_spatial",), (edcm.VERIFIER_ID,)),
    ]
    expected = edcm.round_activation_counts(records)
    periodic = edcm.periodic_round_matched_schedule(records)
    shuffled = edcm.shuffled_round_matched_schedule(records, 17)
    coalition_shuffled = edcm.shuffled_coalition_matched_schedule(records, 17)
    assert edcm.round_activation_counts(periodic) == expected
    assert edcm.round_activation_counts(shuffled) == expected
    assert sum(bool(record.extra_round) for record in periodic) == 2
    assert all(record.extra_round in ((), (edcm.VERIFIER_ID,)) for record in shuffled)
    assert sorted((record.initial, record.extra_round) for record in coalition_shuffled) == sorted(
        (record.initial, record.extra_round) for record in records
    )
    assert {id(record) for record in coalition_shuffled} == {id(record) for record in records}


def test_homogeneous_copy_selection_rotates_balanced_ids() -> None:
    config = edcm.load_config()
    controller = edcm.HomogeneousController(config, "reactive_spatial")
    selected = []
    for tick in range(4):
        prepared = controller.prepare(_observation(tick), edcm.ActivationRecord(("reactive_spatial",)))
        selected.append(prepared.active_ids[0])
    assert len(set(selected)) == 4


def test_homogeneous_preserves_verifier_role_in_separate_round() -> None:
    config = edcm.load_config()
    controller = edcm.HomogeneousController(config, "reactive_spatial")
    reference = edcm.ActivationRecord(
        ("reactive_spatial", "episodic_retrieval"),
        (edcm.VERIFIER_ID,),
    )
    prepared = controller.prepare(_observation(), reference)
    resolution = controller.resolve(prepared)
    transition = _transition(_observation(), resolution.action, _observation(1))
    controller.update(transition)
    assert len(prepared.active_ids) == len(reference.initial)
    assert prepared.activation.extra_round == (edcm.VERIFIER_ID,)
    assert controller.verifier.verify_calls == 1
    assert controller.verifier.update_calls == 1


def test_recurrent_budget_has_only_meaningful_sweeps_and_no_padding() -> None:
    config = edcm.load_config()
    ledger = edcm.BudgetLedger(total_budget=5000, total_steps=1)
    controller = edcm.EqualBudgetRecurrentController(config, 7, 8, 0.03, 0.5, ledger)
    observation = _observation()
    initial_hidden = list(controller.hidden)
    trace = controller.act(observation)
    after = _observation(1, relative_goal=(1, 1))
    update_work = controller.update(_transition(observation, trace.action, after))
    total = trace.work.copy().add(update_work).total(config["abstract_work"]["weights"])
    assert trace.sweeps >= 1
    assert controller.hidden != initial_hidden
    assert ledger.spent == total
    assert ledger.credit < 5000
    assert not hasattr(controller, "padding_ops")


def test_clean_fixed_replay_reproduces_actions_in_tiny_fixture() -> None:
    config = copy.deepcopy(edcm.load_config())
    event = edcm.run_coalition_arm(
        config,
        seed=config["seeds"][0],
        split="tune",
        episodes=2,
        name="event",
        mode="event_triggered",
    )
    replay = [
        [dataclasses.replace(record, scheduler_tag="replay") for record in schedule]
        for schedule in event.schedules
    ]
    fixed = edcm.run_coalition_arm(
        config,
        seed=config["seeds"][0],
        split="tune",
        episodes=2,
        name="replay",
        mode="fixed",
        fixed_schedules=replay,
    )
    assert [episode.actions_sha256 for episode in event.episodes] == [
        episode.actions_sha256 for episode in fixed.episodes
    ]


def test_semantic_arm_verifier_rejects_forged_aggregate() -> None:
    config = edcm.load_config()
    accumulator = edcm.run_coalition_arm(
        config,
        seed=config["seeds"][0],
        split="tune",
        episodes=1,
        name="unit",
        mode="tuned_best_single",
        single_kind="reactive_spatial",
    )
    summary = accumulator.summary(config["world"]["horizon"])
    edcm.validate_arm_summary(summary, config, config["seeds"][0], "tune")
    with pytest.raises(ValueError, match="preregistered episode-count"):
        edcm.validate_arm_summary(
            summary,
            config,
            config["seeds"][0],
            "tune",
            expected_episode_count=2,
        )
    forged = copy.deepcopy(summary)
    forged["mean_utility"] += 0.1
    with pytest.raises(ValueError, match="utility aggregate mismatch"):
        edcm.validate_arm_summary(forged, config, config["seeds"][0], "tune")


def test_counterfactual_resolution_does_not_mutate_source_controller_state() -> None:
    config = edcm.load_config()
    controller = edcm.CoalitionController(config, "always_on")
    world = edcm.PartialChangePointWorld(config, config["seeds"][0], "tune", 2)
    prepared = controller.prepare(world.observe())
    clean = controller.resolve(prepared)
    before = edcm.canonical_sha256(controller.state_payload())
    effects = edcm.DirectEffects()
    edcm._collect_direct_effects(config, effects, world, controller, prepared, clean)
    assert edcm.canonical_sha256(controller.state_payload()) == before


def test_common_state_restoration_fork_has_identical_state_hashes() -> None:
    config = edcm.load_config()
    controller = edcm.CoalitionController(config, "always_on")
    world = edcm.PartialChangePointWorld(config, config["seeds"][0], "tune", 3)
    for _ in range(config["interventions"]["lesion_start_tick"]):
        prepared = controller.prepare(world.observe())
        resolution = controller.resolve(prepared)
        transition = world.step(resolution.action)
        controller.update(transition.visible)
    schedule = [
        edcm.ActivationRecord(edcm.PROPOSER_ORDER, (edcm.VERIFIER_ID,), "replay")
        for _ in range(config["world"]["horizon"])
    ]
    _, violations = edcm._restoration_counterfactual(config, world, controller, schedule)
    assert violations == 0


def test_checkpoint_binds_authority_implementation_and_phase_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = edcm.load_config()
    authority = edcm.canonical_sha256(config)
    implementation = edcm.canonical_sha256({"implementation": "unit"})
    implementation_authority = edcm.canonical_sha256({"manifest": "unit"})
    gate_row = _gate_row_stub(config["seeds"][0])
    checkpoint = tmp_path / "checkpoint.json"
    payload = edcm._write_checkpoint(
        checkpoint,
        authority,
        implementation_authority,
        implementation,
        [gate_row],
        [],
        config["resources"]["max_checkpoint_bytes"],
    )
    gate, heldout = edcm._load_checkpoint(
        checkpoint,
        authority,
        implementation_authority,
        implementation,
        config["seeds"],
        config["resources"]["max_checkpoint_bytes"],
    )
    assert gate == [gate_row] and heldout == []
    binding = {
        "file": edcm._file_receipt(checkpoint),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "gate_row_sha256": payload["gate_row_sha256"],
        "heldout_row_sha256": payload["heldout_row_sha256"],
    }
    checkpoint_reads = 0
    original_read = edcm._read_regular_file

    def counted_read(path: Path, max_bytes: int, label: str) -> bytes:
        nonlocal checkpoint_reads
        if Path(path).resolve() == checkpoint.resolve():
            checkpoint_reads += 1
        return original_read(path, max_bytes, label)

    monkeypatch.setattr(edcm, "_read_regular_file", counted_read)
    edcm._verify_checkpoint_binding(
        binding,
        checkpoint,
        authority,
        implementation_authority,
        implementation,
        config["seeds"],
        [gate_row],
        [],
        config["resources"]["max_checkpoint_bytes"],
    )
    assert checkpoint_reads == 1
    with pytest.raises(ValueError, match="implementation mismatch"):
        edcm._load_checkpoint(
            checkpoint,
            authority,
            implementation_authority,
            "wrong",
            config["seeds"],
            config["resources"]["max_checkpoint_bytes"],
        )
    with pytest.raises(ValueError, match="implementation authority mismatch"):
        edcm._load_checkpoint(
            checkpoint,
            authority,
            "wrong-manifest",
            implementation,
            config["seeds"],
            config["resources"]["max_checkpoint_bytes"],
        )


def test_written_checkpoint_snapshot_accepts_json_equivalent_nested_tuples(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tuple.checkpoint.json"
    expected = {"episode": {"actions": (0, 1, 2)}}
    edcm._atomic_json(checkpoint, expected)
    decoded, source_receipt = edcm._read_written_checkpoint_snapshot(
        checkpoint,
        expected,
        1024,
    )
    assert decoded == {"episode": {"actions": [0, 1, 2]}}
    assert source_receipt == edcm._file_receipt(checkpoint)


def test_incomplete_global_gate_never_authorizes_routing() -> None:
    config = edcm.load_config()
    result = edcm.aggregate_gate([], config)
    assert result == {"status": "incomplete", "passed": False}


def test_terminal_transition_retains_actual_successor_observation() -> None:
    config = edcm.load_config()
    world = edcm.PartialChangePointWorld(config, config["seeds"][0], "tune", 0)
    transition = None
    while not world.terminal:
        transition = world.step(0)
    assert transition is not None
    assert transition.visible.terminal
    assert transition.visible.after is not None
    assert transition.visible.after.tick == config["world"]["horizon"]


def test_producer_work_claim_includes_full_state_hashing() -> None:
    config = edcm.load_config()
    observation = _observation()
    small_work = edcm.AbstractWork(scalar_ops=1)
    large_work = edcm.AbstractWork(scalar_ops=1)
    small = edcm.ProposalMessage.create(
        observation=observation,
        specialist_id="small",
        specialist_kind="reactive_spatial",
        action=0,
        confidence=0.5,
        expected_progress=0.0,
        evidence=("unit",),
        state_payload={"state": "x"},
        weights=config["abstract_work"]["weights"],
        max_age=1,
        work=small_work,
    )
    large = edcm.ProposalMessage.create(
        observation=observation,
        specialist_id="large",
        specialist_kind="reactive_spatial",
        action=0,
        confidence=0.5,
        expected_progress=0.0,
        evidence=("unit",),
        state_payload={"state": "x" * 1000},
        weights=config["abstract_work"]["weights"],
        max_age=1,
        work=large_work,
    )
    assert large.producer_work_units > small.producer_work_units
    assert large.integrity_valid() and small.integrity_valid()


def test_accounting_sensitivity_reports_conservative_bounds() -> None:
    config = edcm.load_config()
    components = {name: 10 for name in dataclasses.asdict(edcm.AbstractWork())}
    report = edcm.accounting_sensitivity(
        components,
        config["abstract_work"]["weights"],
        config["abstract_work"]["sensitivity_factors"],
    )
    assert report["scenario_min"] < report["nominal"] < report["scenario_max"]
    assert set(report["one_at_a_time"]) == set(components)


def test_two_tick_delay_assay_uses_common_origin_and_exact_origin_message() -> None:
    config = edcm.load_config()
    controller = edcm.CoalitionController(config, "always_on")
    world = edcm.PartialChangePointWorld(config, config["seeds"][0], "tune", 0)
    prepared = controller.prepare(world.observe())
    controller.resolve(prepared)
    result = edcm._two_tick_channel_delay_assay(config, world, controller, prepared)
    assert result is not None
    _, _, violations = result
    assert violations == 0


def test_recurrent_td_bootstrap_depends_on_successor_and_charges_encoding() -> None:
    config = edcm.load_config()
    controllers = []
    for _ in range(2):
        ledger = edcm.BudgetLedger(total_budget=8000, total_steps=1)
        controller = edcm.EqualBudgetRecurrentController(config, 19, 8, 0.08, 0.5, ledger)
        controller.w_input = [[0.0] * 12 for _ in range(controller.hidden_size)]
        controller.w_input[0][4] = 1.0
        controller.w_input[0][8] = 1.0
        controller.w_hidden = [[0.0] * controller.hidden_size for _ in range(controller.hidden_size)]
        controller.q_head = [[0.0] * controller.hidden_size for _ in range(4)]
        for action in range(4):
            controller.q_head[action][0] = 0.1 * (action + 1)
        controllers.append(controller)
    before = _observation(relative_goal=(2, 0))
    traces = [controller.act(before) for controller in controllers]
    assert traces[0].action == traces[1].action
    after_a = _observation(1, relative_goal=(1, 0))
    after_b = _observation(1, relative_goal=(1, 0), novelty=(1, 0, 0, 0))
    work_a = controllers[0].update(_transition(before, traces[0].action, after_a))
    work_b = controllers[1].update(_transition(before, traces[1].action, after_b))
    assert controllers[0].q_head != controllers[1].q_head
    assert work_a.nonlinearities >= controllers[0].hidden_size
    assert work_b.nonlinearities >= controllers[1].hidden_size


def test_mixed_json_resumed_and_fresh_gate_rows_ignore_candidate_insertion_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = edcm.load_config()
    fresh_keys = [
        edcm._candidate_key(int(hidden), float(learning), float(reservoir))
        for hidden in config["recurrent_control"]["hidden_sizes"]
        for learning in config["recurrent_control"]["learning_rates"]
        for reservoir in config["recurrent_control"]["reservoir_scales"]
    ]
    fresh_candidates = {key: {"mean_utility": float(index)} for index, key in enumerate(fresh_keys)}
    resumed_candidates = json.loads(json.dumps(fresh_candidates, sort_keys=True))
    assert tuple(resumed_candidates) != tuple(fresh_candidates)

    def row(seed: int, candidates: dict) -> dict:
        arm = {"mean_utility": 0.1, "hard_dispatch_violations": 0}
        return {
            "seed": seed,
            "oracle_headroom": 1.0,
            "niche_advantages": {kind: [1.0] for kind in edcm.PROPOSER_ORDER},
            "verifier": {
                "disagreement_gain": 1.0,
                "agreement_absolute_effect": 0.0,
                "hard_dispatch_violations": 0,
            },
            "tune": {kind: dict(arm) for kind in edcm.PROPOSER_ORDER},
            "gate": {kind: {**arm, "success_rate": 0.5} for kind in edcm.PROPOSER_ORDER},
            "unique_win_counts": {kind: 20 for kind in edcm.PROPOSER_ORDER},
            "unique_win_rates": {kind: 0.1 for kind in edcm.PROPOSER_ORDER},
            "best_gate_success_rate": 0.5,
            "recurrent_candidates": candidates,
        }

    rows = [
        row(config["seeds"][0], resumed_candidates),
        *[row(seed, copy.deepcopy(fresh_candidates)) for seed in config["seeds"][1:]],
    ]
    monkeypatch.setattr(edcm, "validate_gate_row", lambda *_: None)
    result = edcm.aggregate_gate(rows, config)
    assert result["status"] == "complete"
    selected_key = max(
        fresh_keys,
        key=lambda key: (fresh_candidates[key]["mean_utility"], key),
    )
    assert result["selected_recurrent"] == edcm._candidate_payload(selected_key)


def test_semantic_replay_rejects_rehashed_action_forgery() -> None:
    config = edcm.load_config()
    seed = config["seeds"][0]
    summary = edcm.run_coalition_arm(
        config,
        seed=seed,
        split="tune",
        episodes=1,
        name="semantic-unit",
        mode="tuned_best_single",
        single_kind="reactive_spatial",
    ).summary(config["world"]["horizon"])
    forged = copy.deepcopy(summary)
    original_claim = {
        key: forged["per_episode"][0][key] for key in ("total_return", "success", "utility", "niche_values")
    }
    replacement = None
    for action in range(4):
        candidate = [action] * config["world"]["horizon"]
        if edcm.replay_episode_actions(config, seed, "tune", 0, candidate) != original_claim:
            replacement = candidate
            break
    assert replacement is not None
    forged["per_episode"][0]["actions"] = replacement
    forged["per_episode"][0]["actions_sha256"] = edcm.canonical_sha256(replacement)
    edcm.validate_arm_summary(forged)
    with pytest.raises(ValueError, match="semantic replay"):
        edcm.validate_arm_summary(forged, config, seed, "tune")


def test_intervention_capture_is_capped_and_evidence_is_compact() -> None:
    config = copy.deepcopy(edcm.load_config())
    config["splits"]["intervention_episodes"] = 1
    accumulator = edcm.run_coalition_arm(
        config,
        seed=config["seeds"][0],
        split="tune",
        episodes=2,
        name="intervention-cap",
        mode="event_triggered",
        capture_interventions=True,
    )
    summary = accumulator.summary(config["world"]["horizon"])
    direct = summary["direct_effects"]
    assert "raw" not in direct
    assert "restoration_gains" not in direct
    assert direct["restoration"]["count"] == 1
    assert summary["noise_pair_opportunities"] == len(config["world"]["noise_ticks"])
    assert all(evidence["count"] <= config["world"]["horizon"] for evidence in direct["metrics"].values())
    forged = copy.deepcopy(direct)
    first_metric = next(iter(forged["metrics"].values()))
    first_metric["count"] = config["world"]["horizon"] + 1
    assert not edcm._direct_effect_counts_within_cap(forged, config)


def test_compact_causal_evidence_is_authenticated_by_regeneration() -> None:
    config = copy.deepcopy(edcm.load_config())
    config["splits"]["intervention_episodes"] = 1
    stored = edcm.run_coalition_arm(
        config,
        seed=config["seeds"][0],
        split="heldout",
        episodes=1,
        name="event_triggered",
        mode="event_triggered",
        capture_interventions=True,
    ).summary(config["world"]["horizon"])
    row = {"seed": config["seeds"][0], "arms": {"event_triggered": stored}}
    edcm.validate_intervention_evidence(row, config)
    forged = copy.deepcopy(row)
    first_metric = next(iter(forged["arms"]["event_triggered"]["direct_effects"]["metrics"].values()))
    first_metric["mean"] += 1.0
    with pytest.raises(ValueError, match="regenerated intervention evidence mismatch"):
        edcm.validate_intervention_evidence(forged, config)


def test_compact_gate_verifier_evidence_is_authenticated_by_regeneration() -> None:
    config = copy.deepcopy(edcm.load_config())
    config["splits"]["gate_episodes"] = 1
    verifier = edcm.run_verifier_gate(config, config["seeds"][0])
    row = {"seed": config["seeds"][0], "verifier": verifier}
    edcm.validate_gate_verifier_evidence(row, config)
    forged = copy.deepcopy(row)
    forged["verifier"]["disagreement_gain"] += 1.0
    with pytest.raises(ValueError, match="regenerated gate verifier evidence mismatch"):
        edcm.validate_gate_verifier_evidence(forged, config)


def test_prospective_and_actual_artifact_guards_fail_closed(tmp_path: Path) -> None:
    config = copy.deepcopy(edcm.load_config())
    config["resources"]["prospective_episode_record_bytes"] = 10**9
    with pytest.raises(ValueError, match="prospective artifact"):
        edcm._prospective_artifact_guard(config)
    checkpoint = tmp_path / "too-large.json"
    with pytest.raises(ValueError, match="checkpoint byte envelope"):
        edcm._write_checkpoint(checkpoint, "a", "manifest", "b", [], [], 8)


def test_checkpoint_binding_rejects_semantically_different_rows(tmp_path: Path) -> None:
    config = edcm.load_config()
    authority = edcm.canonical_sha256(config)
    implementation = "unit-implementation"
    expected = _gate_row_stub(config["seeds"][0])
    changed = copy.deepcopy(expected)
    changed["best_gate_kind"] = "episodic_retrieval"
    checkpoint = tmp_path / "changed.checkpoint.json"
    payload = edcm._write_checkpoint(
        checkpoint,
        authority,
        "unit-manifest",
        implementation,
        [changed],
        [],
        config["resources"]["max_checkpoint_bytes"],
    )
    binding = {
        "file": edcm._file_receipt(checkpoint),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "gate_row_sha256": payload["gate_row_sha256"],
        "heldout_row_sha256": payload["heldout_row_sha256"],
    }
    with pytest.raises(ValueError, match="gate row join mismatch"):
        edcm._verify_checkpoint_binding(
            binding,
            checkpoint,
            authority,
            "unit-manifest",
            implementation,
            config["seeds"],
            [expected],
            [],
            config["resources"]["max_checkpoint_bytes"],
        )


def test_selected_heldout_controls_must_equal_gate_selection() -> None:
    gate = {
        "selected_best_single": "reactive_spatial",
        "selected_recurrent": {"hidden_size": 8, "learning_rate": 0.03, "reservoir_scale": 0.5},
    }
    row = {
        "selected_best_single": "reactive_spatial",
        "selected_recurrent": dict(gate["selected_recurrent"]),
    }
    edcm._validate_selected_controls([row], gate)
    row["selected_best_single"] = "episodic_retrieval"
    with pytest.raises(ValueError, match="selected single"):
        edcm._validate_selected_controls([row], gate)


def test_execution_manifest_distinguishes_partial_gate_stop_and_complete() -> None:
    config = edcm.load_config()
    seeds = config["seeds"]
    partial = edcm._execution_manifest(config, {"status": "incomplete", "passed": False}, [], [])
    assert partial["execution_status"] == "partial"
    assert partial["resumable"] and not partial["all_ok"]
    gate_rows = [{"seed": seed} for seed in seeds]
    stopped = edcm._execution_manifest(config, {"status": "complete", "passed": False}, gate_rows, [])
    assert stopped["execution_status"] == "terminal_scientific_stop"
    assert stopped["all_ok"] and not stopped["resumable"]
    heldout_rows = [{"seed": seed} for seed in seeds]
    complete = edcm._execution_manifest(
        config, {"status": "complete", "passed": True}, gate_rows, heldout_rows
    )
    assert complete["execution_status"] == "complete"
    assert complete["completed_heldout_seeds"] == seeds
    with pytest.raises(ValueError, match="nonterminal partial"):
        edcm._require_terminal_execution(partial)
    edcm._require_terminal_execution(stopped)
    edcm._require_terminal_execution(complete)


def test_out_alias_and_partial_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parsed = edcm.build_parser().parse_args(["--out", str(tmp_path / "proof.json")])
    assert parsed.output == tmp_path / "proof.json"
    monkeypatch.setattr(
        edcm,
        "run_from_config",
        lambda *args, **kwargs: {"aggregate": {"status": "incomplete"}, "resumable": True},
    )
    assert edcm.main(["--out", str(tmp_path / "proof.json")]) == 2
    monkeypatch.setattr(
        edcm,
        "run_from_config",
        lambda *args, **kwargs: {
            "aggregate": {"status": "complementarity-gate-failed"},
            "resumable": False,
        },
    )
    assert edcm.main(["--out", str(tmp_path / "proof.json")]) == 0


def test_negative_max_new_seeds_is_rejected_before_execution() -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        edcm.run_from_config(max_new_seeds=-1)


def test_verification_exit_is_zero_only_after_valid_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    proof = tmp_path / "proof.json"
    verification_output = tmp_path / "verification.json"
    terminal_result = {
        "valid": True,
        "execution_status": "complete",
        "scientific_promotion": False,
    }
    monkeypatch.setattr(
        edcm,
        "verify_receipt",
        lambda *args, **kwargs: dict(terminal_result),
    )
    assert (
        edcm.main(
            [
                "--verify",
                str(proof),
                "--verification-out",
                str(verification_output),
            ]
        )
        == 0
    )
    artifact = json.loads(verification_output.read_text())
    artifact_core = dict(artifact)
    artifact_digest = artifact_core.pop("verification_artifact_sha256")
    assert artifact["schema"] == edcm.VERIFICATION_ARTIFACT_SCHEMA
    assert artifact["verification"] == terminal_result
    assert artifact_digest == edcm.canonical_sha256(artifact_core)

    partial_output = tmp_path / "partial-verification.json"
    monkeypatch.setattr(
        edcm,
        "verify_receipt",
        lambda *args, **kwargs: {
            "valid": True,
            "execution_status": "partial",
            "scientific_promotion": False,
        },
    )
    with pytest.raises(ValueError, match="valid terminal result"):
        edcm.main(
            [
                "--verify",
                str(proof),
                "--verification-out",
                str(partial_output),
            ]
        )
    assert not partial_output.exists()

    def invalid(*args, **kwargs):
        raise ValueError("invalid receipt")

    monkeypatch.setattr(edcm, "verify_receipt", invalid)
    with pytest.raises(ValueError, match="invalid receipt"):
        edcm.main(
            [
                "--verify",
                str(proof),
                "--verification-out",
                str(tmp_path / "invalid-verification.json"),
            ]
        )


def test_verification_artifact_is_atomic_bounded_and_cannot_overwrite_inputs(
    tmp_path: Path,
) -> None:
    result = {
        "valid": True,
        "execution_status": "terminal_scientific_stop",
        "scientific_promotion": False,
    }
    output = tmp_path / "verification.json"
    artifact = edcm.write_verification_artifact(output, result)
    assert json.loads(output.read_text()) == artifact
    assert not output.with_suffix(".json.tmp").exists()
    with pytest.raises(ValueError, match="artifact path collision"):
        edcm.write_verification_artifact(
            output,
            result,
            protected_paths={"source_receipt": output},
        )


def test_exploratory_verification_requires_a_nonofficial_artifact_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exploratory verification may not use"):
        edcm.main(["--verify", str(tmp_path / "receipt.json"), "--exploratory"])


def test_official_receipt_cannot_select_diagnostic_verifier_mode() -> None:
    with pytest.raises(ValueError, match="official receipts require full deterministic regeneration"):
        edcm.run_from_config(
            max_new_seeds=0,
            verifier_mode=edcm.DIAGNOSTIC_VERIFIER_MODE,
        )


def test_mutable_and_authority_artifact_paths_must_be_disjoint(tmp_path: Path) -> None:
    collision = tmp_path / "same.json"
    with pytest.raises(ValueError, match="artifact path collision"):
        edcm.run_from_config(
            output_path=collision,
            checkpoint_path=collision,
            implementation_authority_sha256="0" * 64,
            max_new_seeds=0,
        )
    with pytest.raises(ValueError, match="artifact path collision"):
        edcm._require_distinct_paths(
            {
                "case_variant_a": tmp_path / "Prospective.json",
                "case_variant_b": tmp_path / "prospective.json",
            }
        )

    inode_source = tmp_path / "inode-source.json"
    inode_source.write_text("source", encoding="utf-8")
    inode_alias = tmp_path / "inode-alias.json"
    inode_alias.hardlink_to(inode_source)
    with pytest.raises(ValueError, match="artifact path collision"):
        edcm._require_distinct_paths({"inode_source": inode_source, "inode_alias": inode_alias})


def test_receipt_verifier_is_same_implementation_with_replay_checks() -> None:
    assert edcm.verify_receipt.__module__ == edcm.run_from_config.__module__
    assert edcm.replay_episode_actions.__module__ == edcm.verify_receipt.__module__


def test_frozen_implementation_authority_self_hashes_and_binds_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = edcm.load_config()
    authority_path = tmp_path / "implementation-authority.json"
    document = edcm.build_implementation_authority(
        config_authority_sha256=edcm.canonical_sha256(config),
        mode="official",
        review_status=edcm.OFFICIAL_IMPLEMENTATION_REVIEW_STATUS,
    )
    edcm._atomic_json(authority_path, document)
    monkeypatch.setattr(edcm, "DEFAULT_IMPLEMENTATION_AUTHORITY_PATH", authority_path)
    with pytest.raises(ValueError, match="independently supplied"):
        edcm.load_implementation_authority(
            authority_path,
            config,
            expected_sha256=None,
            exploratory=False,
        )
    loaded = edcm.load_implementation_authority(
        authority_path,
        config,
        expected_sha256=document["manifest_sha256"],
        exploratory=False,
    )
    assert loaded["manifest_sha256"] == document["manifest_sha256"]
    assert loaded["files"] == [
        edcm._file_receipt(edcm.REPO_ROOT / relative) for relative in edcm.IMPLEMENTATION_PATHS
    ]

    tampered = copy.deepcopy(document)
    tampered["review_status"] = "forged"
    edcm._atomic_json(authority_path, tampered)
    with pytest.raises(ValueError, match="self-hash mismatch"):
        edcm.load_implementation_authority(
            authority_path,
            config,
            expected_sha256=document["manifest_sha256"],
            exploratory=False,
        )
    tampered_core = dict(tampered)
    tampered_core.pop("manifest_sha256")
    tampered["manifest_sha256"] = edcm.canonical_sha256(tampered_core)
    edcm._atomic_json(authority_path, tampered)
    with pytest.raises(ValueError, match="authority pin mismatch"):
        edcm.load_implementation_authority(
            authority_path,
            config,
            expected_sha256=document["manifest_sha256"],
            exploratory=False,
        )


def test_implementation_authority_writer_cannot_overwrite_scoped_input() -> None:
    with pytest.raises(ValueError, match="artifact path collision"):
        edcm.write_implementation_authority(edcm.DEFAULT_CONFIG_PATH)


def test_exploratory_implementation_authority_must_be_explicit(tmp_path: Path) -> None:
    config = edcm.load_config()
    authority_path = tmp_path / "exploratory-authority.json"
    document = edcm.build_implementation_authority(
        config_authority_sha256=edcm.canonical_sha256(config),
        mode="exploratory",
        review_status="unreviewed-local-variant",
        study_id=config["study_id"],
    )
    edcm._atomic_json(authority_path, document)
    loaded = edcm.load_implementation_authority(
        authority_path,
        config,
        expected_sha256=None,
        exploratory=True,
    )
    assert loaded["mode"] == "exploratory"


def test_official_full_regeneration_rejects_self_consistent_alternate_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(edcm.load_config())
    seed = config["seeds"][0]
    original = {
        "seed": seed,
        "arm": {
            "actions": [0, 1],
            "actions_sha256": edcm.canonical_sha256([0, 1]),
            "work_components": {"scalar_ops": 2},
            "total_abstract_work": 2,
        },
    }
    gate_result = {
        "status": "complete",
        "passed": True,
        "selected_best_single": "reactive_spatial",
        "selected_recurrent": {
            "hidden_size": 8,
            "learning_rate": 0.03,
            "reservoir_scale": 0.5,
        },
    }
    original_heldout = {
        "seed": seed,
        "arm": {
            "actions": [1, 0],
            "actions_sha256": edcm.canonical_sha256([1, 0]),
            "work_components": {"scalar_ops": 4},
            "total_abstract_work": 4,
        },
    }
    monkeypatch.setattr(edcm, "run_gate_seed", lambda *_: copy.deepcopy(original))
    monkeypatch.setattr(edcm, "aggregate_gate", lambda *_: dict(gate_result))
    monkeypatch.setattr(
        edcm,
        "run_heldout_seed",
        lambda *_: copy.deepcopy(original_heldout),
    )
    edcm.validate_full_regeneration(config, [original], [original_heldout], gate_result)

    alternate = copy.deepcopy(original)
    alternate["arm"]["actions"] = [2, 3]
    alternate["arm"]["actions_sha256"] = edcm.canonical_sha256([2, 3])
    alternate["arm"]["work_components"]["scalar_ops"] = 3
    alternate["arm"]["total_abstract_work"] = 3
    with pytest.raises(ValueError, match="full regeneration gate row mismatch"):
        edcm.validate_full_regeneration(config, [alternate], [original_heldout], gate_result)

    alternate_heldout = copy.deepcopy(original_heldout)
    alternate_heldout["arm"]["actions"] = [3, 2]
    alternate_heldout["arm"]["actions_sha256"] = edcm.canonical_sha256([3, 2])
    alternate_heldout["arm"]["work_components"]["scalar_ops"] = 5
    alternate_heldout["arm"]["total_abstract_work"] = 5
    with pytest.raises(ValueError, match="full regeneration heldout row mismatch"):
        edcm.validate_full_regeneration(
            config,
            [original],
            [alternate_heldout],
            gate_result,
        )


def test_artifact_read_caps_canonical_bytes_and_unknown_keys_fail(
    tmp_path: Path,
) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b'{"payload":"' + b"x" * 64 + b'"}\n')
    with pytest.raises(ValueError, match="before read"):
        edcm._read_json_artifact(oversized, 16, "receipt")

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text('{\n  "value": 1\n}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical encoded size/on-disk size mismatch"):
        edcm._read_json_artifact(noncanonical, 1024, "receipt")

    non_json_config = tmp_path / "non-json-config.yaml"
    non_json_config.write_text("schema: mop-edcm1-envelope/v3\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        edcm._read_envelope(non_json_config)

    with pytest.raises(ValueError, match="regular file"):
        edcm._read_regular_file(tmp_path, 1024, "receipt")

    with pytest.raises(ValueError, match="receipt keys mismatch"):
        edcm._require_exact_keys(
            {**{key: None for key in edcm.RECEIPT_KEYS}, "unknown": True},
            edcm.RECEIPT_KEYS,
            "receipt",
        )
    with pytest.raises(ValueError, match="receipt empirical timing keys mismatch"):
        edcm._require_exact_keys(
            {**edcm.EMPIRICAL_TIMING_CLAIM, "unknown": True},
            edcm.EMPIRICAL_TIMING_KEYS,
            "receipt empirical timing",
        )
    with pytest.raises(ValueError, match="receipt checkpoint binding keys mismatch"):
        edcm._require_exact_keys(
            {
                "file": {},
                "checkpoint_sha256": "x",
                "gate_row_sha256": [],
                "heldout_row_sha256": [],
                "unknown": True,
            },
            edcm.CHECKPOINT_BINDING_KEYS,
            "receipt checkpoint binding",
        )


def test_checkpoint_and_arm_unknown_keys_fail_closed(tmp_path: Path) -> None:
    config = edcm.load_config()
    checkpoint = tmp_path / "unknown-key.checkpoint.json"
    payload = edcm._write_checkpoint(
        checkpoint,
        edcm.canonical_sha256(config),
        "manifest",
        "implementation",
        [],
        [],
        config["resources"]["max_checkpoint_bytes"],
    )
    payload["unknown"] = True
    payload.pop("checkpoint_sha256")
    payload["checkpoint_sha256"] = edcm.canonical_sha256(payload)
    edcm._atomic_json(checkpoint, payload)
    with pytest.raises(ValueError, match="checkpoint keys mismatch"):
        edcm._load_checkpoint(
            checkpoint,
            edcm.canonical_sha256(config),
            "manifest",
            "implementation",
            config["seeds"],
            config["resources"]["max_checkpoint_bytes"],
        )

    summary = edcm.run_coalition_arm(
        config,
        seed=config["seeds"][0],
        split="tune",
        episodes=1,
        name="unknown-key-arm",
        mode="tuned_best_single",
        single_kind="reactive_spatial",
    ).summary(config["world"]["horizon"])
    summary["unknown"] = True
    with pytest.raises(ValueError, match="arm summary .* keys mismatch"):
        edcm.validate_arm_summary(summary)
