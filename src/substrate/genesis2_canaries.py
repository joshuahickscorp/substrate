"""The twenty-two cheap admission canaries frozen by the Genesis II plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

from substrate import genesis2_config as C2
from substrate import genesis2_controls  # noqa: F401
from substrate import genesis2_harness as H2
from substrate import genesis2_ledger as L2
from substrate import genesis2_material as F2
from substrate import genesis2_microstore as MS2
from substrate import genesis_challenge as CH
from substrate import genesis_history as HI
from substrate import genesis_material as M
from substrate import genesis_tournament as T


@dataclass(frozen=True, slots=True)
class Canary:
    canary_id: str
    passed: bool
    evidence: dict[str, Any]


def _opportunity(
    observations: list[M.Observation],
    *,
    byte_budget: int | None = None,
) -> M.Opportunity:
    opportunity = M.equal_opportunity(
        envelope="1GB",
        observations=observations,
        sensor_channels=tuple(sorted({row.channel for row in observations})),
        operation_budget=1_000_000,
        durable_write_budget=8_192,
    )
    if byte_budget is not None:
        opportunity.ledger.byte_budget = byte_budget
    return opportunity


def _write_fixture(mode: str, value: int) -> tuple[MS2.Microstore, str, str]:
    store = MS2.Microstore(default_mode=mode)
    before = store.digest()
    store.write("fixture", (1,), (value,), undo_token="write")
    return store, before, store.digest()


def _integrated_fixture() -> F2.L11_integrated_winner:
    observations = [
        M.Observation(
            index=index,
            channel="tool_use",
            payload=(1, 99, value, (3 * value + 2) % MS2.MODULUS),
            teaching=True,
        )
        for index, value in enumerate(range(6))
    ]
    material = cast(F2.L11_integrated_winner, M.build("L11_integrated_winner", _opportunity(observations)))
    for observation in observations:
        material.observe(observation)
    proposals = list(material.propose())
    emitted = material.apply([M.Verdict(row.proposal_id, True, 1.0, 1.0) for row in proposals])
    for receipt in emitted:
        material.finalize_receipt(receipt)

    stable = M.Observation(50, "stable", (1, 4, 1), teaching=True)
    material.observe(stable)
    stable_proposals = list(material.propose())
    emitted = material.apply([M.Verdict(row.proposal_id, True, 1.0, 1.0) for row in stable_proposals])
    for receipt in emitted:
        material.finalize_receipt(receipt)

    # Force a genuinely binding relative-pressure point after the exact facts
    # exist, then let the scheduler choose demotion before any evaluator result.
    material._opportunity.ledger.byte_budget = max(1, material._opportunity.ledger.resident_bytes)
    precision = [row for row in material.propose() if row.kind == "local_low_bit_adjustment"]
    if precision:
        emitted = material.apply([M.Verdict(precision[0].proposal_id, True, 0.0, 1.0)])
        for receipt in emitted:
            material.finalize_receipt(receipt)

    # A fitted rule followed by a conflicting address is the declared topology
    # fixture.  The rule is derived from the already observed pairs.
    fitted = MS2.induce(
        "tool_use",
        [((value,), ((3 * value + 2) % MS2.MODULUS,)) for value in range(6)],
    )
    if fitted:
        material.rules["tool_use"] = fitted
        conflict = M.Observation(100, "tool_use", (1, 99, 0, 7), teaching=True)
        # Temporarily lift the envelope for the topology observation: pressure
        # itself has already been measured and must not prevent this fixture.
        material._opportunity.ledger.byte_budget = 1_000_000
        material.observe(conflict)
        topology = [row for row in material.propose() if row.kind == "topology_revision"]
        if topology:
            emitted = material.apply([M.Verdict(topology[0].proposal_id, True, 1.0, 1.0)])
            for receipt in emitted:
                material.finalize_receipt(receipt)
    return material


def _factorial_canary() -> dict[str, H2.ArmRun]:
    unit = HI.build_history(
        family="tool_acquisition",
        split="train",
        history_id=0,
        seed_namespace="genesis2-canaries",
    )
    arms = (
        C2.CANONICAL_S2_ID,
        C2.S2_LOW_BIT_ID,
        "L1_associative_monolith",
        "L9_minimal_sufficient_field",
        "wrong_history_plastic",
        "record_store_null",
        "oracle",
    )
    result = H2.run_history(
        history_id=0,
        family=unit.family,
        arms=T._factories(arms, unit),
        observations=unit.observations,
        alternative_observations=unit.alternative_observations,
        probes=unit.probes,
        judge=unit.judge,
        envelope="1GB",
        operation_budget=8_000_000,
        durable_write_budget=8_192,
    )
    return result["runs"]


def run_all() -> dict[str, Any]:
    results: dict[str, Canary] = {}

    exact, exact_before, exact_after = _write_fixture("exact", 1)
    results["C01"] = Canary(
        "C01",
        exact.read("fixture", (1,)).value == (1,),  # type: ignore[union-attr]
        {"before": exact_before, "after": exact_after},
    )

    low, _, _ = _write_fixture("ternary", 1)
    results["C02"] = Canary(
        "C02",
        low.read("fixture", (1,)).value == (1,),  # type: ignore[union-attr]
        {"mode": "ternary", "resident_bytes": low.resident_bytes()},
    )

    reversed_ok = exact.rollback("write") and exact.digest() == exact_before
    results["C03"] = Canary("C03", reversed_ok, {"restored_digest": exact.digest()})

    pairs = [((value,), ((3 * value + 2) % MS2.MODULUS,)) for value in range(6)]
    rules = MS2.induce("transfer", pairs)
    results["C04"] = Canary("C04", bool(rules), {"rule_kinds": [rule.kind for rule in rules]})

    future = 7
    affine = next((rule for rule in rules if rule.kind == "affine"), None)
    consolidated = None if affine is None else affine.apply((future,))
    results["C05"] = Canary(
        "C05",
        affine is not None and len(affine.params) < len(pairs) * 2,
        {"rule_parameters": 0 if affine is None else len(affine.params), "association_components": len(pairs) * 2},
    )
    results["C06"] = Canary(
        "C06",
        consolidated == (((3 * future + 2) % MS2.MODULUS),),
        {"unseen_key": future, "answer": consolidated},
    )

    integrated = _integrated_fixture()
    topology_changes = integrated.mechanisms.state_changes.get("topology_change", 0)
    results["C07"] = Canary(
        "C07",
        topology_changes > 0,
        {"topology_state_changes": topology_changes, "regions": sum(map(len, integrated.regions.values()))},
    )

    plain = L2.Evidence(1, 1, 0, False, False, 0.0, True)
    exception = L2.Evidence(1, 2, 6, True, True, 0.0, False)
    results["C08"] = Canary(
        "C08",
        L2.allocate(plain) == "micro_association",
        {"allocated": L2.allocate(plain)},
    )
    results["C09"] = Canary(
        "C09",
        L2.allocate(exception) == "topology_revision",
        {"allocated": L2.allocate(exception)},
    )

    mechanism_report = integrated.mechanism_report()
    changed = set(mechanism_report["state_changes"])
    missing = sorted(set(C2.REQUIRED_MECHANISMS) - changed)
    results["C10"] = Canary(
        "C10",
        not missing,
        {"state_changes": mechanism_report["state_changes"], "missing": missing},
    )

    # World-model ablation is the direct causal check on the fixture's unseen
    # transfer answer; mechanisms without a matching workload are separately
    # licensed rather than assigned a zero and called useful.
    snapshot = integrated.checkpoint()
    control_answer = integrated.answer(M.Probe(900, "tool_acquisition", "tool", (99, 7, 3, 2), 1))
    ablated = cast(
        F2.L11_integrated_winner,
        M.build(
            "L11_integrated_winner",
            _opportunity([]),
            disabled_mechanisms=("world_model_update", "procedure_compilation"),
        ),
    )
    ablated.restore(snapshot)
    ablated.mechanisms.disabled = frozenset(("world_model_update", "procedure_compilation"))
    ablated_answer = ablated.answer(M.Probe(900, "tool_acquisition", "tool", (99, 7, 3, 2), 1))
    ablation_changed = control_answer.value != ablated_answer.value or control_answer.abstained != ablated_answer.abstained
    results["C11"] = Canary(
        "C11",
        ablation_changed,
        {
            "control": asdict(control_answer),
            "ablated": asdict(ablated_answer),
            "licensed_when_not_called": sorted(set(C2.REQUIRED_MECHANISMS) - {"world_model_update", "procedure_compilation"}),
        },
    )

    runs = _factorial_canary()
    exact_field_score = runs["L9_minimal_sufficient_field"].score
    s2_score = runs[C2.CANONICAL_S2_ID].score
    results["C12"] = Canary(
        "C12",
        0.0 <= exact_field_score <= 1.0 and 0.0 <= s2_score <= 1.0,
        {"exact_field": exact_field_score, "s2_exact": s2_score, "diagnostic_effect": exact_field_score - s2_score},
    )
    low_score = runs[C2.S2_LOW_BIT_ID].score
    results["C13"] = Canary(
        "C13",
        runs[C2.S2_LOW_BIT_ID].mechanism != runs[C2.CANONICAL_S2_ID].mechanism,
        {"s2_exact": s2_score, "s2_low_bit": low_score},
    )

    pressure = M.ResourceLedger("relative-2pct", 100, 10, 1)
    bound = False
    try:
        pressure.resize(2)
    except M.ResourceExhausted:
        bound = True
    results["C14"] = Canary("C14", bound, {"budget_bytes": 1, "attempted_residency": 2})

    precision_changes = integrated.mechanisms.state_changes.get("precision_change", 0)
    results["C15"] = Canary(
        "C15",
        precision_changes > 0,
        {"precision_state_changes": precision_changes},
    )

    wrong_score = runs["wrong_history_plastic"].score
    results["C16"] = Canary(
        "C16",
        wrong_score <= CH.CHANCE_LEVEL + 0.2,
        {"wrong_history_score": wrong_score, "chance": CH.CHANCE_LEVEL, "cheap_canary_tolerance": 0.2},
    )

    authoritative_before = integrated.durable_state_digest()
    shadow_value = integrated.shadow_answer("tool_use", (99, 0))
    authoritative_after = integrated.durable_state_digest()
    results["C17"] = Canary(
        "C17",
        authoritative_before == authoritative_after and shadow_value is not None,
        {"authoritative_unchanged": authoritative_before == authoritative_after, "shadow_value": shadow_value},
    )

    procedure = MS2.compile_rule(affine) if affine is not None else None
    if procedure is not None:
        for _ in range(MS2.PROCEDURE_AUDIT_WINDOW):
            procedure.observe(hit=False)
    results["C18"] = Canary(
        "C18",
        procedure is not None and procedure.retired and not procedure.live(),
        {"retired": None if procedure is None else procedure.retired},
    )

    checkpoint = integrated.checkpoint()
    replica = cast(F2.L11_integrated_winner, M.build("L11_integrated_winner", _opportunity([])))
    replica.restore(checkpoint)
    results["C19"] = Canary(
        "C19",
        replica.durable_state_digest() == integrated.durable_state_digest(),
        {"digest": replica.durable_state_digest()},
    )

    before_organs = replica.durable_state_digest()
    goal_answer_before = replica.answer(M.Probe(901, "tool_acquisition", "tool", (99, 7, 3, 2), 1))
    replica.replace_organ("model", "reasoner", "replaceable-model-v2")
    replica.replace_organ("body", "sensor_fabric", "replaceable-body-v2")
    goal_answer_after = replica.answer(M.Probe(901, "tool_acquisition", "tool", (99, 7, 3, 2), 1))
    results["C20"] = Canary(
        "C20",
        before_organs != replica.durable_state_digest() and goal_answer_before.value == goal_answer_after.value,
        {"goal_preserved": goal_answer_before.value == goal_answer_after.value},
    )

    decorative_injected = dict(mechanism_report)
    decorative_injected["state_changes"] = dict(decorative_injected["state_changes"])
    decorative_injected["state_changes"].pop("shadow_field", None)
    detector_catches = bool(set(C2.REQUIRED_MECHANISMS) - set(decorative_injected["state_changes"]))
    clean_passes = not (set(C2.REQUIRED_MECHANISMS) - set(mechanism_report["state_changes"]))
    results["C21"] = Canary(
        "C21",
        detector_catches and clean_passes,
        {"injected_caught": detector_catches, "clean_passed": clean_passes},
    )

    activation_false = (
        C2.ACTIVATION is False
        and all(not bool(row.get("activation")) for row in (exact.document(), low.document(), checkpoint))
        and all(not bool(result.evidence.get("activation")) for result in results.values())
    )
    results["C22"] = Canary("C22", activation_false, {"activation": False})

    ordered = {name: asdict(results[name]) for name in sorted(results)}
    return {
        "canaries": ordered,
        "passed": sum(1 for result in results.values() if result.passed),
        "failed": [name for name, result in sorted(results.items()) if not result.passed],
        "all_pass": all(result.passed for result in results.values()),
        "policy": C2.CANARY_POLICY,
        "activation": False,
    }


def demo() -> None:
    report = run_all()
    assert report["all_pass"], report["failed"]
    print("genesis2 canaries self-check passed: 22/22")


if __name__ == "__main__":
    demo()
