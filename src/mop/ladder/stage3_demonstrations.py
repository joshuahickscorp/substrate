from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

SCHEMA = "mop-stage3-demonstration/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

STAGE3_EPOCHS: tuple[str, ...] = (
    "trace_stability",
    "niche_dispatch",
    "event_formation",
    "messaging_repair",
    "intervention_simulation",
    "memory_organization",
    "construction_search",
    "integrated_escs",
)

_Outcome = tuple[str, bool, bool, dict[str, Any]]


class Stage3DemonstrationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DemonstrationResult:
    epoch: str
    seed: int
    verdict_status: str
    null_holds: bool
    controls_ok: bool
    detail: dict[str, Any]
    schema: str = SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise Stage3DemonstrationError(f"unsupported demonstration schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise Stage3DemonstrationError("demonstration claim scope cannot be widened")
        if self.epoch not in STAGE3_EPOCHS:
            raise Stage3DemonstrationError(f"unknown Stage 3 epoch {self.epoch!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise Stage3DemonstrationError("seed must be a nonnegative integer")
        if not isinstance(self.null_holds, bool) or not isinstance(self.controls_ok, bool):
            raise Stage3DemonstrationError("null_holds and controls_ok must be booleans")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "epoch": self.epoch,
            "seed": self.seed,
            "verdict_status": self.verdict_status,
            "null_holds": self.null_holds,
            "controls_ok": self.controls_ok,
            "detail": self.detail,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def _refuses(thunk: Callable[[], object], expected: type[BaseException]) -> bool:

    try:
        thunk()
    except expected:
        return True
    return False


def _completes(thunk: Callable[[], object], expected: type[BaseException]) -> bool:

    try:
        thunk()
    except expected:
        return False
    return True


def _demo_trace_stability(seed: int) -> _Outcome:
    from ..mechanisms import trace_stability_scaffold as mech

    contract = mech.build_default_contract()
    seeds = (seed, seed + 1)
    records = mech.synthesize_trace_records(trace_id=contract.trace_id, seeds=seeds)
    outcomes = mech.dead_control_outcomes()
    controls_ok = _completes(
        lambda: mech.assert_controls_complete(mech.REQUIRED_CONTROLS),
        mech.TraceStabilityRefusal,
    )
    null_holds = _refuses(lambda: mech.MechanismLicenseGate().authorize(None), mech.TraceStabilityRefusal)
    detail: dict[str, Any] = {
        "contract_sha256": contract.sha256,
        "record_seeds": [int(value) for value in seeds],
        "record_digests": [record.digest() for record in records],
        "dead_controls": [outcome.control for outcome in outcomes],
        "controls_declared": list(mech.REQUIRED_CONTROLS),
        "activation_gate_closed": null_holds,
    }
    return "trace-stability-prior-null", null_holds, controls_ok, detail


def _demo_niche_dispatch(seed: int) -> _Outcome:
    from ..mechanisms import niche_dispatch_scaffold as mech

    contract = mech.build_default_dispatch_value_contract()
    bed = mech.synthesize_disjoint_bed(seeds=(seed, seed + 1))
    assessments = mech.synthesize_valid_assessments(seed=seed)
    controls_ok = tuple(contract.controls) == mech.DISPATCH_CONTROLS
    null_holds = _refuses(lambda: mech.DispatchActivationGate().authorize(None), mech.NicheDispatchRefusal)
    detail: dict[str, Any] = {
        "contract_digest": contract.digest(),
        "bed_digest": bed.digest(),
        "assessment_digests": [item.digest() for item in assessments],
        "controls_declared": list(mech.DISPATCH_CONTROLS),
        "activation_gate_closed": null_holds,
    }
    return "niche-dispatch-prior-null", null_holds, controls_ok, detail


def _demo_event_formation(seed: int) -> _Outcome:
    from ..mechanisms import event_formation_scaffold as mech

    episode = mech.synthesize_relational_episode(seed)
    verdict = mech.build_hypothetical_useful_verdict()
    controls_ok = _completes(
        lambda: mech.assert_control_ledger(mech.REQUIRED_CONTROLS), mech.EventFormationRefusal
    )
    null_holds = _refuses(
        lambda: mech.EventFormationActivationGate().authorize(None), mech.EventFormationRefusal
    )
    detail: dict[str, Any] = {
        "episode_digest": episode.digest(),
        "verdict_digest": verdict.digest(),
        "verdict_claims_useful_event": verdict.claims_useful_event(),
        "controls_declared": list(mech.REQUIRED_CONTROLS),
        "activation_gate_closed": null_holds,
        "note": (
            "the hypothetical useful verdict is a construction, not a measurement; the fail "
            "closed activation gate withholds activation so the x0 strong null holds"
        ),
    }
    return "x0-strong-null-holds", null_holds, controls_ok, detail


def _demo_messaging_repair(seed: int) -> _Outcome:
    from ..mechanisms import messaging_repair_scaffold as mech

    bounded = mech.default_bounded_message_contract()
    verification = mech.default_verification_value_contract()
    repair_contract = mech.default_contradiction_repair_contract()
    edges = [("node.root", "node.a"), ("node.root", "node.b"), ("node.root", "node.c")]
    plan = mech.causal_message_plan(edges=edges, bandwidth_limit=2, seed=seed)
    agreeing = [("agent.a", 1), ("agent.b", 1), ("agent.c", 1)]
    repair = mech.detect_and_repair(claims=agreeing, seed=seed)
    controls_ok = mech.verify_control_registry()
    null_holds = _refuses(lambda: mech.MessagingActivationGate().authorize(None), mech.MessagingRepairRefusal)
    detail: dict[str, Any] = {
        "bounded_contract_sha256": bounded.sha256,
        "verification_contract_sha256": verification.sha256,
        "repair_contract_sha256": repair_contract.sha256,
        "plan_sha256": plan.sha256,
        "repair_triggered": repair.triggered,
        "repair_sha256": repair.sha256,
        "activation_gate_closed": null_holds,
    }
    return "messaging-repair-prior-null", null_holds, controls_ok, detail


def _demo_intervention_simulation(seed: int) -> _Outcome:
    from ..mechanisms import intervention_simulation_scaffold as mech

    scaffold = mech.build_epoch_scaffold()
    outcome = mech.evaluate_control_win(
        arm_id="do-operator-arm",
        control_id="observational-only",
        seed=seed,
        margin_required=0.02,
    )
    controls_ok = _completes(mech.assert_control_registry_intact, mech.InterventionSimulationRefusal)
    null_holds = _refuses(mech.default_activation_gate().authorize, mech.InterventionSimulationRefusal)
    detail: dict[str, Any] = {
        "scaffold_sha256": scaffold.sha256,
        "control_registry_digest": mech.control_registry_digest(),
        "arm_score": outcome.arm_score,
        "control_score": outcome.control_score,
        "beats_control": outcome.beats_control,
        "activation_gate_closed": null_holds,
    }
    return "intervention-simulation-prior-null", null_holds, controls_ok, detail


def _demo_memory_organization(seed: int) -> _Outcome:
    from ..mechanisms import memory_organization_scaffold as mech

    contract = mech.build_memory_organization(seed)
    future = mech.default_future_decision_contract()
    values = mech.toy_decision_values(seed)
    controls_ok = tuple(future.controls) == mech.FUTURE_DECISION_CONTROLS
    null_holds = _refuses(
        lambda: mech.default_activation_gate().authorize(None), mech.MemoryOrganizationRefusal
    )
    detail: dict[str, Any] = {
        "organization_sha256": contract.sha256,
        "future_contract_sha256": future.sha256,
        "decision_values": {key: values[key] for key in sorted(values)},
        "controls_declared": list(mech.FUTURE_DECISION_CONTROLS),
        "activation_gate_closed": null_holds,
    }
    return "memory-organization-prior-null", null_holds, controls_ok, detail


def _demo_construction_search(seed: int) -> _Outcome:
    from ..mechanisms import construction_search_scaffold as mech

    contract = mech.build_default_contract()
    trace = mech.run_construction_search(seed=seed)
    verdict = mech.verdict_from_trace(trace)
    controls_ok = _completes(mech.build_default_control_set, mech.ConstructionSearchRefusal)
    null_holds = _refuses(
        mech.ConstructionSearchActivationGate().authorize_claim, mech.ConstructionSearchRefusal
    )
    detail: dict[str, Any] = {
        "contract_digest": contract.digest(),
        "trace_digest": trace.digest(),
        "verdict_digest": verdict.digest(),
        "toy_claims_improvement": verdict.claims_improvement,
        "net_improvement": verdict.net_improvement,
        "oracle_headroom_gap": verdict.oracle_headroom_gap,
        "activation_gate_closed": null_holds,
    }
    return "construction-search-prior-null", null_holds, controls_ok, detail


def _demo_integrated_escs(seed: int) -> _Outcome:
    from ..mechanisms import integrated_escs_scaffold as mech

    verdict = mech.build_null_frontier_verdict(seed)
    baselines = mech.build_default_baseline_set()
    controls_ok = _completes(
        lambda: baselines.assert_all_cost_matched(verdict.integrated.cost),
        mech.IntegratedEscsRefusal,
    )
    null_holds = _refuses(verdict.assert_advantage_earned, mech.IntegratedEscsRefusal)
    detail: dict[str, Any] = {
        "verdict_digest": verdict.digest(),
        "verdict_token": verdict.verdict(),
        "baseline_families": [row.family for row in baselines.declarations],
        "replications": verdict.replications,
        "activation_gate_closed": null_holds,
    }
    return verdict.verdict(), null_holds, controls_ok, detail


_HANDLERS: dict[str, Callable[[int], _Outcome]] = {
    "trace_stability": _demo_trace_stability,
    "niche_dispatch": _demo_niche_dispatch,
    "event_formation": _demo_event_formation,
    "messaging_repair": _demo_messaging_repair,
    "intervention_simulation": _demo_intervention_simulation,
    "memory_organization": _demo_memory_organization,
    "construction_search": _demo_construction_search,
    "integrated_escs": _demo_integrated_escs,
}


def run_demonstration(epoch: str, seed: int) -> DemonstrationResult:

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise Stage3DemonstrationError("seed must be a nonnegative integer")
    handler = _HANDLERS.get(epoch)
    if handler is None:
        raise Stage3DemonstrationError(f"unknown Stage 3 epoch {epoch!r}")
    verdict_status, null_holds, controls_ok, detail = handler(seed)
    return DemonstrationResult(
        epoch=epoch,
        seed=seed,
        verdict_status=verdict_status,
        null_holds=null_holds,
        controls_ok=controls_ok,
        detail=detail,
    )
