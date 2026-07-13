#!/usr/bin/env python3
"""Run the deterministic ESCS end-to-end mechanics smoke and emit an integrity receipt.

This exercises scripted mechanics only. It is not a capability or efficiency experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mop.escs.accounting import LifecycleLedger, WorkVector  # noqa: E402
from mop.escs.actors import (  # noqa: E402
    ActionIntent,
    ActorActivationContext,
    ActorActivationResult,
    ActorDescriptor,
    ActorUpdateContext,
    ActorUpdatePlan,
    ReadinessEstimate,
)
from mop.escs.archive import ArchiveCharge, BoundedArchive  # noqa: E402
from mop.escs.chassis import (  # noqa: E402
    ChassisStatus,
    EffectOutcome,
    EffectRequest,
    EventSourcedCoalitionChassis,
)
from mop.escs.event_former import (  # noqa: E402
    ChargedEventFormer,
    EventFormerConfig,
    EventFormerDecision,
    EventFormerDescriptor,
    EventProposal,
    RawPacket,
)
from mop.escs.events import (  # noqa: E402
    ConsequenceEvent,
    EpistemicStatus,
    EvidenceClass,
)
from mop.escs.ledger import EventLedger  # noqa: E402
from mop.escs.messages import SchemaRegistry  # noqa: E402
from mop.escs.runtime import (  # noqa: E402
    CandidateMode,
    CoalitionRuntime,
    RuntimeCaps,
    RuntimeConfig,
    ScriptedDispatchPolicy,
)
from mop.substrate.events import EventRef, canonical_bytes, canonical_sha256  # noqa: E402

CONFIG_SCHEMA = "mop-escs-mechanics-config/v1"
PROOF_SCHEMA = "mop-escs-mechanics-proof/v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/experiment/escs_mechanics_chassis.json"
DEFAULT_OUT = REPO_ROOT / "proof/ESCS_MECHANICS_CHASSIS.json"
_ROOT_FIELDS = {
    "schema",
    "claim_scope",
    "clock",
    "event_former",
    "runtime_caps",
    "archive",
    "nonclaims",
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != _ROOT_FIELDS:
        raise ValueError("ESCS mechanics config has missing or unknown root fields")
    if value["schema"] != CONFIG_SCHEMA:
        raise ValueError("unsupported ESCS mechanics config schema")
    if value["claim_scope"] != EvidenceClass.SCRIPTED_MECHANICS.value:
        raise ValueError("mechanics runner requires scripted-mechanics-only scope")
    if set(value["clock"]) != {"deployment_start_tick", "packet_tick", "final_tick"}:
        raise ValueError("ESCS mechanics clock fields drifted")
    clock = value["clock"]
    if not 0 <= clock["deployment_start_tick"] <= clock["packet_tick"] <= clock["final_tick"]:
        raise ValueError("ESCS mechanics clock order is invalid")
    if not isinstance(value["nonclaims"], list) or not all(
        isinstance(row, str) and row.strip() for row in value["nonclaims"]
    ):
        raise ValueError("ESCS mechanics nonclaims must be nonempty strings")
    EventFormerConfig(**value["event_former"])
    RuntimeCaps(**value["runtime_caps"])
    BoundedArchive(**value["archive"])
    return value, raw


class _ScriptedEventPolicy:
    def __init__(self) -> None:
        self._descriptor = EventFormerDescriptor(
            policy_id="event-former:escs-mechanics/v1",
            evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
        )
        self._state_version = _digest("escs-mechanics-event-policy-state")

    @property
    def descriptor(self) -> EventFormerDescriptor:
        return self._descriptor

    @property
    def state_version(self) -> str:
        return self._state_version

    @property
    def retained_state_bytes(self) -> int:
        return 64

    def evaluate(self, packet: RawPacket) -> EventFormerDecision:
        proposal = EventProposal.create(
            epistemic_status=EpistemicStatus.INFERRED,
            referent_hypotheses={"referent:fixture/one": 1.0},
            factor_change_distribution={"factor:motion": 1.0},
            decision_relevance_distribution={"relevant": 1.0},
            reducibility_distribution={"reducible": 1.0},
            calibrated_confidence=0.75,
            predicted_value_of_further_computation=0.5,
            formation_operations=2,
        )
        return EventFormerDecision(
            proposals=(proposal,),
            discarded_candidates=1,
            policy_operations=3,
            retained_state_bytes=self.retained_state_bytes,
            idle_operations=0,
            abstention_reason=None,
            evidence_class=self.descriptor.evidence_class,
        )


class _MechanicsActor:
    def __init__(self, *, event_ledger: EventLedger, counter: int = 0) -> None:
        self._event_ledger = event_ledger
        self._counter = counter
        self._descriptor = ActorDescriptor(
            actor_id="actor:escs-mechanics",
            subscribed_event_types=("hypothesis",),
        )

    @property
    def descriptor(self) -> ActorDescriptor:
        return self._descriptor

    @property
    def state_version(self) -> str:
        return _digest(f"escs-mechanics-actor:{self._counter}")

    @property
    def retained_state_bytes(self) -> int:
        return 32

    def readiness(self, _header) -> ReadinessEstimate:  # type: ignore[no-untyped-def]
        return ReadinessEstimate(
            actor_id=self.descriptor.actor_id,
            state_version=self.state_version,
            compatible=True,
            expected_decision_value=1.0,
            predicted_operations=3,
            predicted_message_bytes=0,
            estimation_operations=1,
        )

    def activate(self, context: ActorActivationContext) -> ActorActivationResult:
        action = ActionIntent.create(
            source_event_id=context.event_header.event_id,
            branch_id=context.event_header.branch_id,
            referent_hypotheses=context.event_header.referent_hypotheses,
            epistemic_status=context.event_header.epistemic_status,
            evidence_class=context.event_header.evidence_class,
            producer_actor_id=self.descriptor.actor_id,
            producer_state_version=self.state_version,
            created_tick=context.event_header.created_tick,
            expiry_tick=context.event_header.expiry_tick,
            producer_operations=2,
            payload_form="fixture-motor-command",
            payload_bytes=b"move:left",
        )
        return ActorActivationResult(action_intents=(action,), executed_operations=3)

    def stage_update(self, context: ActorUpdateContext) -> ActorUpdatePlan:
        consequence = self._event_ledger.get(EventRef(context.consequence_event_id))
        if not isinstance(consequence, ConsequenceEvent):
            raise ValueError("mechanics actor update lacks a ledger consequence")
        replacement = _MechanicsActor(event_ledger=self._event_ledger, counter=self._counter + 1)
        return ActorUpdatePlan(
            actor_id=self.descriptor.actor_id,
            prior_state_version=self.state_version,
            next_state_version=replacement.state_version,
            idempotency_key=context.idempotency_key,
            executed_operations=2,
            replacement_actor=replacement,
        )


class _FixtureEffect:
    def __init__(self, event_ledger: EventLedger) -> None:
        self._events = event_ledger
        self.requests: list[EffectRequest] = []

    def execute(self, request: EffectRequest) -> EffectOutcome:
        self._events.get(EventRef(request.commitment_event_id))
        self.requests.append(request)
        return EffectOutcome.create(
            observed_outcome={"fixture_position": "left"},
            realized_utility_vector={"fixture_reward": 1.0},
            realized_full_cost=WorkVector(actor_execution=2),
        )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _implementation_receipt() -> dict[str, Any]:
    paths = [Path(__file__), *sorted((REPO_ROOT / "src/mop/escs").glob("*.py"))]
    rows = []
    for path in paths:
        raw = path.read_bytes()
        rows.append(
            {
                "path": str(path.resolve().relative_to(REPO_ROOT)),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return {"files": rows, "manifest_sha256": canonical_sha256(rows)}


def run(config_path: Path, out_path: Path) -> dict[str, Any]:
    config, config_bytes = _load_config(config_path)
    events = EventLedger()
    lifecycle = LifecycleLedger()
    clock = config["clock"]
    former = ChargedEventFormer(
        policy=_ScriptedEventPolicy(),
        config=EventFormerConfig(**config["event_former"]),
        event_ledger=events,
        lifecycle_ledger=lifecycle,
        deployment_start_tick=clock["deployment_start_tick"],
    )
    former.poll(tick=clock["deployment_start_tick"], polling_operations=1)
    packet_payload = canonical_bytes({"motion_delta": [1, 0], "sensor_noise": 0.05})
    packet = RawPacket.create(
        sensor_id="sensor:escs-mechanics/0",
        capture_start_tick=clock["deployment_start_tick"],
        capture_end_tick=clock["packet_tick"],
        arrival_tick=clock["packet_tick"],
        clock_uncertainty=2,
        payload_bytes=packet_payload,
        sensor_scope={"modality": "fixture-vector", "channel": 0},
        source_and_provenance={"adapter": "fixture:escs-mechanics/v1"},
        transport_operations=len(packet_payload),
        adaptation_operations=3,
        detection_operations=4,
        evidence_class=EvidenceClass.SCRIPTED_MECHANICS,
    )
    formed = former.process(packet)
    hypothesis_id = EventRef(formed.hypothesis_event_ids[0])

    actor = _MechanicsActor(event_ledger=events)
    initial_actor_version = actor.state_version
    runtime = CoalitionRuntime(
        actors=(actor,),
        policy=ScriptedDispatchPolicy({}, default=(actor.descriptor.actor_id,)),
        schemas=SchemaRegistry(()),
        config=RuntimeConfig(
            mode=CandidateMode.BOUNDED_CENTRAL,
            caps=RuntimeCaps(**config["runtime_caps"]),
        ),
        ledger=lifecycle,
        event_ledger=events,
        clock_ns=lambda: 1,
    )
    chassis = EventSourcedCoalitionChassis(
        event_ledger=events,
        lifecycle_ledger=lifecycle,
        runtime=runtime,
    )
    effect = _FixtureEffect(events)
    chassis_result = chassis.execute_hypothesis(
        hypothesis_id,
        effect=effect,
        now_tick=clock["packet_tick"],
    )
    runtime.finalize(end_tick=clock["final_tick"])
    former.finalize(end_tick=clock["final_tick"])

    archive_tick = [clock["final_tick"]]
    archive_event = [None]

    def archive_accounting(charge: ArchiveCharge) -> None:
        causal_ids = (archive_event[0],) if archive_event[0] is not None else ()
        lifecycle.charge(
            owner="escs.archive-bridge",
            reason=f"archive-{charge.operation}",
            work=WorkVector(
                archival_and_erasure=charge.work_units + charge.bytes_touched,
                retained_byte_time=charge.retained_byte_ticks,
            ),
            start_tick=archive_tick[0],
            end_tick=archive_tick[0],
            causal_event_ids=causal_ids,
        )

    archive = BoundedArchive(**config["archive"], accounting_hook=archive_accounting)
    for event in events.events:
        archive_event[0] = event.event_id
        archive.append(
            event.envelope.payload(),
            canonical_bytes(event.body_payload()),
            admitted_tick=archive_tick[0],
        )
    archive_event[0] = None
    archive.compact(archive_tick[0], force=True)

    problems = list(events.verify())
    problems.extend(lifecycle.verify(event_ids=set(events.event_ids)))
    problems.extend(archive.audit())
    if not formed.admitted or len(formed.hypothesis_event_ids) != 1:
        problems.append("event former did not admit exactly one hypothesis")
    if chassis_result.status is not ChassisStatus.COMPLETED:
        problems.append(f"chassis did not complete: {chassis_result.status.value}")
    if len(effect.requests) != 1:
        problems.append("external fixture was not invoked exactly once")
    next_version = runtime.actor_state_versions[actor.descriptor.actor_id]
    if next_version == initial_actor_version:
        problems.append("authorized consequence did not replace actor state")
    if not former.finalized or former.retained_state_bytes != 0:
        problems.append("event former did not release retained policy state")
    if not runtime.finalized:
        problems.append("runtime retention ownership was not finalized")
    if any(event.evidence_class is not EvidenceClass.SCRIPTED_MECHANICS for event in events.events):
        problems.append("mechanics event escaped scripted evidence scope")

    proof: dict[str, Any] = {
        "schema": PROOF_SCHEMA,
        "complete": True,
        "all_ok": not problems,
        "problems": problems,
        "claim_scope": config["claim_scope"],
        "config_receipt": {
            "path": str(config_path.resolve().relative_to(REPO_ROOT)),
            "bytes": len(config_bytes),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "canonical_sha256": canonical_sha256(config),
        },
        "implementation_receipt": _implementation_receipt(),
        "formed_result": {
            "packet_id": formed.packet_id,
            "observation_event_id": formed.observation_event_id,
            "hypothesis_event_ids": list(formed.hypothesis_event_ids),
            "evidence_class": formed.evidence_class.value,
            "admitted": formed.admitted,
            "lifecycle_start_sequence": formed.lifecycle_start_sequence,
            "lifecycle_end_sequence": formed.lifecycle_end_sequence,
        },
        "chassis_result": chassis_result.payload(),
        "event_ledger": {
            "event_count": events.entry_count,
            "event_ids": sorted(events.event_ids),
            "event_kinds": [event.kind.value for event in events.events],
            "sha256": events.sha256,
            "verify": events.verify(),
        },
        "lifecycle_ledger": {
            "charge_count": lifecycle.entry_count,
            "total": lifecycle.total.payload(),
            "sha256": lifecycle.sha256,
            "verify": lifecycle.verify(event_ids=set(events.event_ids)),
            "archive_bridge": (
                "archival work = work_units + bytes_touched; retained byte-ticks copied exactly once"
            ),
        },
        "archive": {
            "audit": list(archive.audit()),
            "replay_authority": archive.replay_authority.value,
            "lineage_count": len(archive.lineages),
            "snapshot_count": len(archive.snapshots),
            "accounting": archive.accounting_snapshot.payload(),
        },
        "finalization": {
            "event_former_finalized": former.finalized,
            "event_former_retained_state_bytes": former.retained_state_bytes,
            "runtime_finalized": runtime.finalized,
            "runtime_last_accounted_tick": runtime.last_accounted_tick,
        },
        "nonclaims": config["nonclaims"],
    }
    proof["proof_sha256"] = canonical_sha256(proof)
    _atomic_json(out_path, proof)
    return proof


def verify_receipt(config_path: Path, proof_path: Path) -> tuple[bool, list[str]]:
    config, config_bytes = _load_config(config_path)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if not isinstance(proof, dict) or proof.get("schema") != PROOF_SCHEMA:
        return False, ["unsupported proof schema"]
    claimed = proof.get("proof_sha256")
    unsigned = dict(proof)
    unsigned.pop("proof_sha256", None)
    if claimed != canonical_sha256(unsigned):
        problems.append("proof digest mismatch")
    receipt = proof.get("config_receipt", {})
    if receipt.get("sha256") != hashlib.sha256(config_bytes).hexdigest():
        problems.append("config byte receipt mismatch")
    if receipt.get("canonical_sha256") != canonical_sha256(config):
        problems.append("config canonical receipt mismatch")
    if proof.get("implementation_receipt") != _implementation_receipt():
        problems.append("implementation receipt mismatch")
    if proof.get("claim_scope") != EvidenceClass.SCRIPTED_MECHANICS.value:
        problems.append("proof escaped scripted mechanics scope")
    if proof.get("complete") is not True or proof.get("all_ok") is not True:
        problems.append("proof is incomplete or not all-ok")
    if proof.get("problems") != []:
        problems.append("proof contains recorded problems")
    return not problems, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        ok, problems = verify_receipt(args.config, args.out)
        if not ok:
            for problem in problems:
                print(problem, file=sys.stderr)
        return 0 if ok else 1
    proof = run(args.config, args.out)
    return 0 if proof["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
