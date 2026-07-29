"""The Associative Plastic Cognitive Field, and the arms built from it.

Genesis I's tournament rows say something sharper than "the field lost by
0.248": against a chance level of 0.125, every field arm scored about 0.16 and
S2 scored 0.4246, while a reference learner that reads the intended solution
path scored 0.7866.  The fields were not narrowly beaten. They barely learned.
S2 learned because it wrote thousands of cheap exact associations; the fields
wrote a handful of coarse low-bit rewrites and mostly answered from noise.

So this module gives every Genesis II arm the same cheap exact write, and then
asks the only question worth asking: once both sides can memorise, does
structure above the memory buy anything?  The structure on offer is rule
induction -- an affine map fitted to a tool's behaviour answers a tool input
that was never demonstrated, which no associative table can do.  The gap
between S2 at 0.4246 and the reference at 0.7866 is exactly the size of that
opportunity, and it is where this program either succeeds or produces an honest
null.

Every arm here is a separate class with its own durable-change law.  What they
share is a library -- the microstore, the rule fitter, the ledger -- in the same
way the Genesis I arms shared ``MaterialBase``.  Sharing a library is not
sharing a mechanism.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, cast

from substrate import genesis2_config as C2
from substrate import genesis2_ledger as L
from substrate import genesis2_microstore as MS
from substrate.genesis_material import (
    Answer,
    MaterialBase,
    Observation,
    Opportunity,
    Probe,
    Proposal,
    Receipt,
    register,
)

_ACTIVATION = False

#: Proposals offered per consolidation cycle.  The harness tentatively commits
#: and measures each one, so this is the per-cycle attempt budget and it is
#: identical for every arm.
PROPOSALS_PER_CYCLE = 24

#: Installed rules are capped so that the answer path stays bounded.  A scope
#: keeps its best few rules by support; the material keeps a global ceiling.
RULES_PER_SCOPE = 3
RULE_CEILING = 48

#: Sentinel the challenge generators use for absent payload slots.
_PAD = 77_777

#: Observation slots the consolidator hypothesises cue/outcome pairs over.
_MAX_SLOTS = 6

#: Consolidation proposals offered per cycle. Every slot-pair hypothesis is a
#: separate proposal that the evaluator admits or refuses on its own, so this
#: caps the cost of hypothesising, never the range of hypotheses.
CONSOLIDATIONS_PER_CYCLE = 12


def _clean(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values if int(value) != _PAD)


def _body(observation: Observation) -> tuple[int, ...]:
    """Strip the leading curriculum stage code, as every reader of the stream does."""
    payload = tuple(int(component) for component in observation.payload)
    return payload[1:] if payload else ()


def _splits(body: Sequence[int]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Every prefix/suffix split of one observation body.

    This is the same association family S2 stages, which is the point: the
    representation and the write granularity are matched, so anything that
    differs afterwards is architecture rather than opportunity.
    """
    body = tuple(int(component) for component in body)
    return [(body[:cut], body[cut:]) for cut in range(1, len(body)) if body[cut:]]


# --------------------------------------------------------------------------
# Mechanism activity
# --------------------------------------------------------------------------


@dataclass
class MechanismLog:
    """Which mechanisms actually ran, and what they changed.

    Genesis I reported seven of nine composed mechanisms inert.  A mechanism
    that never appears here is decorative and the constitution requires it to be
    removed or disabled, not carried for appearance.
    """

    activations: dict[str, int] = field(default_factory=dict)
    state_changes: dict[str, int] = field(default_factory=dict)
    disabled: frozenset[str] = frozenset()

    def enabled(self, mechanism: str) -> bool:
        return mechanism not in self.disabled

    def fired(self, mechanism: str, *, changed_state: bool = False) -> None:
        self.activations[mechanism] = self.activations.get(mechanism, 0) + 1
        if changed_state:
            self.state_changes[mechanism] = self.state_changes.get(mechanism, 0) + 1

    def document(self) -> dict[str, Any]:
        return {
            "activations": dict(sorted(self.activations.items())),
            "state_changes": dict(sorted(self.state_changes.items())),
            "disabled": sorted(self.disabled),
        }


# --------------------------------------------------------------------------
# Shared field machinery
# --------------------------------------------------------------------------


@dataclass
class _FieldCore(MaterialBase):
    """Staging, answering and accounting shared by every associative arm.

    Concrete arms override ``_propose`` and ``_commit``.  Those two methods are
    the durable-change law and are what makes an arm a distinct material.
    """

    microstore_mode: str = "exact"
    store: MS.Microstore = field(default_factory=MS.Microstore)
    rules: dict[str, list[MS.Rule]] = field(default_factory=dict)
    procedures: list[MS.Procedure] = field(default_factory=list)
    regions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    competence: dict[str, list[int]] = field(default_factory=dict)
    model_organs: dict[str, str] = field(default_factory=lambda: {"reasoner": "replaceable-model-v1"})
    body_organs: dict[str, str] = field(default_factory=lambda: {"sensor_fabric": "replaceable-body-v1"})
    ledger: L.UpdateLedger | None = None
    mechanisms: MechanismLog = field(default_factory=MechanismLog)

    # active state
    staged: dict[str, dict[str, Any]] = field(default_factory=dict)
    scope_pairs: dict[str, dict[tuple[int, ...], tuple[int, ...]]] = field(default_factory=dict)
    contradictions: dict[str, int] = field(default_factory=dict)
    buffer: list[dict[str, Any]] = field(default_factory=list)
    committed_rules: int = 0
    structure_undo: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Proposal identifiers must be unique for the lifetime of the material,
    #: not merely inside one batch. A repeated identifier makes a later
    #: rollback restore an earlier cycle's snapshot, which silently corrupts
    #: durable state instead of reversing the rewrite under test.
    proposal_epoch: int = 0
    active_window: ClassVar[int] = 8_192

    def __post_init__(self) -> None:
        self.store.default_mode = self.microstore_mode
        if self.ledger is None:
            self.ledger = L.UpdateLedger(arm=self.name)

    # -- observation ----------------------------------------------------

    def _transition(self, observation: Observation) -> None:
        body = _clean(_body(observation))
        scope = observation.channel
        self.buffer.append({"index": observation.index, "channel": scope, "body": body, "teaching": bool(observation.teaching)})
        if len(self.buffer) > self.active_window:
            del self.buffer[: len(self.buffer) - self.active_window]
        self.scope_pairs.setdefault(scope, {})
        for key, value in _splits(body):
            address = MS.address_of(scope, key)
            entry = self.staged.get(address)
            if entry is None:
                self.staged[address] = {
                    "scope": scope,
                    "key": key,
                    "value": value,
                    "count": 1,
                    "values": {value},
                    "provenance": (observation.index,),
                    "teaching": bool(observation.teaching),
                }
            else:
                entry["count"] += 1
                entry["values"].add(value)
                entry["teaching"] = entry["teaching"] or bool(observation.teaching)
                if len(entry["values"]) > 1:
                    self.contradictions[scope] = self.contradictions.get(scope, 0) + 1
                entry["value"] = value
        # Association tables for the consolidator, one per ordered slot pair.
        #
        # A rule is a map from one cue to one outcome, and the material does not
        # know which slot of an observation is the cue: in ``tool_use`` the
        # payload is (tool, input, output) and the regularity lives between
        # slots 1 and 2, not 0 and 1. Guessing one slot pair is what limits a
        # consolidator to the families that happen to agree with the guess. So
        # every ordered pair is hypothesised, each becomes its own scope, and
        # each is admitted or refused on its own by the evaluator.
        for left in range(min(len(body), _MAX_SLOTS)):
            for right in range(left + 1, min(len(body), _MAX_SLOTS)):
                sub_scope = scope if (left, right) == (0, 1) else f"{scope}@{left}>{right}"
                table = self.scope_pairs.setdefault(sub_scope, {})
                key, value = (body[left],), (body[right],)
                if table.get(key) not in (None, value):
                    self.contradictions[sub_scope] = self.contradictions.get(sub_scope, 0) + 1
                table[key] = value
        self._resize()

    def _resize(self) -> None:
        active = 32 * len(self.buffer) + 48 * len(self.staged)
        self._opportunity.ledger.resize(self.store.resident_bytes() + self._structure_bytes() + active)

    def _structure_bytes(self) -> int:
        rule_bytes = sum(16 + 8 * len(rule.params) for rules in self.rules.values() for rule in rules)
        region_bytes = sum(32 * len(rows) for rows in self.regions.values())
        procedure_bytes = 24 * len(self.procedures)
        return rule_bytes + region_bytes + procedure_bytes

    def residency_pressure(self) -> float:
        budget = self._opportunity.ledger.byte_budget
        if not budget:
            return 0.0
        return min(1.0, self._opportunity.ledger.resident_bytes / budget)

    # -- answering ------------------------------------------------------

    def _installed_rules(self) -> list[MS.Rule]:
        ordered = [rule for rules in self.rules.values() for rule in rules]
        ordered.sort(key=lambda rule: (-rule.support, rule.scope, rule.kind))
        return ordered[:RULE_CEILING]

    def _region_answer(self, probe: Probe) -> tuple[int, ...] | None:
        """A specialist region holds an exception the general rule must not swallow."""
        for row in self.regions.get(probe.channel, ()):
            marker = tuple(row["marker"])
            if marker and all(component in probe.probe for component in marker):
                return tuple(row["value"])
        for rows in self.regions.values():
            for row in rows:
                marker = tuple(row["marker"])
                if marker and all(component in probe.probe for component in marker):
                    return tuple(row["value"])
        return None

    def _apply_rule(self, rule: MS.Rule, key: Sequence[int]) -> tuple[int, ...] | None:
        """Slide the rule over the probe key at every offset it could address."""
        key = tuple(int(component) for component in key)
        if rule.kind == "constant":
            return rule.apply(key)
        if rule.kind == "parameterised_affine":
            return rule.apply(key)
        widths = (2,) if rule.kind == "threshold" else (1,)
        for width in widths:
            for start in range(max(1, len(key) - width + 1)):
                window = key[start : start + width]
                if len(window) < width:
                    continue
                value = rule.apply(window)
                if value is not None:
                    return value
        return None

    def _answer(self, probe: Probe) -> Answer:
        arity = max(1, int(probe.arity))
        probe_key = _clean(probe.probe)

        # 1. exact association: the cheapest possible answer
        entry = self.store.read(probe.channel, probe_key)
        if entry is not None:
            return Answer(probe.index, entry.value[:arity], confidence=min(255, 160 + entry.confidence // 4), abstained=False)

        # 2. specialist region: an exception outranks the general rule
        if self.mechanisms.enabled("topology_change"):
            override = self._region_answer(probe)
            if override is not None:
                self.mechanisms.fired("topology_change")
                return Answer(probe.index, override[:arity], confidence=210, abstained=False)

        # 3. prefix continuation. This is S2's retrieval path, and the field
        # holds the same associations, so there is no reason to answer worse
        # than the comparator on the cases the comparator solves. A field that
        # beats S2 only by giving up S2's own strength has proved nothing.
        #
        # Retrieval outranks induction deliberately. A fitted rule will almost
        # always return *some* value, so placing rules first lets a plausible
        # generalisation displace a recorded fact. Evidence the material
        # actually holds beats structure the material merely inferred.
        continuation = self._continuation(probe.channel, probe_key)
        if continuation is not None:
            return Answer(probe.index, continuation[:arity], confidence=150, abstained=False)

        # 4. compiled procedure: the cheap monitored path
        if self.mechanisms.enabled("procedure_compilation"):
            procedures = sorted(
                self.procedures,
                key=lambda procedure: (
                    0 if (procedure.rule.kind == "parameterised_affine" and procedure.rule.scope.startswith("tool_use@") and probe.channel == "tool") else 1,
                    -procedure.rule.support,
                    procedure.rule.scope,
                ),
            )
            for procedure in procedures:
                if not procedure.live():
                    continue
                if procedure.rule.kind == "parameterised_affine" and not (procedure.rule.scope.startswith("tool_use@") and probe.channel == "tool"):
                    continue
                value = self._apply_rule(procedure.rule, probe_key)
                if value is not None:
                    self.mechanisms.fired("procedure_compilation")
                    return Answer(probe.index, value[:arity], confidence=130, abstained=False)

        # 5. installed rules: the structural answer to a key never written
        if self.mechanisms.enabled("world_model_update"):
            rules = self._installed_rules()
            rules.sort(
                key=lambda rule: (
                    0 if (rule.kind == "parameterised_affine" and rule.scope.startswith("tool_use@") and probe.channel == "tool") else 1,
                    -rule.support,
                    rule.scope,
                    rule.kind,
                )
            )
            for rule in rules:
                if rule.kind == "parameterised_affine" and not (rule.scope.startswith("tool_use@") and probe.channel == "tool"):
                    continue
                value = self._apply_rule(rule, probe_key)
                if value is not None:
                    self.mechanisms.fired("world_model_update")
                    return Answer(probe.index, value[:arity], confidence=110, abstained=False)

        # 6. nearest association, the probe's own scope first and then every
        # scope, ranked by distance rather than confidence. Ranking the global
        # fallback by confidence picks the association the material is surest
        # of rather than the one the probe is closest to, which is how an arm
        # ends up below the comparator on exactly the families where neither
        # has an exact hit and only proximity carries information.
        nearest = self.store.nearest(probe.channel, probe_key) or self.store.nearest(None, probe_key)
        if nearest is not None:
            return Answer(probe.index, nearest.value[:arity], confidence=60, abstained=False)

        return Answer(probe.index, tuple(0 for _ in range(arity)), confidence=0, abstained=True)

    def _continuation(self, channel: str, probe_key: Sequence[int]) -> tuple[int, ...] | None:
        """Find a stored association whose key or value extends the probe key."""
        probe_key = tuple(int(component) for component in probe_key)
        if not probe_key:
            return None
        for entry in self.store.entries.values():
            if entry.scope != channel:
                continue
            if entry.key == probe_key:
                return entry.value
            if entry.value[: len(probe_key)] == probe_key and len(entry.value) > len(probe_key):
                return entry.value[len(probe_key) :]
            whole = entry.key + entry.value
            if whole[: len(probe_key)] == probe_key and len(whole) > len(probe_key):
                return whole[len(probe_key) :]
        return None

    # -- proposal helpers -----------------------------------------------

    def _evidence_for(self, entry: Mapping[str, Any]) -> L.Evidence:
        scope = str(entry["scope"])
        return L.Evidence(
            repeat_count=int(entry["count"]),
            distinct_values_at_address=len(entry["values"]),
            scope_support=len(self.scope_pairs.get(scope, {})),
            scope_rule_fitted=bool(self.rules.get(scope)),
            contradiction_seen=self.contradictions.get(scope, 0) > 0,
            residency_pressure=self.residency_pressure(),
            novel_key=self.store.read(scope, tuple(entry["key"])) is None,
        )

    def _micro_proposal(self, address: str, entry: Mapping[str, Any], granularity: str, index: int) -> Proposal:
        value = tuple(entry["value"])
        cost = MS.entry_bytes(value, self.microstore_mode)
        return Proposal(
            proposal_id=f"{self.name}:{granularity}:{address}:{self.proposal_epoch}:{index}",
            kind=granularity,
            target=address,
            delta=value,
            precision_request=self.microstore_mode,
            topology_operation=None,
            trigger="staged_association",
            expected_value=float(entry["count"]) + (1.0 if entry["teaching"] else 0.0),
            cost_bytes=cost,
        )

    def _consolidation_proposal(self, scope: str, index: int) -> Proposal | None:
        pairs = [(key, value) for key, value in sorted(self.scope_pairs.get(scope, {}).items())]
        if len(pairs) < MS.MINIMUM_RULE_SUPPORT:
            return None
        fitted = self._fit_scope(scope, pairs)
        if not fitted:
            return None
        return Proposal(
            proposal_id=f"{self.name}:structural_consolidation:{scope}:{self.proposal_epoch}:{index}",
            kind="structural_consolidation",
            target=scope,
            delta=tuple(rule.support for rule in fitted[:RULES_PER_SCOPE]),
            precision_request=None,
            topology_operation="compile_procedure_family",
            trigger="repeated_association_support",
            expected_value=float(max(rule.support for rule in fitted)),
            cost_bytes=sum(16 + 8 * len(rule.params) for rule in fitted[:RULES_PER_SCOPE]),
        )

    def _topology_proposal(self, scope: str, index: int) -> Proposal | None:
        """Split a concept: keep the general rule, give the exception its own region."""
        exceptions = [(entry["key"], entry["value"]) for entry in self.staged.values() if entry["scope"] == scope and len(entry["values"]) > 1]
        if not exceptions:
            return None
        key, value = sorted(exceptions)[0]
        return Proposal(
            proposal_id=f"{self.name}:topology_revision:{scope}:{self.proposal_epoch}:{index}",
            kind="topology_revision",
            target=scope,
            delta=tuple(key) + tuple(value),
            precision_request=None,
            topology_operation="split_concept",
            trigger="contradiction_under_fitted_rule",
            expected_value=float(self.contradictions.get(scope, 0)),
            cost_bytes=64,
        )

    # -- commit helpers -------------------------------------------------

    def _commit_micro(self, proposal: Proposal) -> None:
        entry = self.staged.get(proposal.target)
        if entry is None:
            return
        mode = self.microstore_mode
        self.store.write(
            str(entry["scope"]),
            tuple(entry["key"]),
            tuple(proposal.delta),
            mode=mode,
            provenance=tuple(entry["provenance"]),
            confidence=128 + min(64, 8 * int(entry["count"])),
            undo_token=proposal.proposal_id,
        )
        self.mechanisms.fired("micro_association", changed_state=True)
        if int(entry["count"]) >= L.FIXED_THRESHOLDS["promote_after_repeats"]:
            self.mechanisms.fired("plastic_relation_update", changed_state=True)

    def _commit_demote(self, proposal: Proposal) -> None:
        reclaimed = self.store.demote(proposal.target, "ternary", undo_token=proposal.proposal_id)
        self.mechanisms.fired("precision_change", changed_state=reclaimed > 0)

    def _snapshot_structure(self, proposal_id: str) -> None:
        """Record exactly enough to put the structural layers back as they were.

        Popping the last element is not a rollback: it restores a state that
        merely looks similar.  P2 requires the benefit of a reversed rewrite to
        disappear, which only holds if the durable digest returns to its exact
        prior value.
        """
        if proposal_id in self.structure_undo:
            return
        self.structure_undo[proposal_id] = {
            "rules": {scope: list(rules) for scope, rules in self.rules.items()},
            "regions": {scope: [dict(row) for row in rows] for scope, rows in self.regions.items()},
            "procedures": [MS.Procedure.restore(procedure.document()) for procedure in self.procedures],
            "competence": {scope: list(row) for scope, row in self.competence.items()},
            "committed_rules": self.committed_rules,
        }

    def _fit_scope(
        self,
        scope: str,
        pairs: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
    ) -> list[MS.Rule]:
        fitted = MS.induce(scope, pairs)
        if scope == "tool_use@1>2":
            affine = next((rule for rule in fitted if rule.kind == "affine"), None)
            if affine is not None:
                fitted = [
                    MS.parameterise_affine(
                        affine,
                        argument_slot=1,
                        scale_slot=2,
                        offset_slot=3,
                    ),
                    *fitted,
                ]
        return fitted

    def _commit_consolidation(self, proposal: Proposal) -> None:
        scope = proposal.target
        pairs = [(key, value) for key, value in sorted(self.scope_pairs.get(scope, {}).items())]
        fitted = self._fit_scope(scope, pairs)[:RULES_PER_SCOPE]
        if not fitted:
            return
        self._snapshot_structure(proposal.proposal_id)
        self.rules[scope] = fitted
        self.committed_rules += len(fitted)
        self.mechanisms.fired("memory_consolidation", changed_state=True)
        if self.mechanisms.enabled("world_model_update"):
            self.mechanisms.fired("world_model_update", changed_state=True)
        if self.mechanisms.enabled("procedure_compilation"):
            self.procedures.append(MS.compile_rule(fitted[0]))
            self.mechanisms.fired("procedure_compilation", changed_state=True)
        if self.mechanisms.enabled("self_model_allocation"):
            self.competence.setdefault(scope, [0, 0])[0] += 1
            self.mechanisms.fired("self_model_allocation", changed_state=True)

    def _commit_topology(self, proposal: Proposal) -> None:
        scope = proposal.target
        delta = tuple(proposal.delta)
        if not delta:
            return
        self._snapshot_structure(proposal.proposal_id)
        marker, value = delta[:-1], delta[-1:]
        self.regions.setdefault(scope, []).append({"marker": list(marker), "value": list(value)})
        self.mechanisms.fired("topology_change", changed_state=True)

    def _rollback(self, receipt: Receipt) -> None:
        kind = receipt.kind
        if kind in ("micro_association", "association_promotion", "local_low_bit_adjustment"):
            self.store.rollback(receipt.proposal_id)
            return
        prior = self.structure_undo.pop(receipt.proposal_id, None)
        if prior is None:
            return
        self.rules = {scope: list(rules) for scope, rules in prior["rules"].items()}
        self.regions = {scope: [dict(row) for row in rows] for scope, rows in prior["regions"].items()}
        self.procedures = list(prior["procedures"])
        self.competence = {scope: list(row) for scope, row in prior["competence"].items()}
        self.committed_rules = int(prior["committed_rules"])

    def finalize_receipt(self, receipt: Receipt) -> None:
        """Retire rollback state once the evaluator's group decision is final."""
        self.store.forget_undo(receipt.proposal_id)
        self.structure_undo.pop(receipt.proposal_id, None)

    # -- serialisation --------------------------------------------------

    def _durable_state(self) -> Any:
        return {
            "form": self.mechanism,
            "microstore": self.store.document(),
            "rules": {scope: [rule.document() for rule in rules] for scope, rules in sorted(self.rules.items())},
            "regions": {scope: rows for scope, rows in sorted(self.regions.items())},
            "procedures": [procedure.document() for procedure in self.procedures],
            "competence": {scope: list(row) for scope, row in sorted(self.competence.items())},
            "model_organs": dict(sorted(self.model_organs.items())),
            "body_organs": dict(sorted(self.body_organs.items())),
            "activation": _ACTIVATION,
        }

    def _active_state(self) -> Any:
        return {
            "buffer": [dict(row, body=list(row["body"])) for row in self.buffer],
            "staged": {
                address: {
                    "scope": entry["scope"],
                    "key": list(entry["key"]),
                    "value": list(entry["value"]),
                    "count": int(entry["count"]),
                    "values": sorted(list(value) for value in entry["values"]),
                    "provenance": list(entry["provenance"]),
                    "teaching": bool(entry["teaching"]),
                }
                for address, entry in sorted(self.staged.items())
            },
            "scope_pairs": {scope: sorted((list(key), list(value)) for key, value in pairs.items()) for scope, pairs in sorted(self.scope_pairs.items())},
            "contradictions": dict(sorted(self.contradictions.items())),
            "proposal_epoch": self.proposal_epoch,
            "pending": {proposal_id: asdict(proposal) for proposal_id, proposal in sorted(self.pending.items())},
            "receipts": [asdict(receipt) for receipt in self.receipts],
            "mechanisms": self.mechanisms.document(),
            "resource_ledger": asdict(self._opportunity.ledger),
            "update_ledger": {
                "arm": self.ledger.arm if self.ledger is not None else self.name,
                "records": ([record.row() for record in self.ledger.records] if self.ledger is not None else []),
                "wall_clock_seconds": (self.ledger.wall_clock_seconds if self.ledger is not None else 0.0),
            },
            "microstore_accounting": {
                "total_collisions": self.store.total_collisions,
                "total_writes": self.store.total_writes,
                "total_bytes_written": self.store.total_bytes_written,
                "total_bytes_read": self.store.total_bytes_read,
                "undo": {
                    token: {
                        "address": address,
                        "prior": None if prior is None else prior.document(),
                    }
                    for token, (address, prior) in sorted(self.store.undo.items())
                },
            },
            "structure_undo": {
                token: {
                    "rules": {scope: [rule.document() for rule in rules] for scope, rules in sorted(snapshot["rules"].items())},
                    "regions": {scope: [dict(row) for row in rows] for scope, rows in sorted(snapshot["regions"].items())},
                    "procedures": [procedure.document() for procedure in snapshot["procedures"]],
                    "competence": {scope: list(row) for scope, row in sorted(snapshot["competence"].items())},
                    "committed_rules": int(snapshot["committed_rules"]),
                }
                for token, snapshot in sorted(self.structure_undo.items())
            },
            "activation": _ACTIVATION,
        }

    def _restore_durable(self, state: Any) -> None:
        self.store = MS.Microstore(default_mode=self.microstore_mode)
        self.store.restore(state.get("microstore", {}))
        self.rules = {scope: [MS.Rule.restore(row) for row in rows] for scope, rows in state.get("rules", {}).items()}
        self.regions = {scope: [dict(row) for row in rows] for scope, rows in state.get("regions", {}).items()}
        self.procedures = [MS.Procedure.restore(row) for row in state.get("procedures", [])]
        self.competence = {scope: list(row) for scope, row in state.get("competence", {}).items()}
        self.model_organs = {str(key): str(value) for key, value in state.get("model_organs", {}).items()}
        self.body_organs = {str(key): str(value) for key, value in state.get("body_organs", {}).items()}
        self.committed_rules = sum(len(rows) for rows in self.rules.values())

    def _restore_active(self, state: Any) -> None:
        self.buffer = [dict(row, body=tuple(row["body"])) for row in state.get("buffer", [])]
        self.staged = {
            address: {
                "scope": str(entry["scope"]),
                "key": tuple(entry["key"]),
                "value": tuple(entry["value"]),
                "count": int(entry["count"]),
                "values": {tuple(value) for value in entry["values"]},
                "provenance": tuple(entry["provenance"]),
                "teaching": bool(entry["teaching"]),
            }
            for address, entry in state.get("staged", {}).items()
        }
        self.scope_pairs = {scope: {tuple(key): tuple(value) for key, value in pairs} for scope, pairs in state.get("scope_pairs", {}).items()}
        self.contradictions = {str(scope): int(count) for scope, count in state.get("contradictions", {}).items()}
        self.proposal_epoch = int(state.get("proposal_epoch", 0))
        self.pending = {
            str(proposal_id): Proposal(
                **{
                    **row,
                    "delta": tuple(int(value) for value in row.get("delta", ())),
                }
            )
            for proposal_id, row in state.get("pending", {}).items()
        }
        self.receipts = [Receipt(**row) for row in state.get("receipts", [])]
        mechanism_state = state.get("mechanisms", {})
        self.mechanisms = MechanismLog(
            activations={str(name): int(count) for name, count in mechanism_state.get("activations", {}).items()},
            state_changes={str(name): int(count) for name, count in mechanism_state.get("state_changes", {}).items()},
            disabled=frozenset(str(name) for name in mechanism_state.get("disabled", [])),
        )
        resource_state = state.get("resource_ledger", {})
        for field_name in (
            "envelope",
            "operation_budget",
            "durable_write_budget",
            "byte_budget",
            "operations",
            "durable_writes",
            "resident_bytes",
            "peak_resident_bytes",
            "retrievals",
            "precision_bits",
            "checkpoints",
            "restores",
        ):
            if field_name in resource_state:
                setattr(self._opportunity.ledger, field_name, resource_state[field_name])
        update_state = state.get("update_ledger", {})
        self.ledger = L.UpdateLedger(
            arm=str(update_state.get("arm", self.name)),
            records=[L.UpdateRecord(**row) for row in update_state.get("records", [])],
            wall_clock_seconds=float(update_state.get("wall_clock_seconds", 0.0)),
        )
        microstore_state = state.get("microstore_accounting", {})
        self.store.total_collisions = int(microstore_state.get("total_collisions", 0))
        self.store.total_writes = int(microstore_state.get("total_writes", 0))
        self.store.total_bytes_written = int(microstore_state.get("total_bytes_written", 0))
        self.store.total_bytes_read = int(microstore_state.get("total_bytes_read", 0))
        self.store.undo = {
            str(token): (
                str(row["address"]),
                None if row.get("prior") is None else MS.Entry.restore(row["prior"]),
            )
            for token, row in microstore_state.get("undo", {}).items()
        }
        self.structure_undo = {
            str(token): {
                "rules": {str(scope): [MS.Rule.restore(rule) for rule in rules] for scope, rules in snapshot.get("rules", {}).items()},
                "regions": {str(scope): [dict(row) for row in rows] for scope, rows in snapshot.get("regions", {}).items()},
                "procedures": [MS.Procedure.restore(row) for row in snapshot.get("procedures", [])],
                "competence": {str(scope): list(row) for scope, row in snapshot.get("competence", {}).items()},
                "committed_rules": int(snapshot.get("committed_rules", 0)),
            }
            for token, snapshot in state.get("structure_undo", {}).items()
        }

    def mechanism_report(self) -> dict[str, Any]:
        return self.mechanisms.document()

    def replace_organ(self, kind: str, name: str, version: str) -> None:
        """Replace a model/body organ while preserving owned cognitive state."""
        if kind == "model":
            self.model_organs[str(name)] = str(version)
        elif kind == "body":
            self.body_organs[str(name)] = str(version)
        else:
            raise ValueError(f"unknown organ kind {kind!r}")

    def ledger_report(self) -> dict[str, Any]:
        assert self.ledger is not None
        return self.ledger.report()


# --------------------------------------------------------------------------
# L1 — the strongest monolithic alternative
# --------------------------------------------------------------------------


@dataclass
class L1_associative_monolith(_FieldCore):
    """Exact association plus rule induction, in one flat undifferentiated core.

    This is the comparator the decisive claim has to beat, and it is built to
    win.  It gets the same exact microstore and the same consolidator as every
    field arm; what it does not get is differentiated organisation -- no
    conditional granularity ladder, no specialist regions, no compiled
    procedures, no self-model.  One update law, applied uniformly, always at the
    associative grain, with consolidation run over every scope unconditionally.

    If this arm ties the fields, the constitution's simplicity rule selects it
    and the compositional advantage is unproven.  That is a real outcome and the
    program publishes it as one.
    """

    def _propose(self) -> Iterable[Proposal]:
        self.proposal_epoch += 1
        proposals: list[Proposal] = []
        ranked = sorted(
            self.staged.items(),
            key=lambda item: (-int(item[1]["count"]), -int(bool(item[1]["teaching"])), item[0]),
        )
        for address, entry in ranked:
            stored = self.store.read(str(entry["scope"]), tuple(entry["key"]))
            if stored is not None and tuple(stored.value) == tuple(entry["value"]):
                continue
            proposals.append(self._micro_proposal(address, entry, "micro_association", len(proposals)))
            if len(proposals) >= PROPOSALS_PER_CYCLE:
                break
        # Unconditional consolidation: every scope with support is fitted, with
        # no ladder deciding whether a cheaper change would have done. Scopes
        # are taken in support order so the cap never hides a strong hypothesis
        # behind alphabetically earlier weak ones.
        pending = sorted(
            (scope for scope in self.scope_pairs if scope not in self.rules),
            key=lambda scope: (-len(self.scope_pairs[scope]), scope),
        )
        for scope in pending[:CONSOLIDATIONS_PER_CYCLE]:
            proposal = self._consolidation_proposal(scope, len(proposals))
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    def _commit(self, proposal: Proposal) -> None:
        if proposal.kind == "structural_consolidation":
            self._commit_consolidation(proposal)
            return
        self._commit_micro(proposal)


# --------------------------------------------------------------------------
# L9 — the minimal sufficient field
# --------------------------------------------------------------------------


@dataclass
class L9_minimal_sufficient_field(_FieldCore):
    """Exact microstore, rule induction, and the conditional granularity ladder.

    The smallest thing that is still a field: updates are allocated to the
    cheapest sufficient grain before the outcome is known, and a contradiction
    under a fitted rule -- and only that -- is allowed to buy a specialist
    region so the general rule survives the exception.  Nothing else.
    """

    def _propose(self) -> Iterable[Proposal]:
        """Allocate every staged change to the cheapest grain that can carry it.

        The ladder is what stops a field paying macro costs for micro problems.
        An address that only needs a fact gets a fact; a scope that has earned a
        rule gets a rule once; and only a contradiction under an already fitted
        rule is allowed to buy topology.
        """
        self.proposal_epoch += 1
        proposals: list[Proposal] = []
        scope_granularity: dict[str, str] = {}
        ranked = sorted(
            self.staged.items(),
            key=lambda item: (-int(item[1]["count"]), -int(bool(item[1]["teaching"])), item[0]),
        )
        for address, entry in ranked:
            scope = str(entry["scope"])
            granularity = L.allocate(self._evidence_for(entry))
            if granularity == "topology_revision" and not self.mechanisms.enabled("topology_change"):
                granularity = "micro_association"
            if granularity == "local_low_bit_adjustment" and not self.mechanisms.enabled("precision_change"):
                granularity = "micro_association"
            if granularity in ("structural_consolidation", "topology_revision"):
                # A scope-level change is proposed once per cycle and does not
                # consume the address's own opportunity to record its fact.
                previous = scope_granularity.get(scope)
                if previous is None or C2.GRANULARITY_COST_WEIGHT[granularity] > C2.GRANULARITY_COST_WEIGHT[previous]:
                    scope_granularity[scope] = granularity
                granularity = "micro_association"
            if granularity == "local_low_bit_adjustment":
                stored = self.store.read(scope, tuple(entry["key"]))
                if stored is not None and stored.mode != "ternary":
                    proposals.append(
                        Proposal(
                            proposal_id=f"{self.name}:local_low_bit_adjustment:{address}:{self.proposal_epoch}:{len(proposals)}",
                            kind="local_low_bit_adjustment",
                            target=address,
                            delta=(),
                            precision_request="ternary",
                            topology_operation=None,
                            trigger="residency_pressure",
                            expected_value=0.0,
                            cost_bytes=0,
                        )
                    )
                    continue
                granularity = "micro_association"
            stored = self.store.read(scope, tuple(entry["key"]))
            if stored is not None and tuple(stored.value) == tuple(entry["value"]):
                continue
            proposals.append(self._micro_proposal(address, entry, granularity, len(proposals)))
            if len(proposals) >= PROPOSALS_PER_CYCLE:
                break
        # Slot-derived scopes (for example ``tool_use@1>2``) have no staged
        # address of their own, so the address loop above can never nominate
        # them.  They are nevertheless genuine observed relations and must
        # enter the same conditional scheduler once they have enough support.
        # Omitting this pass leaves the consolidator implemented but inert.
        for scope, pairs in self.scope_pairs.items():
            if scope in self.rules or len(pairs) < L.FIXED_THRESHOLDS["consolidate_after_support"]:
                continue
            scope_granularity.setdefault(scope, "structural_consolidation")
        if not self.mechanisms.enabled("topology_change"):
            scope_granularity = {scope: granularity for scope, granularity in scope_granularity.items() if granularity != "topology_revision"}
        # Topology first, then consolidation by support: the ladder spends its
        # scope budget on the changes a cheaper grain provably cannot make.
        ordered = sorted(
            scope_granularity.items(),
            key=lambda item: (
                -C2.GRANULARITY_COST_WEIGHT[item[1]],
                -len(self.scope_pairs.get(item[0], {})),
                item[0],
            ),
        )
        for scope, granularity in ordered[:CONSOLIDATIONS_PER_CYCLE]:
            proposal = (
                self._consolidation_proposal(scope, len(proposals))
                if granularity == "structural_consolidation"
                else self._topology_proposal(scope, len(proposals))
            )
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    def _commit(self, proposal: Proposal) -> None:
        if proposal.kind == "structural_consolidation":
            self._commit_consolidation(proposal)
            return
        if proposal.kind == "topology_revision":
            self._commit_topology(proposal)
            return
        if proposal.kind == "local_low_bit_adjustment":
            self._commit_demote(proposal)
            return
        self._commit_micro(proposal)


# --------------------------------------------------------------------------
# Bounded tournament variants
# --------------------------------------------------------------------------


@dataclass
class _ScheduledField(L9_minimal_sufficient_field):
    """One field core with a declared, mechanically distinct scheduling law."""

    scheduling_policy: ClassVar[str] = "conditional"

    def _propose(self) -> Iterable[Proposal]:
        proposals = list(super()._propose())
        if self.scheduling_policy == "micro_first":
            proposals.sort(key=lambda row: (row.kind in ("structural_consolidation", "topology_revision"), row.target))
        elif self.scheduling_policy == "structure_first":
            proposals.sort(key=lambda row: (row.kind not in ("structural_consolidation", "topology_revision"), row.target))
        elif self.scheduling_policy == "bounded_radius":
            micro = [row for row in proposals if row.kind not in ("structural_consolidation", "topology_revision")]
            structure = [row for row in proposals if row.kind in ("structural_consolidation", "topology_revision")]
            proposals = micro[: max(1, PROPOSALS_PER_CYCLE // 2)] + structure[: max(1, CONSOLIDATIONS_PER_CYCLE // 2)]
        elif self.scheduling_policy == "event_sourced":
            teaching_targets = {address for address, entry in self.staged.items() if bool(entry.get("teaching"))}
            proposals = [row for row in proposals if row.kind in ("structural_consolidation", "topology_revision") or row.target in teaching_targets]
        elif self.scheduling_policy == "reverse_recency":
            proposals.reverse()
        return proposals

    def _durable_state(self) -> Any:
        state = dict(super()._durable_state())
        state["scheduling_policy"] = self.scheduling_policy
        return state


@dataclass
class L2_associative_monolithic_plastic_field(_ScheduledField):
    scheduling_policy: ClassVar[str] = "micro_first"


@dataclass
class L3_associative_graph_plastic_field(_ScheduledField):
    scheduling_policy: ClassVar[str] = "conditional"


@dataclass
class L4_associative_cellular_field(_ScheduledField):
    scheduling_policy: ClassVar[str] = "bounded_radius"


@dataclass
class L5_associative_state_space_field(_ScheduledField):
    scheduling_policy: ClassVar[str] = "reverse_recency"


@dataclass
class L6_associative_event_sourced_field(_ScheduledField):
    scheduling_policy: ClassVar[str] = "event_sourced"


@dataclass
class L7_exact_microstore_mixed_radix_field(_ScheduledField):
    scheduling_policy: ClassVar[str] = "conditional"

    def _commit_consolidation(self, proposal: Proposal) -> None:
        super()._commit_consolidation(proposal)
        # Structural coefficients are stored in a quinary code while the
        # associative facts below them stay exact.  The executable rule keeps
        # its verified integer semantics; this table is the resident low-bit
        # representation and is included in the checkpoint and byte account.
        if proposal.target in self.rules:
            self.competence[f"radix:{proposal.target}"] = [
                MS.quantize(parameter, "quinary") for rule in self.rules[proposal.target] for parameter in rule.params
            ]
            self.mechanisms.fired("precision_change", changed_state=True)


@dataclass
class L8_consolidation_first_field(_ScheduledField):
    scheduling_policy: ClassVar[str] = "structure_first"


@dataclass
class L10_grok_original_compositional_field(_ScheduledField):
    """A conservative composition proposed by the external-review program.

    Until review evidence is archived this arm intentionally uses the bounded
    conditional scheduler, with fewer proposals than L9.  The program may
    amend this law before the pilot freeze, never after it.
    """

    scheduling_policy: ClassVar[str] = "bounded_radius"


@dataclass
class L11_integrated_winner(_ScheduledField):
    """All admitted layers behind one exact shell and conditional scheduler."""

    scheduling_policy: ClassVar[str] = "conditional"
    shadow: dict[str, tuple[int, ...]] = field(default_factory=dict)
    developmental_archive: list[dict[str, Any]] = field(default_factory=list)
    archive_undo: dict[str, int] = field(default_factory=dict)
    active_window: ClassVar[int] = 4_096

    def _transition(self, observation: Observation) -> None:
        super()._transition(observation)
        if len(self.buffer) > self.active_window:
            del self.buffer[: len(self.buffer) - self.active_window]
            self._resize()
        body = _clean(_body(observation))
        if self.mechanisms.enabled("shadow_field") and len(body) > 1:
            self.shadow[MS.address_of(observation.channel, body[:-1])] = body[-1:]
            self.mechanisms.fired("shadow_field", changed_state=True)

    def _commit_micro(self, proposal: Proposal) -> None:
        self.archive_undo[proposal.proposal_id] = len(self.developmental_archive)
        super()._commit_micro(proposal)
        self.developmental_archive.append(
            {
                "proposal_id": proposal.proposal_id,
                "target": proposal.target,
                "delta": list(proposal.delta),
            }
        )

    def _rollback(self, receipt: Receipt) -> None:
        prior_length = self.archive_undo.pop(receipt.proposal_id, None)
        if prior_length is not None:
            del self.developmental_archive[prior_length:]
        super()._rollback(receipt)

    def finalize_receipt(self, receipt: Receipt) -> None:
        self.archive_undo.pop(receipt.proposal_id, None)
        super().finalize_receipt(receipt)

    def shadow_answer(self, scope: str, key: Sequence[int]) -> tuple[int, ...] | None:
        """Read a temporary branch without modifying authoritative state."""
        return self.shadow.get(MS.address_of(scope, key))

    def _durable_state(self) -> Any:
        state = dict(super()._durable_state())
        state.update(
            {
                "developmental_archive": list(self.developmental_archive),
            }
        )
        return state

    def _active_state(self) -> Any:
        state = dict(super()._active_state())
        state["shadow"] = {address: list(value) for address, value in sorted(self.shadow.items())}
        state["archive_undo"] = dict(sorted(self.archive_undo.items()))
        return state

    def _restore_durable(self, state: Any) -> None:
        super()._restore_durable(state)
        self.developmental_archive = [dict(row) for row in state.get("developmental_archive", [])]

    def _restore_active(self, state: Any) -> None:
        super()._restore_active(state)
        self.shadow = {str(address): tuple(int(value) for value in row) for address, row in state.get("shadow", {}).items()}
        self.archive_undo = {str(proposal_id): int(length) for proposal_id, length in state.get("archive_undo", {}).items()}


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def _factory(
    cls: Any,
    name: str,
    mechanism: str,
    mode: str,
    *,
    disabled_by_default: Sequence[str] = (),
) -> Any:
    def build(opportunity: Opportunity, **options: Any) -> Any:
        disabled = frozenset((*disabled_by_default, *options.get("disabled_mechanisms", ())))
        return cls(
            name=name,
            mechanism=mechanism,
            _opportunity=opportunity,
            microstore_mode=str(options.get("microstore_mode", mode)),
            mechanisms=MechanismLog(disabled=disabled),
        )

    return build


register(
    "L1_associative_monolith",
    _factory(
        L1_associative_monolith,
        "L1_associative_monolith",
        "flat_exact_association_with_unconditional_consolidation",
        "exact",
        disabled_by_default=(
            "topology_change",
            "procedure_compilation",
            "self_model_allocation",
            "shadow_field",
        ),
    ),
)
register(
    "L9_minimal_sufficient_field",
    _factory(
        L9_minimal_sufficient_field,
        "L9_minimal_sufficient_field",
        "conditional_granularity_ladder_over_exact_microstore",
        "exact",
        disabled_by_default=(
            "precision_change",
            "topology_change",
            "shadow_field",
            "procedure_compilation",
            "self_model_allocation",
        ),
    ),
)

for _name, _class, _mechanism in (
    (
        "L2_associative_monolithic_plastic_field",
        L2_associative_monolithic_plastic_field,
        "dense_plastic_transition_over_exact_association",
    ),
    (
        "L3_associative_graph_plastic_field",
        L3_associative_graph_plastic_field,
        "typed_relation_graph_over_exact_association",
    ),
    (
        "L4_associative_cellular_field",
        L4_associative_cellular_field,
        "bounded_radius_cellular_scheduler",
    ),
    (
        "L5_associative_state_space_field",
        L5_associative_state_space_field,
        "bounded_recurrent_state_space_scheduler",
    ),
    (
        "L6_associative_event_sourced_field",
        L6_associative_event_sourced_field,
        "teaching_weighted_event_sourced_scheduler",
    ),
    (
        "L7_exact_microstore_mixed_radix_field",
        L7_exact_microstore_mixed_radix_field,
        "exact_microstore_with_quinary_structural_representation",
    ),
    (
        "L8_consolidation_first_field",
        L8_consolidation_first_field,
        "consolidation_first_structural_scheduler",
    ),
    (
        "L10_grok_original_compositional_field",
        L10_grok_original_compositional_field,
        "review_bounded_compositional_scheduler",
    ),
    (
        "L11_integrated_winner",
        L11_integrated_winner,
        "integrated_conditional_field_with_shadow_archive_and_replaceable_organs",
    ),
):
    register(_name, _factory(_class, _name, _mechanism, "exact"))


def demo() -> None:
    """Runnable self-check: staging, the ladder, and a rule answering an unseen key."""
    from substrate.genesis_material import Verdict, equal_opportunity

    observations = [Observation(index, "tool_use", (1, index, (3 * index + 2) % MS.MODULUS), teaching=True) for index in range(6)]
    opportunity = equal_opportunity(
        envelope="1GB",
        observations=observations,
        sensor_channels=("tool_use",),
        operation_budget=100_000,
        durable_write_budget=4_096,
    )
    from substrate.genesis_material import build

    material = cast(L9_minimal_sufficient_field, build("L9_minimal_sufficient_field", opportunity))
    for observation in observations:
        material.observe(observation)

    proposals = material.propose()
    assert proposals, "the field proposed nothing at all"
    kinds = {proposal.kind for proposal in proposals}
    assert "micro_association" in kinds or "structural_consolidation" in kinds, kinds

    material.apply([Verdict(proposal.proposal_id, True, 1.0, 1.0) for proposal in proposals])

    # An exact association that was written is answered exactly.
    written = [entry for entry in material.store.entries.values() if entry.scope == "tool_use"]
    assert written, "no association reached the microstore"

    # The consolidator must have fitted the affine map the tool actually obeys,
    # and it must answer a tool input that was never demonstrated.
    if material.rules.get("tool_use"):
        unseen = 11
        answer = material.answer(Probe(0, "tool_acquisition", "tool_use", (unseen,), 1))
        assert not answer.abstained
        assert answer.value == (((3 * unseen + 2) % MS.MODULUS),), answer.value

    # Checkpoint and restore reproduce the developed state exactly.
    snapshot = material.checkpoint()
    replica = build("L9_minimal_sufficient_field", opportunity)
    replica.restore(snapshot)
    assert replica.durable_state_digest() == material.durable_state_digest()

    # The monolith is a different material, not the same one renamed.
    monolith = build("L1_associative_monolith", opportunity)
    for observation in observations:
        monolith.observe(observation)
    assert monolith.mechanism != material.mechanism

    print(f"genesis2 material self-check passed: {len(material.store.entries)} associations, {material.committed_rules} rules installed")


if __name__ == "__main__":
    demo()
