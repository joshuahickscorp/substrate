"""The materialized program graph. One structure, no prose waves.

The previous arrangement had two things: a stage graph the driver could execute, and a wave plan written in
English that named prerequisites nobody had built. That split is comfortable and wrong. A wave described as
blocked on a model body will stay blocked forever, because nothing in the system is responsible for
building one. Section 5 says the world model bed, the body adapters and the real session authority become
actual nodes, and section 3 says that when a blocker can be built with available data, code and compute,
the program creates the node that builds it.

So the blockers are nodes here. `world_model_bed`, `body_adapter_compact`, `body_adapter_general`,
`body_adapter_tool`, `real_session_authority` are implementation nodes with entry and exit gates like any
other, and they run.

A node is terminal only when its exit gate passes. A gate is a predicate over the tree, not a promise, so
a node cannot be marked done by declaring it done.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from mop.cognition import io

# section 5, the declared task types
TASK_TYPES = ("implementation", "data_acquisition", "instrument_calibration", "experiment", "training",
              "verification", "mutation", "analysis", "report", "integration", "clean_clone")

# section 5, what every node declares
NODE_FIELDS = ("identity", "task_type", "dependencies", "entry_gate", "exit_gate", "source_authority",
               "resource_profile", "checkpoint_policy", "verification", "failure_policy", "successors")

RESOURCE_PROFILES = ("trivial", "light", "moderate", "principal", "external")
FAILURE_POLICIES = ("hold_after_two", "hold_immediately", "advisory")


class Refused(RuntimeError):
    """A graph operation the structure does not permit."""


@dataclass(frozen=True)
class Node:
    identity: str
    task_type: str
    module: str = ""
    args: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    entry_gate: str = "dependencies terminal"
    exit_gate: str = ""
    produces: tuple[str, ...] = ()  # proof artifacts that must exist for the exit gate to pass
    requires_impl: tuple[str, ...] = ()  # repo paths that must exist for the entry gate to pass
    source_authority: str = "SUBSTRATE_FINAL_AUTONOMOUS_PROGRAM.md"
    resource_profile: str = "light"
    checkpoint_policy: str = "receipt per node, resumable from disk"
    verification: str = "independent recomputation plus mutation"
    failure_policy: str = "hold_after_two"
    external_blocker: str = ""  # non empty means genuinely unavailable, not merely unbuilt

    def violations(self) -> list[str]:
        v = []
        if self.task_type not in TASK_TYPES:
            v.append(f"{self.identity}: undeclared task type {self.task_type!r}")
        if self.resource_profile not in RESOURCE_PROFILES:
            v.append(f"{self.identity}: undeclared resource profile {self.resource_profile!r}")
        if self.failure_policy not in FAILURE_POLICIES:
            v.append(f"{self.identity}: undeclared failure policy {self.failure_policy!r}")
        if not self.exit_gate:
            v.append(f"{self.identity}: no exit gate, so it can never be terminal")
        if not self.module and not self.external_blocker:
            v.append(f"{self.identity}: no module and no external blocker, so nothing can run it")
        return v


def _n(identity, task_type, **kw) -> Node:
    return Node(identity=identity, task_type=task_type, **kw)


COG = "src/mop/cognition"
D, X, V = "mop.cognition.deliverables", "mop.cognition.experiments", "mop.cognition.verify"

NODES: tuple[Node, ...] = (
    # ---------------------------------------------------------------- category 1
    _n("declarations", "implementation", module=D, args=("seal-modules",),
       exit_gate="every module declaration is sealed at a commit reachable from HEAD",
       produces=("SUBSTRATE_ONTOLOGY.json", "SUBSTRATE_EPISTEMOLOGY.json", "SUBSTRATE_WORKSPACE.json",
                 "SUBSTRATE_RUNTIME.json"),
       resource_profile="trivial"),
    _n("tests", "verification", module="mop.cognition.program", args=("tests",),
       dependencies=("declarations",),
       exit_gate="every declared test node is recorded passing", resource_profile="light"),
    _n("authority", "report", module="mop.cognition.authority", args=("seal",),
       dependencies=("tests",),
       exit_gate="the six final authority artifacts exist and bind to the current commit",
       produces=("SUBSTRATE_FINAL_MASTER_AUTHORITY.json", "SUBSTRATE_FINAL_ANCESTRY.json",
                 "SUBSTRATE_FINAL_STATE.json", "SUBSTRATE_FINAL_SCORECARD.json",
                 "SUBSTRATE_FINAL_VALUE_QUEUE.json", "SUBSTRATE_FINAL_PROGRAM_GRAPH.json"),
       requires_impl=(f"{COG}/authority.py",), resource_profile="trivial"),

    # ---------------------------------------------------------------- category 2, the built blockers
    _n("temporal_core_integration", "integration", module="mop.cognition.temporal_link", args=("seal",),
       dependencies=("declarations",),
       entry_gate="the temporal core authority is terminal, licensed or not",
       exit_gate="the runtime reads temporal state through a versioned interface and the five "
                 "information sources stay distinguishable",
       produces=("SUBSTRATE_TEMPORAL_CORE.json",),
       requires_impl=(f"{COG}/temporal_link.py",), resource_profile="light"),
    _n("world_model_bed", "data_acquisition", module="mop.cognition.worldbed", args=("build",),
       dependencies=("declarations",),
       entry_gate="a corpus under custody with state dependent decisions",
       exit_gate="a bed exists whose best fixed action is not optimal everywhere",
       produces=("SUBSTRATE_WORLD_MODEL_BED.json",),
       requires_impl=(f"{COG}/worldbed.py",), resource_profile="moderate"),
    _n("world_model_in_loop", "integration", module="mop.cognition.worldbed", args=("integrate",),
       dependencies=("world_model_bed", "temporal_core_integration"),
       entry_gate="the bed exists and the runtime cycle is terminal",
       exit_gate="a world model prediction changes which action the arbiter selects, measured inside "
                 "the cycle",
       produces=("SUBSTRATE_WORLD_MODEL_BATTERY.json",), resource_profile="moderate"),
    _n("body_adapter_compact", "implementation", module="mop.cognition.bodies", args=("compact",),
       dependencies=("declarations",),
       entry_gate="the nine message kinds are declared",
       exit_gate="a compact specialist body conforms and the runtime drives it",
       produces=("SUBSTRATE_BODY_COMPACT.json",),
       requires_impl=(f"{COG}/bodies.py",), resource_profile="light"),
    _n("body_adapter_general", "implementation", module="mop.cognition.bodies", args=("general",),
       dependencies=("body_adapter_compact",),
       entry_gate="the compact adapter conforms",
       exit_gate="a larger general body conforms through the same interface",
       produces=("SUBSTRATE_BODY_GENERAL.json",), resource_profile="light"),
    _n("body_adapter_tool", "implementation", module="mop.cognition.bodies", args=("tool",),
       dependencies=("body_adapter_compact",),
       entry_gate="the compact adapter conforms",
       exit_gate="a tool dominant body conforms through the same interface",
       produces=("SUBSTRATE_BODY_TOOL.json",), resource_profile="light"),
    _n("body_comparison", "experiment", module="mop.cognition.bodies", args=("compare",),
       dependencies=("body_adapter_compact", "body_adapter_general", "body_adapter_tool"),
       entry_gate="three bodies conform",
       exit_gate="the six declared ablations are measured against all three bodies",
       produces=("SUBSTRATE_MODEL_BODY_INTERFACE.json",), resource_profile="moderate"),
    _n("real_session_authority", "data_acquisition", module="mop.cognition.sessions", args=("build",),
       dependencies=("declarations",),
       entry_gate="a recorded session this program did not write for the purpose",
       exit_gate="a session authority exists with its provenance and its length not chosen by us",
       produces=("SUBSTRATE_REAL_SESSION_AUTHORITY.json",),
       requires_impl=(f"{COG}/sessions.py",), resource_profile="light"),

    # ---------------------------------------------------------------- category 3 to 5
    _n("goal_and_valuation", "implementation", module="mop.cognition.goals", args=("seal",),
       dependencies=("declarations",),
       entry_gate="the safety envelope declares goal authority",
       exit_gate="goals carry nine fields and valuation is externally authorized and auditable",
       produces=("SUBSTRATE_GOAL_SYSTEM.json", "SUBSTRATE_VALUATION_SYSTEM.json"),
       requires_impl=(f"{COG}/goals.py",), resource_profile="light"),
    _n("grounding", "implementation", module="mop.cognition.grounding", args=("seal",),
       dependencies=("real_session_authority", "goal_and_valuation"),
       entry_gate="a session with referents and outcomes",
       exit_gate="the nine grounding tests run and a symbol with no referent is refused",
       produces=("SUBSTRATE_GROUNDING.json",),
       requires_impl=(f"{COG}/grounding.py",), resource_profile="light"),
    _n("perspective_expansion", "implementation", module="mop.cognition.perspectives", args=("seal",),
       dependencies=("declarations",),
       entry_gate="every family has a declared verification method or a stated refusal",
       exit_gate="every declared family is implemented or terminally gated with its reason",
       produces=("SUBSTRATE_PERSPECTIVE_SYSTEM.json",), resource_profile="light"),
    _n("developmental_divergence", "experiment", module="mop.cognition.divergence", args=("run",),
       dependencies=("real_session_authority", "temporal_core_integration"),
       entry_gate="two identical instances and two different verified histories",
       exit_gate="divergence is measured across the nine declared dimensions and tested for transfer",
       produces=("SUBSTRATE_DEVELOPMENTAL_HISTORY.json",),
       requires_impl=(f"{COG}/divergence.py",), resource_profile="moderate"),
    _n("entity_batteries", "experiment", module="mop.cognition.batteries", args=("seal",),
       dependencies=("world_model_in_loop", "body_comparison", "grounding", "goal_and_valuation"),
       entry_gate="a runtime with a world model, a body and grounded symbols",
       exit_gate="thinking, continuity, unity, reflection, agency and integrity each reach a "
                 "classification or a stated measurement boundary",
       produces=("SUBSTRATE_THINKING_BATTERY.json", "SUBSTRATE_CONTINUITY_BATTERY.json",
                 "SUBSTRATE_UNITY_BATTERY.json", "SUBSTRATE_REFLECTIVE_ACCESS_BATTERY.json",
                 "SUBSTRATE_AGENCY_BATTERY.json", "SUBSTRATE_COGNITIVE_INTEGRITY_BATTERY.json"),
       resource_profile="moderate"),

    # ---------------------------------------------------------------- category 6
    _n("experiments", "experiment", module=X, args=("seal",), dependencies=("tests",),
       exit_gate="every declared experiment reaches a decision with its reason recorded",
       resource_profile="light"),
    _n("bed_screen", "instrument_calibration", module=X, args=("screen",),
       dependencies=("experiments",),
       exit_gate="every bed under custody is screened against the declared SESOI",
       resource_profile="light"),
    _n("sx1b", "experiment", module=X, args=("sx1b",), dependencies=("bed_screen",),
       exit_gate="a classification or a refusal naming the number that decides it",
       resource_profile="light"),
    _n("sx5", "experiment", module=X, args=("sx5",), dependencies=("bed_screen",),
       exit_gate="a classification or a refusal naming the number that decides it",
       resource_profile="light"),
    _n("deliverables", "report", module=D, args=("write",), dependencies=("sx1b", "sx5", "authority"),
       exit_gate="every master artifact is regenerated from the tree",
       produces=("SUBSTRATE_STATE.json", "SUBSTRATE_NEXT_FRONTIER.json"), resource_profile="trivial"),
    _n("recompute", "verification", module=V, args=("recompute",), dependencies=("deliverables",),
       exit_gate="every sealed number agrees with a second route and no seal is broken",
       produces=("SUBSTRATE_INDEPENDENT_VERIFICATION.json",), resource_profile="light"),
    _n("mutate", "mutation", module=V, args=("mutate",), dependencies=("recompute",),
       exit_gate="every mutation is rejected by the guard named beside it",
       produces=("SUBSTRATE_MUTATION_REPORT.json",), resource_profile="moderate"),
    _n("classify", "analysis", module=X, args=("classify",), dependencies=("mutate",),
       exit_gate="a verified measurement carries a recorded classification",
       resource_profile="trivial"),
    _n("clean_clone", "clean_clone", module="mop.cognition.cleanclone", args=("run",),
       dependencies=("classify",),
       entry_gate="the working tree is committed",
       exit_gate="a fresh clone at this commit imports, tests and reproduces every sealed number",
       produces=("SUBSTRATE_CLEAN_CLONE.json",),
       requires_impl=(f"{COG}/cleanclone.py",), resource_profile="moderate"),
    _n("consolidate", "report", module=D, args=("write",),
       dependencies=("classify", "entity_batteries", "developmental_divergence", "clean_clone"),
       exit_gate="the frontier, the value queue and the scorecard carry every new classification",
       produces=("SUBSTRATE_CURRENT_ENTITY_SPEC.json",), resource_profile="trivial"),
)

BY_ID = {n.identity: n for n in NODES}


# ---------------------------------------------------------------- gates


def entry_open(node: Node) -> dict:
    missing_impl = [p for p in node.requires_impl if not (io.ROOT / p).exists()]
    return {"open": not missing_impl and not node.external_blocker,
            "missing_implementation": missing_impl,
            "external_blocker": node.external_blocker}


def exit_passed(node: Node) -> dict:
    from mop.cognition import program as P

    missing = [a for a in node.produces if not P.evidence_state(a)["counts"]]
    return {"passed": not missing and not entry_open(node)["missing_implementation"],
            "missing_artifacts": missing}


def validate() -> list[str]:
    v = [x for n in NODES for x in n.violations()]
    seen = set()
    for n in NODES:
        for d in n.dependencies:
            if d not in BY_ID:
                v.append(f"{n.identity} depends on undeclared node {d}")
        if not set(n.dependencies) <= seen:
            v.append(f"{n.identity} is declared before a dependency")
        seen.add(n.identity)
    return v


def successors(identity: str) -> list[str]:
    return sorted(n.identity for n in NODES if identity in n.dependencies)


def buildable_blockers() -> list[dict]:
    """Nodes whose only obstacle is that nobody has written the module yet. Those are work, not blockers."""
    rows = []
    for n in NODES:
        gate = entry_open(n)
        if gate["missing_implementation"] and not n.external_blocker:
            rows.append({"node": n.identity, "missing": gate["missing_implementation"],
                         "classification": "buildable_prerequisite",
                         "rule": "a blocker that can be built with available code and compute is a node"})
    return rows


def externally_blocked() -> list[dict]:
    return [{"node": n.identity, "blocker": n.external_blocker} for n in NODES if n.external_blocker]


def declaration() -> dict:
    from mop.cognition import program as P

    rows = []
    for n in NODES:
        gate, exit_ = entry_open(n), exit_passed(n)
        rows.append({"identity": n.identity, "task_type": n.task_type,
                     "dependencies": list(n.dependencies), "entry_gate": n.entry_gate,
                     "exit_gate": n.exit_gate, "source_authority": n.source_authority,
                     "resource_profile": n.resource_profile,
                     "checkpoint_policy": n.checkpoint_policy, "verification": n.verification,
                     "failure_policy": n.failure_policy, "successors": successors(n.identity),
                     "entry_open": gate["open"], "exit_passed": exit_["passed"],
                     "missing_implementation": gate["missing_implementation"],
                     "missing_artifacts": exit_["missing_artifacts"],
                     "external_blocker": n.external_blocker})
    violations = validate()
    return {
        "schema": "substrate-final-program-graph/v1",
        "node_fields": list(NODE_FIELDS),
        "task_types": list(TASK_TYPES),
        "nodes": rows,
        "node_count": len(NODES),
        "declaration_violations": violations,
        "valid": not violations,
        "terminal_nodes": sorted(r["identity"] for r in rows if r["exit_passed"]),
        "buildable_prerequisites": buildable_blockers(),
        "externally_blocked": externally_blocked(),
        "rule": ("a blocker that can be built with available data, code and compute is a node in this "
                 "graph, not a sentence in a plan. Only an unavailable external resource is terminal"),
        "no_prose_waves": True,
        "activation": False,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    doc = declaration()
    path = io.seal("SUBSTRATE_FINAL_PROGRAM_GRAPH.json", doc)
    print(json.dumps({"sealed": path.relative_to(io.ROOT).as_posix(), "nodes": doc["node_count"],
                      "valid": doc["valid"], "terminal": len(doc["terminal_nodes"]),
                      "buildable_prerequisites": [b["node"] for b in doc["buildable_prerequisites"]],
                      "externally_blocked": doc["externally_blocked"]}, indent=2))


if __name__ == "__main__":
    main()
