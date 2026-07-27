"""Generation 2 scientific closure generator. Reads the sealed Gen2 results and emits append-only closure
authorities: closure, claim boundary, mechanism ledger, compute ledger, orchestration result, supersession map,
null-derived constraints, artifact index, and an evidence table. Originals are never modified. No dashes."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
from pathlib import Path

W = Path("/Users/scammermike/Downloads/mop-scientific-frontier")
OUT = W / "frontier/closure"
OUT.mkdir(parents=True, exist_ok=True)


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def L(rel):
    p = W / rel
    return json.loads(p.read_text()) if p.exists() else {}


# ---- complete mechanism ledger (14 fields per entry) ----
def M(premise, best_ev, pos, null, control, ext, arch, breadth, units, cost, fail, cls, replicate, new_premise):
    return {"scientific_premise": premise, "best_evidence": best_ev, "strongest_positive": pos,
            "strongest_null": null, "strongest_control": control, "external_replication_status": ext,
            "architecture_robustness": arch, "data_breadth": breadth, "independent_unit_quality": units,
            "compute_cost": cost, "failure_mode": fail, "current_classification": cls,
            "another_replication_justified": replicate, "new_premise_required": new_premise}


LEDGER = {
    "D1": M("route compute to the uniquely best perspective (value of compute)", "controlled-bed only",
            "none surviving", "retired before external test", "matched-compute router", "not tested",
            "unknown", "narrow", "n/a", "moderate", "router representation never cleared confirmation",
            "retired_frozen_design", False, "a router whose value beats matched-compute allocation on real tasks"),
    "U1": M("calibrated confidence tracks correctness and is worth its verification cost", "real-data canary",
            "none", "real-data canary null (MNIST-family): raw uncertainty fires on irreducible items",
            "raw entropy / margin", "failed at canary", "n/a", "MNIST-family regimes", "distribution regimes",
            "low", "raw uncertainty is not reducible-uncertainty", "real_data_canary_null", False,
            "an uncertainty signal that discriminates reducible from irreducible error"),
    "N1": M("chase reducible structure not irreducible noise", "MNIST canary positive then CIFAR-10 null",
            "MNIST real-data canary_positive", "CIFAR-10 second-source confirmation_null", "novelty heuristics",
            "failed second-source confirmation", "n/a", "MNIST then CIFAR-10", "distribution regimes",
            "moderate", "source-specific reducibility does not transfer across sources",
            "canary_positive_confirmation_null", False, "a novelty signal whose reducibility transfers across sources"),
    "R1": M("bounded retrieval selects information beyond recency/similarity/frequency/random", "KMNIST canary",
            "none", "KMNIST real-data canary_null: no incremental value beyond nearest-similarity",
            "nearest_similarity", "failed at canary", "0 of 3 capable estimators", "KMNIST", "class-group tasks",
            "low", "learned retrieval did not beat nearest-similarity", "real_data_canary_null", False,
            "a retrieval mechanism that beats nearest-similarity by SESOI across sources"),
    "P1": M("hold stability and plasticity at matched cost", "pruned in mechanics-extended", "none",
            "pruned (superseded by P1R)", "uniform replay", "not tested", "n/a", "MNIST-family", "tasks",
            "low", "subsumed by the P1R formulation", "pruned_superseded_by_P1R", False, "n/a (folded into P1R)"),
    "P1R": M("predict per-item replay value to hold stability and plasticity", "three same-team positives",
             "same_team_cross_architecture_positive on split-MNIST, CIFAR-100, KMNIST",
             "EMNIST external-method replication_null: faithful P1R 0.104 vs GDumb 0.594, above no-replay 0.068",
             "GDumb (best established), reservoir", "external_method_replication_null",
             "cross-architecture within same team", "MNIST-family, CIFAR-100, KMNIST, EMNIST", "tasks",
             "high (EMNIST class-incremental)", "per-item value prediction does not yield a competitive replay policy",
             "strongest_surviving_internal_hypothesis (same-team benchmark positives, external null, no downstream license)",
             True, "value used as a soft sampling priority over a representative buffer, made to beat GDumb"),
    "V1": M("selective re-checking is worth its matched cost", "frontier real-data admission",
            "incremental value over uncertainty controls (D lcb 0.062)", "only 1 of 3 capable estimators decodes it",
            "raw uncertainty (U1 frozen negative)", "not reached (admission architecture_dependent)",
            "1 of 3 capable estimators", "CIFAR-10 regimes", "distribution regimes", "moderate",
            "verification value not robustly decodable across capable architectures", "architecture_dependent",
            True, "verification value decodable by a capable estimator family, downstream-value targeted"),
    "K1": M("repair fires only on detected disagreement and corrects material errors", "frontier admission",
            "architecture-robust value (F 3/3, D lcb 0.10)", "fails noisy-TV: fires on contradictions not warranting repair",
            "consistency check / majority", "not reached", "3 of 3 capable estimators", "CIFAR-10 regimes",
            "distribution regimes", "moderate", "contradiction detection is not repair necessity", "pruned_mechanism",
            True, "estimate repair necessity (net corrected-decision value), not mere disagreement"),
    "M1": M("bounded causal messaging beats a limited broadcast", "frontier admission",
            "message value highly predictable (D lcb 0.33, F 3/3)", "fails noisy-TV: fires where the message does not causally help",
            "no-message / A-uncertainty", "not reached", "3 of 3 capable estimators", "split-view MNIST",
            "view splits / regimes", "low", "message predictability is not causal message value", "pruned_mechanism",
            True, "estimate intervention-level (causal) message value vs a centralized matched-capacity control"),
    "E1": M("carve the stream into meaningful events that aid downstream learning", "frontier admission",
            "relational boundaries help (oracle headroom 0.19)", "not beyond simple change detectors by SESOI (lcb -0.004)",
            "novelty / change-point detectors", "not reached", "n/a (sequence harness)", "ordered KMNIST stream",
            "sessions", "low", "relational boundaries did not beat simple change detectors", "pruned_mechanism",
            True, "model relations and temporal role changes simple change detectors cannot capture"),
    "C0": M("a stable persistent trace aids downstream decisions", "frontier admission", "none",
            "worse than EMA smoothing (wrong direction, lcb -0.036)", "EMA smoothing / matched-memory buffer",
            "not reached", "n/a", "noisy KMNIST stream", "sessions", "low",
            "the trace did not beat EMA smoothing in downstream value", "pruned_mechanism", False,
            "a trace whose downstream value exceeds EMA smoothing and matched-memory buffers"),
    "A1": M("read action-relevance (affordance) straight from the latent", "frontier admission (gym)",
            "competent policy (beats random by 109 return)", "ties a fitted value estimator; no incremental value",
            "fitted value estimator / behavior cloning", "not reached", "point-negative vs controls",
            "gymnasium classic-control", "distinct dynamical systems", "low",
            "affordance reading is not distinct from a fitted value estimator", "pruned_mechanism", False,
            "an affordance readout that beats a fitted value estimator on the same latent"),
    "S1": M("simulating the consequence beats acting reactively", "frontier admission (gym)", "none",
            "learned-model planning worse than random (model errors compound over the horizon)",
            "reactive / one-step planner", "not reached", "point-negative", "gymnasium classic-control",
            "distinct dynamical systems", "low", "compounding learned-model error", "pruned_mechanism", True,
            "simulation that conditions depth/trust on estimated rollout error and controls compounding"),
    "I1": M("integrate confirmed subsystems into an escalating architecture", "none (never licensed)", "none",
            "retired route", "n/a", "not tested", "n/a", "n/a", "n/a", "n/a", "no dependency closure ever existed",
            "retired_route", False, "three externally confirmed functional domains with explicit dependency closure"),
    "G1": M("cost-charged topology search over shadow coalitions", "none (never licensed)", "none",
            "unlicensed", "fixed simple composition", "not tested", "n/a", "n/a", "n/a", "n/a",
            "no confirmed components to search over", "unlicensed_never_launched", False,
            "at least one externally replicated component or confirmed cluster to compose"),
    "Cluster_A_action_simulation": M("A1 and S1 interact to improve control beyond either alone", "none",
            "none", "unlicensed: A1 and S1 both pruned at admission", "strongest simple planner", "not tested",
            "n/a", "n/a", "n/a", "n/a", "no admitted component", "unlicensed", False,
            "both A1 and S1 pass confirmation first"),
    "Cluster_B_memory_plasticity": M("R1 retrieval and P1R replay interact to improve retention and adaptation",
            "R1 admission null on KMNIST", "none", "blocked by R1 canary_null; factorial not licensed",
            "P1R alone / strongest simple replay", "not tested", "n/a", "KMNIST", "tasks", "moderate",
            "R1 did not clear admission", "blocked_terminated", False,
            "a retrieval component that clears its own admission"),
    "Cluster_C_verification_repair_messaging": M("a preregistered V1/K1/M1 subset beats the best simpler alternative",
            "none", "none", "unlicensed: V1 architecture_dependent, K1/M1 pruned; no passing subset", "centralized matched-capacity",
            "not tested", "n/a", "n/a", "n/a", "n/a", "no confirmed component", "unlicensed", False,
            "at least one confirmed reliability component"),
    "Cluster_Event_Trace": M("E1 boundaries and C0 traces interact for downstream reasoning", "none", "none",
            "unlicensed: E1 and C0 both pruned", "matched-memory recurrent control", "not tested", "n/a", "n/a",
            "n/a", "n/a", "no admitted component", "unlicensed", False, "both E1 and C0 pass confirmation"),
}


def build():
    adm = {x: L(f"frontier/reports/MOP_FRONTIER_{x}_ADMISSION_RESULT.json") for x in ["V", "K", "M", "E", "C", "A", "S"]}
    P = L("frontier/reports/MOP_FRONTIER_P_RESULT.json")
    bench = L("frontier/reports/MOP_FRONTIER_PARALLEL_BENCHMARK.json")
    replay = L("frontier/MOP_FRONTIER_PARALLEL_REPLAY.json")

    # 1. supersession map (append-only correction of the terminal wording)
    supersession = {
        "schema": "mop-gen2-supersession/v1",
        "supersedes": [{
            "original_statement": "confirmed architecture remains P1R alone",
            "original_artifacts_unchanged": ["frontier/MOP_GENERATION2_FRONTIER_SYNTHESIS.md",
                                             "frontier/MOP_GENERATION2_ARCHITECTURE_BOUNDARY.json"],
            "corrected_authority": ("P1R is the strongest surviving internal hypothesis: three same-team benchmark "
                                    "positives (split-MNIST, CIFAR-100, KMNIST), bounded by a stronger external "
                                    "replication null (EMNIST), with no construction, integration, or activation license."),
            "reason": "P1R was never externally independently replicated and its external-method replication is null; "
                      "calling it a confirmed architecture overstates the evidence class.",
        }],
        "rule": "corrections are append-only; original sealed artifacts are preserved unchanged.",
    }
    supersession["sha256"] = sha(supersession)
    (W / "frontier/MOP_GENERATION2_SUPERSESSION_MAP.json").write_text(json.dumps(supersession, indent=2))

    # 2. claim boundary
    claim = {
        "schema": "mop-gen2-claim-boundary/v1",
        "mop_claims": ["robust seeded experimental mechanics across many mechanism families",
                       "a calibrated, construct-validity-checked admission battery that prunes null/oracle-free beds",
                       "real-data canary and same-team cross-architecture evidence for P1R",
                       "a work-conserving resource-token DAG orchestration layer"],
        "mop_does_not_claim": ["any externally independently replicated mechanism",
                               "any mechanism with incremental value beyond strong established controls that survives external test",
                               "any licensed cluster, construction, integration, or activation",
                               "that P1R is a competitive replay method (it is not, vs GDumb)"],
        "strongest_surviving_hypothesis": supersession["supersedes"][0]["corrected_authority"],
        "activation": False,
    }
    claim["sha256"] = sha(claim)
    (W / "frontier/MOP_GENERATION2_CLAIM_BOUNDARY.json").write_text(json.dumps(claim, indent=2))

    # 3. mechanism ledger
    ledger = {"schema": "mop-gen2-mechanism-ledger/v1", "fields": list(next(iter(LEDGER.values())).keys()),
              "mechanisms": LEDGER, "count": len(LEDGER)}
    ledger["sha256"] = sha(ledger)
    (W / "frontier/MOP_GENERATION2_MECHANISM_LEDGER.json").write_text(json.dumps(ledger, indent=2))

    # 4. compute ledger
    walls = {f"admission_{x}": adm[x].get("wall_seconds") for x in adm}
    walls["lane_P_faithful"] = P.get("wall_seconds")
    walls["lane_P_v1_superseded"] = 1025.5
    walls["adversarial_audit_workflow"] = 2748.3
    compute = {
        "schema": "mop-gen2-compute-ledger/v1",
        "environment": {"host_cores": 28, "memory_gb": 96, "torch": "2.13", "gymnasium": "1.3.0",
                        "absent": ["torchaudio", "scipy", "sklearn", "librosa"]},
        "per_lane_wall_seconds": walls,
        "observed_wall_seconds_excl_rework": 3865.5, "observed_wall_seconds_incl_rework": 7639.3,
        "modeled_parallel_wall_seconds": replay.get("decomposed_lane_p", {}).get("modeled_parallel_wall_seconds"),
        "datasets_acquired": ["EMNIST-balanced (download)"],
        "datasets_reused": ["MNIST", "KMNIST", "FashionMNIST", "CIFAR-10", "CIFAR-100"],
        "note": "principal compute dominated by Lane P (EMNIST class-incremental, 5 seeds x 6 methods).",
    }
    compute["sha256"] = sha(compute)
    (W / "frontier/MOP_GENERATION2_COMPUTE_LEDGER.json").write_text(json.dumps(compute, indent=2))

    # 5. orchestration result (engineering, separate from science)
    orch = {
        "schema": "mop-gen2-orchestration-result/v1",
        "observed_wall_seconds": 3866, "parallel_modeled_wall_seconds": 1226, "modeled_speedup": 3.15,
        "observed_average_concurrency": 1.24, "corrected_average_concurrency": 3.81,
        "measured_four_capsule_aggregate_throughput_x": 3.5,
        "measured_slowdown_by_concurrency": bench.get("measured_slowdown_by_concurrency", {}),
        "explicit_statement": ("The global DAG scheduler is an engineering improvement. It does not increase any "
                               "mechanism's scientific evidence class."),
        "promoted_infrastructure": ["resource-token DAG scheduler", "task-class resource profiles", "receipt cache",
                                    "resumable verification", "retry escalation", "notifications"],
        "not_promoted": ["any failed scientific mechanism (D1,U1,N1,R1,V1,K1,M1,E1,C0,A1,S1)"],
    }
    orch["sha256"] = sha(orch)
    (W / "frontier/MOP_GENERATION2_ORCHESTRATION_RESULT.json").write_text(json.dumps(orch, indent=2))

    # 6. null-derived constraints (mandatory admission clauses for future mechanisms)
    constraints = {
        "schema": "mop-null-derived-constraints/v1",
        "constraints": [
            {"origin": "U1", "constraint": "high uncertainty is not reducible uncertainty; a mechanism must discriminate reducible from irreducible error, not fire on raw uncertainty"},
            {"origin": "N1", "constraint": "source-specific reducibility does not imply cross-source value; a value signal must transfer across sources before confirmation"},
            {"origin": "R1", "constraint": "learned retrieval must beat nearest-similarity by SESOI, not merely perform a successful nearest-neighbour lookup"},
            {"origin": "E1", "constraint": "relational event boundaries must beat simple change detectors in downstream value, not merely segment the stream"},
            {"origin": "C0", "constraint": "a stable trace must beat EMA smoothing and matched-memory buffers in downstream value"},
            {"origin": "A1", "constraint": "affordance reading must beat a fitted value estimator on the same latent"},
            {"origin": "S1", "constraint": "learned rollouts must control compounding model error and beat direct prediction and simple planning after charging model-error estimation"},
            {"origin": "M1", "constraint": "message predictability is not causal message value; a messaging mechanism must estimate intervention-level value and beat a centralized matched-capacity control"},
            {"origin": "K1", "constraint": "contradiction detection is not repair necessity; a repair mechanism must estimate net corrected-decision value and avoid false-repair harm"},
            {"origin": "P1R", "constraint": "per-item value prediction is not sufficient for a competitive replay policy; a replay mechanism must beat the best established method (GDumb), not only no-replay"},
            {"origin": "V1", "constraint": "incremental value must hold across a family of sufficiently capable architectures, not a single one, to be admitted"},
            {"origin": "battery", "constraint": "every mechanism must clear noisy-TV, shuffled-target, wrong-time, and rate-matched-random controls, and provide oracle headroom, before a canary"},
        ],
        "usage": "these become mandatory admission clauses for every future mechanism.",
    }
    constraints["sha256"] = sha(constraints)
    (W / "frontier/MOP_NULL_DERIVED_CONSTRAINTS.json").write_text(json.dumps(constraints, indent=2))

    # 7. closure
    closure = {
        "schema": "mop-gen2-closure/v1",
        "what_was_tested": "15 mechanism hypotheses (D1,U1,N1,R1,P1,P1R,V1,K1,M1,E1,C0,A1,S1,I1,G1) plus four cluster gates",
        "passed_controlled_beds_only": ["P1R (Phase 4B)", "V1/K1/M1/E1/C0/A1/S1 have mechanics + battery structure but did not clear real-data admission"],
        "passed_benchmark_canaries": ["N1 (MNIST)", "P1R (split-MNIST)"],
        "replicated_same_team_implementations": ["P1R (CIFAR-100, KMNIST)"],
        "failed_external_confirmation": ["N1 (CIFAR-10 second source)", "P1R (EMNIST external-method)"],
        "pruned": ["U1", "P1", "K1", "M1", "E1", "C0", "A1", "S1"],
        "invalid_beds": ["none in the frontier (calibration caught bed invalidity before principal use); Lane P v1 was an unfaithful operationalization, corrected by one bounded repair"],
        "architecture_dependent": ["V1"],
        "canary_null": ["U1", "R1"],
        "why_clusters_did_not_launch": "no cluster had two admitted/confirmed components: Action-Simulation (A1,S1 pruned), Verification-Repair-Messaging (V1 architecture_dependent, K1/M1 pruned), Event-Trace (E1,C0 pruned), Memory-Plasticity/Cluster B (R1 null blocked the factorial)",
        "why_construction_unauthorized": "construction needs external P1R replication (null) or a confirmed cluster (none) or two confirmed components (none)",
        "why_integration_unauthorized": "integration needs three confirmation-level functional domains with dependency closure; zero exist",
        "why_activation_false": "activation requires an explicit separate grant and at minimum a licensed integration; neither exists",
        "strongest_surviving_hypothesis": claim["strongest_surviving_hypothesis"],
        "tags": ["mop-generation2-closure", "mop-generation2-null-map", "mop-generation2-orchestration"],
    }
    closure["sha256"] = sha(closure)
    (W / "frontier/MOP_GENERATION2_CLOSURE.json").write_text(json.dumps(closure, indent=2))

    # 8. artifact index + evidence table
    index = {"schema": "mop-gen2-artifact-index/v1", "artifacts": {}}
    for pat in ["frontier/MOP_GENERATION2_*.json", "frontier/MOP_FRONTIER_*.json", "frontier/reports/*.json",
                "campaign2/reports/*.json", "salvage/MOP_EVIDENCE_CAMPAIGN_STATE.json"]:
        for f in sorted(glob.glob(str(W / pat))):
            rel = os.path.relpath(f, W)
            index["artifacts"][rel] = hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
    index["count"] = len(index["artifacts"])
    index["sha256"] = sha(index)
    (W / "frontier/MOP_GENERATION2_ARTIFACT_INDEX.json").write_text(json.dumps(index, indent=2))

    rows = [["mechanism", "classification", "strongest_positive", "strongest_null", "external_status", "architecture_robustness"]]
    for k, v in LEDGER.items():
        rows.append([k, v["current_classification"], v["strongest_positive"][:60], v["strongest_null"][:60],
                     v["external_replication_status"], v["architecture_robustness"]])
    with open(W / "frontier/MOP_GENERATION2_EVIDENCE_TABLE.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    print("closure sealed:", closure["sha256"][:16])
    print("ledger mechanisms:", len(LEDGER), "| artifacts indexed:", index["count"])
    print("classifications:", {k: v["current_classification"] for k, v in LEDGER.items()})


if __name__ == "__main__":
    build()
