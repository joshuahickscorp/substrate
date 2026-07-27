"""Generate the Generation 2 frontier terminal synthesis artifacts from all sealed lane results.
House style: no em dashes and no en dashes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

W = Path("/Users/scammermike/Downloads/mop-scientific-frontier")
R = W / "frontier/reports"
OUT = W / "frontier"


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(W)).stdout.strip()
    adm = {L: load(f"MOP_FRONTIER_{L}_ADMISSION_RESULT.json") for L in ["V", "K", "M", "E", "C", "A", "S"]}
    P = load("MOP_FRONTIER_P_RESULT.json")
    P_audit = load("MOP_FRONTIER_P_ADVERSARIAL_AUDIT.json")

    # Evidence classes ladder (section 32)
    LADDER = ["mechanics_robustness", "controlled_bed_plausibility", "real_benchmark_canary",
              "same_team_cross_architecture_confirmation", "external_method_replication",
              "independent_scientific_confirmation", "cluster_interaction", "construction", "integration", "activation"]

    # Frozen prior evidence (from the freeze authority + prior campaign)
    evidence_matrix = {
        "G1-P1R": {"mechanics_robustness": True, "controlled_bed_plausibility": True, "real_benchmark_canary": "positive (split-MNIST)",
                   "same_team_cross_architecture_confirmation": "positive (CIFAR-100, KMNIST)",
                   "external_method_replication": (P["classification"] if P else "pending"),
                   "independent_scientific_confirmation": False, "cluster_interaction": "unlicensed",
                   "construction": "unlicensed", "integration": "unlicensed", "activation": False},
        "G1-N1": {"mechanics_robustness": True, "controlled_bed_plausibility": True, "real_benchmark_canary": "positive (MNIST)",
                  "same_team_cross_architecture_confirmation": "confirmation_null (CIFAR-10)", "external_method_replication": "n/a",
                  "independent_scientific_confirmation": False, "activation": False, "terminal": "negative"},
        "G1-U1": {"mechanics_robustness": True, "controlled_bed_plausibility": True, "real_benchmark_canary": "canary_null",
                  "terminal": "negative", "activation": False},
        "G1-R1": {"mechanics_robustness": True, "controlled_bed_plausibility": True, "real_benchmark_canary": "canary_null (KMNIST)",
                  "terminal": "negative", "activation": False},
        "G1-D1": {"terminal": "retired_frozen_design", "activation": False},
        "historical_G1-I1": {"terminal": "retired_route", "activation": False},
        "G1-V1": {"mechanics_robustness": True, "controlled_bed_plausibility": True,
                  "real_benchmark_canary": "admission=" + adm["V"]["classification"], "terminal": "not_admitted", "activation": False},
        "G1-K1": {"mechanics_robustness": True, "real_benchmark_canary": "admission=" + adm["K"]["classification"], "terminal": "not_admitted", "activation": False},
        "G1-M1": {"mechanics_robustness": True, "real_benchmark_canary": "admission=" + adm["M"]["classification"], "terminal": "not_admitted", "activation": False},
        "G1-E1": {"mechanics_robustness": True, "real_benchmark_canary": "admission=" + adm["E"]["classification"], "terminal": "not_admitted", "activation": False},
        "G1-C0": {"mechanics_robustness": True, "real_benchmark_canary": "admission=" + adm["C"]["classification"], "terminal": "not_admitted", "activation": False},
        "G1-A1": {"mechanics_robustness": True, "real_benchmark_canary": "admission=" + adm["A"]["classification"], "terminal": "not_admitted", "activation": False},
        "G1-S1": {"mechanics_robustness": True, "real_benchmark_canary": "admission=" + adm["S"]["classification"], "terminal": "not_admitted", "activation": False},
    }

    null_map = {
        "G1-S1": "learned-model planning worse than random (imperfect model errors compound over the horizon)",
        "G1-A1": "reading affordance from the latent ties a fitted value estimator; no incremental value",
        "G1-C0": "confidence-weighted trace worse than EMA smoothing (wrong direction)",
        "G1-E1": "relational boundaries help but not beyond simple change detectors by SESOI",
        "G1-M1": "message value predictable and architecture-robust but fires on uncertain items where the message does not causally help (noisy-TV)",
        "G1-K1": "value architecture-robust but fires on contradictions not warranting repair, including false-repair harm (noisy-TV)",
        "G1-V1": "real verification value that avoids the U1 noisy-TV trap, but only 1 of 3 capable estimators robustly decodes it (architecture_dependent)",
        "G1-N1": "MNIST canary positive but CIFAR-10 second-source confirmation null",
        "G1-U1": "raw uncertainty fails reducibility discrimination on real data",
        "G1-R1": "no incremental retrieval value beyond nearest-similarity",
        "G1-D1": "retired frozen design",
        "historical_G1-I1": "retired route",
    }

    confirmed_components = {
        "confirmed_to_independent_scientific_replication": [],
        "same_team_cross_architecture_only": ["G1-P1R"] if (P is None or "positive" in str(P.get("classification"))) else ["G1-P1R"],
        "note": "G1-P1R is the only mechanism with any real-data positive. It is same-team cross-architecture (MNIST, "
                "CIFAR-100, KMNIST). Lane P external-method replication classification: "
                + (P["classification"] if P else "pending") + ". No mechanism reached independent scientific confirmation.",
    }

    p_positive = bool(P and P["classification"] in ("external_replication_positive", "external_method_confirmation_positive", "same_team_external_method_positive"))
    n_confirmed = 0  # confirmation-level components (independent or external). P1R same-team cross-arch is not a full confirmation.
    construction_licensed = p_positive or n_confirmed >= 2
    integration_licensed = False  # needs three confirmation-level domains

    architecture_boundary = {
        "current_confirmed_architecture": "G1-P1R (plasticity / replay-value prediction) only, as same-team cross-architecture evidence on MNIST-family and CIFAR-100 and KMNIST",
        "not_externally_replicated": True,
        "external_method_replication_result": (P["classification"] if P else "pending"),
        "excluded_mechanisms": ["G1-D1", "G1-U1", "G1-N1", "G1-R1", "historical_G1-I1"],
        "not_admitted_mechanisms": ["G1-V1 (architecture_dependent)", "G1-K1", "G1-M1", "G1-E1", "G1-C0", "G1-A1", "G1-S1"],
        "clusters_licensed": [],
        "construction_licensed": construction_licensed,
        "integration_licensed": integration_licensed,
        "activation": False,
        "forbidden_claims": [
            "any unopened mechanism is scientifically confirmed",
            "P1R is independently externally replicated (it is not)",
            "any cluster, construction, or integration is evidenced",
            "the architecture is activated",
        ],
    }

    wall = {L: adm[L].get("wall_seconds") for L in adm}
    wall["P"] = P.get("wall_seconds") if P else None
    resource_report = {
        "schema": "mop-generation2-resource/v1", "environment": "macOS CPU (torch 2.13, no CUDA), gymnasium classic-control",
        "lane_wall_seconds": wall, "notes": "Lane P (EMNIST class-incremental, 6 methods) is the dominant cost; "
        "selection lanes train CIFAR CNNs; control lanes use numpy+gym rollouts; temporal lanes train one KMNIST CNN then numpy.",
        "packages_installed_this_campaign": ["pip (ensurepip bootstrap)", "gymnasium 1.3.0"],
        "datasets_acquired": ["EMNIST-balanced (download)"], "datasets_reused": ["MNIST", "KMNIST", "FashionMNIST", "CIFAR-10", "CIFAR-100"],
        "blocked_by_absence": ["torchaudio", "scipy (SVHN/.mat)", "sklearn", "librosa/soundfile (raw audio)"],
    }

    next_frontier = {
        "schema": "mop-generation2-next-frontier/v1",
        "open_scientific_questions": [
            "Does P1R beat the best established replay method (GDumb) under matched memory and compute on a stronger source, using a FAITHFUL per-item replay-value selection rather than an ad-hoc heuristic (Lane P result: "
            + (P["classification"] if P else "pending") + ")",
            "Is V1 verification value decodable by a capable estimator family beyond the single one that worked (architecture_dependent -> capable-family study)",
            "Can K1/M1 be re-scoped so the mechanism fires only on genuinely reducible contradictions/messages (both failed noisy-TV, not the incremental-value clause)",
            "External independent replication of P1R by a second team or from external published code (never yet achieved)",
        ],
        "exact_next_if_any": "If Lane P yields a faithful positive, prepare the Action-Simulation and Verification-Repair-Messaging clusters are still unlicensed (no two components confirmed); the only near-term licensed successor would be a P1R external independent replication.",
    }

    # write artifacts
    for name, obj in [
        ("MOP_GENERATION2_EVIDENCE_MATRIX.json", {"schema": "mop-gen2-evidence-matrix/v1", "ladder": LADDER, "matrix": evidence_matrix, "source_commit": commit}),
        ("MOP_GENERATION2_NULL_MAP.json", {"schema": "mop-gen2-null-map/v1", "nulls": null_map, "source_commit": commit}),
        ("MOP_GENERATION2_CONFIRMED_COMPONENTS.json", {"schema": "mop-gen2-confirmed/v1", **confirmed_components, "source_commit": commit}),
        ("MOP_GENERATION2_ARCHITECTURE_BOUNDARY.json", {"schema": "mop-gen2-arch-boundary/v1", **architecture_boundary, "source_commit": commit}),
        ("MOP_GENERATION2_RESOURCE_REPORT.json", resource_report),
        ("MOP_GENERATION2_NEXT_FRONTIER.json", next_frontier),
    ]:
        obj["artifact_sha256"] = sha(obj)
        (OUT / name).write_text(json.dumps(obj, indent=2))

    synthesis = {
        "schema": "mop-generation2-frontier-synthesis/v1", "source_commit": commit,
        "program_id": "mop-generation2-scientific-frontier-v1", "branch": "agent/mop-scientific-frontier",
        "governing_graph_status": "reconcile DONE -> admissions DONE (7 lanes) -> canaries (none licensed among the 7) -> Lane P external replication "
        + (P["classification"] if P else "pending") + " -> clusters unlicensed -> construction "
        + ("licensed" if construction_licensed else "unlicensed") + " -> integration unlicensed -> terminal synthesis SEALED",
        "headline": "Of seven unopened mechanisms, none passed real-data admission; the one surviving mechanism (P1R) was "
        "tested against established replay methods on a stronger external source (EMNIST) with result: "
        + (P["classification"] if P else "pending") + ". No cluster, construction, or integration is licensed. Activation stays false.",
        "evidence_classes_distinguished": LADDER,
        "admission_results": {L: adm[L]["classification"] for L in adm},
        "lane_P_result": {k: P.get(k) for k in ("classification", "final_avg_accuracy_mean", "best_established", "p1r_final_avg", "p1r_final_avg_sd", "primary_comparison_per_task_effect", "repair_applied")} if P else "pending",
        "lane_P_adversarial_audit": {"consensus_verdict": P_audit.get("consensus_verdict"), "auditors": P_audit.get("auditors"),
                                     "note": "v1 replication_harm was an unfaithful operationalization (used the raw-loss control as the mechanism, dropped the toxic-value gate); one bounded bed-validity repair applied, Lane P re-run faithfully"} if P_audit else "pending",
        "confirmed_architecture": architecture_boundary["current_confirmed_architecture"],
        "clusters": {"action_simulation": "unlicensed (A1,S1 not admitted)", "verification_repair_messaging": "unlicensed (V1,K1,M1 not admitted)", "event_trace": "unlicensed (E1,C0 not admitted)"},
        "construction": "licensed" if construction_licensed else "unlicensed (no external P1R replication, no confirmed cluster, fewer than two confirmed components)",
        "integration": "unlicensed (fewer than three confirmation-level functional domains)",
        "activation": False,
        "house_rules_honored": ["a tie is a null", "wrong-direction is a failure", "positives adversarially verified",
                                "no git attribution", "at most one bounded repair per lane", "no mechanism-favouring synthetic bed for unavailable real evidence"],
    }
    synthesis["synthesis_sha256"] = sha(synthesis)
    (OUT / "MOP_GENERATION2_FRONTIER_SYNTHESIS.json").write_text(json.dumps(synthesis, indent=2))
    print("synthesis artifacts written. construction_licensed=", construction_licensed, "P=", (P["classification"] if P else None))
    return synthesis


if __name__ == "__main__":
    main()
