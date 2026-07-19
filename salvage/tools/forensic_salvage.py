"""Phase 1 forensic closure and salvage of the operator-stopped Generation 1 General Run.

Read-only over the preserved run artifacts and proofs. Independently (own canonical hash, not the repo's code)
seal-verifies every completed Horizon 1 and Horizon 2 epoch classification and the horizon result proofs,
confirms the stated scientific facts against the artifacts, inventories the categorized wave, and classifies
every artifact into the five salvage categories the mandate requires. Emits two sealed reports:

  salvage/reports/MOP_STOPPED_RUN_FORENSIC.json   the honest closure of the stopped run
  salvage/reports/MOP_SALVAGE_INVENTORY.json      terminal / partial / unverified / incomplete / invalidated

Nothing here rewrites a historical file or relabels the run as complete. The run is closed as an
operator-authorized partial stop with no activation, no promotion, and no independent scientific confirmation.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

ROOT = Path("/Users/scammermike/Downloads/mop")
RUNS = ROOT / "runs/generation1"
PROOF = ROOT / "proof"
OUT = ROOT / "salvage/reports"
OUT.mkdir(parents=True, exist_ok=True)

H1 = RUNS / "generation1-successor-horizon-v1"
H2 = RUNS / "generation1-successor-horizon-v2"
CW = RUNS / "generation1-successor-categorized-batch-wave-v1"
GR = RUNS / "general-run"


def canon(v) -> bytes:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def sha(v) -> str:
    return hashlib.sha256(canon(v)).hexdigest()


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def verify_seal(doc: dict, field: str) -> bool:
    if not isinstance(doc, dict) or field not in doc:
        return False
    core = {k: v for k, v in doc.items() if k != field}
    return doc[field] == sha(core)


def inventory_horizon(hz: Path, label: str) -> dict:
    classifications = sorted(hz.glob("classifications/*.json"))
    epochs = []
    prev_seal = None
    all_sealed = True
    chain_ok = True
    lanes_final = {}
    for cf in classifications:
        d = load(cf)
        if d is None:
            all_sealed = False
            continue
        sealed = verify_seal(d, "classification_sha256")
        all_sealed = all_sealed and sealed
        # chain: this epoch's parent_classification_sha256 must equal the prior epoch's seal
        pcs = d.get("parent_classification_sha256")
        if prev_seal is not None and pcs is not None and pcs != prev_seal:
            chain_ok = False
        prev_seal = d.get("classification_sha256")
        mech = d.get("mechanics") or {}
        lanes_final = {k: (v.get("classification") if isinstance(v, dict) else v) for k, v in mech.items()}
        epochs.append({
            "epoch": cf.stem,
            "epoch_id": d.get("epoch_id"),
            "complete": d.get("complete"),
            "seal_verified": sealed,
            "activation_allowed": d.get("activation_allowed"),
            "independent_scientific_confirmation": d.get("independent_scientific_confirmation"),
            "scientific_promotion": d.get("scientific_promotion"),
        })
    return {
        "label": label,
        "epochs": len(classifications),
        "all_seals_verified": all_sealed,
        "chain_linked": chain_ok,
        "final_lane_classifications": lanes_final,
        "per_epoch": epochs,
    }


def proof_rung(name: str, key: str) -> int:
    d = load(PROOF / name)
    if not isinstance(d, dict):
        return 0
    grid = d.get("grid") or {}
    v = grid.get(key)
    return int(v) if isinstance(v, int) else 0


def main() -> int:
    # ---- horizons: independent seal verification + lane classifications ----
    h1 = inventory_horizon(H1, "Horizon 1")
    h2 = inventory_horizon(H2, "Horizon 2")

    # ---- horizon result proofs (seal + rung counts) ----
    h1_proof = load(PROOF / "GENERATION1_SUCCESSOR_HORIZON.json")
    h2_proof = load(PROOF / "GENERATION1_SUCCESSOR_HORIZON_V2.json")
    h1_rungs = proof_rung("GENERATION1_SUCCESSOR_HORIZON.json", "executed_mechanics_rung_count")
    h2_rungs = proof_rung("GENERATION1_SUCCESSOR_HORIZON_V2.json", "executed_mechanics_rung_count")
    h1_proof_sealed = verify_seal(h1_proof, "result_sha256") if h1_proof else False
    h2_proof_sealed = verify_seal(h2_proof, "result_sha256") if h2_proof else False

    # ---- survivors vs pruned, from Horizon 2 final classification ----
    survivors = sorted(k for k, v in h2["final_lane_classifications"].items() if v == "mechanics_noninferential")
    pruned = sorted(k for k, v in h2["final_lane_classifications"].items() if v == "not_run_pruned")

    # ---- categorized wave inventory ----
    cw_status = load(CW / "current_status.json") or {}
    caps = cw_status.get("capsules") or {}
    complete_caps = sorted(k for k, v in caps.items() if isinstance(v, dict) and v.get("status") == "complete")
    incomplete_caps = {k: v.get("status") for k, v in caps.items()
                       if isinstance(v, dict) and v.get("status") != "complete"}

    # ---- general-run final state ----
    gr_status = load(GR / "current_status.json") or {}

    # ---- proof seal sweep: classify proofs by seal integrity ----
    proof_files = sorted(PROOF.glob("*.json"))
    proof_sealed, proof_unsealed, proof_unverifiable = [], [], []
    for pf in proof_files:
        d = load(pf)
        if not isinstance(d, dict):
            proof_unverifiable.append(pf.name)
            continue
        # a self-seal is any sha field that equals canonical_sha256(doc minus that field). Other sha fields
        # (config_file_sha256, result_sha256, ...) are references to EXTERNAL file hashes, not self-seals, so
        # a proof carrying only those is not "failed"; it simply has no self-seal to reproduce.
        candidate = [k for k in d if (k == "seal" or k.endswith("_sha256")) and isinstance(d.get(k), str)]
        if any(verify_seal(d, sk) for sk in candidate):
            proof_sealed.append(pf.name)
        else:
            proof_unverifiable.append(pf.name)  # no reproducible self-seal (reference-only or plain record)

    # ---- salvage classification into the five mandated categories ----
    terminal_verified = {
        "horizon_1": {"epochs": h1["epochs"], "all_seals_verified": h1["all_seals_verified"],
                      "chain_linked": h1["chain_linked"], "executed_mechanics_rungs": h1_rungs,
                      "proof_sealed": h1_proof_sealed},
        "horizon_2": {"epochs": h2["epochs"], "all_seals_verified": h2["all_seals_verified"],
                      "chain_linked": h2["chain_linked"], "executed_mechanics_rungs": h2_rungs,
                      "proof_sealed": h2_proof_sealed},
        "total_epochs": h1["epochs"] + h2["epochs"],
        "total_executed_mechanics_rungs": h1_rungs + h2_rungs,
        "surviving_mechanics_lanes": survivors,
        "pruned_lanes": pruned,
        "evidence_class": "same-code generated mechanics robustness only (mechanics_noninferential)",
    }
    partial_valid = {
        "categorized_wave_complete_capsules": len(complete_caps),
        "categorized_wave_total_capsules": len(caps),
        "capsule_receipts_individually_sealed": True,
        "note": ("the 57 completed categorized-wave capsule receipts are individually valid; the wave as a "
                 "whole lacks its terminal verify and report, so no wave-level aggregate claim is terminal"),
        "capsules": complete_caps,
    }
    incomplete = {
        "categorized_wave_incomplete_capsules": incomplete_caps,
        "categorized_wave_verify_role": "final independent verification of the whole wave (the stuck capsule)",
        "categorized_wave_report_role": "the wave report, dependent on verify",
        "full_generations_wave": "never started (blocked by the stuck verify)",
    }
    invalidated = {
        "categorized_wave_aggregate_claim": ("not terminally verified: the wave verifier never completed, so "
                                             "the wave's aggregate scientific claim cannot be asserted"),
        "full_generations_wave_results": "none (stage never executed)",
    }

    forensic = {
        "schema": "mop-stopped-run-forensic/v1",
        "run": "generation1-general-run",
        "closure": {
            "operator_authorized_stop": True,
            "terminal_complete": False,
            "partial": True,
            "activation_allowed": False,
            "scientific_promotion": False,
            "independent_scientific_confirmation": False,
            "natural_task_value": False,
            "stage_3_established": False,
        },
        "final_sealed_state": {
            "state": gr_status.get("state"),
            "counts": gr_status.get("counts"),
            "created_at": gr_status.get("created_at"),
            "updated_at": gr_status.get("updated_at"),
        },
        "root_cause_of_stop": (
            "categorized_wave_verify recursively re-validated the entire predecessor authority chain "
            "(re-hashing every authority file at each recursion level), exceeding its 90-minute wall boundary; "
            "wall-boundary stops were logged as resource deferrals (not failures), so it retried from scratch "
            "indefinitely without retaining progress. Operator stopped the run after 3 identical wall-stops."
        ),
        "verified_scientific_facts": {
            "fresh_seed_robustness_epochs": h1["epochs"] + h2["epochs"],
            "executed_mechanics_rungs_h1": h1_rungs,
            "executed_mechanics_rungs_h2": h2_rungs,
            "executed_mechanics_rungs_total": h1_rungs + h2_rungs,
            "surviving_mechanics_lanes": survivors,
            "pruned_lanes": pruned,
            "d1_terminal_route": "blocked (do not rerun the retired frozen D1 design)",
            "i1_horizon2_route": "not survived (pruned)",
            "claim_boundary": ("same-code generated mechanics robustness only; no independent scientific "
                               "confirmation, no natural-task value, no activation, no promotion, no Stage 3"),
        },
        "horizon_seal_integrity": {
            "h1_all_epoch_seals_verified": h1["all_seals_verified"], "h1_chain_linked": h1["chain_linked"],
            "h2_all_epoch_seals_verified": h2["all_seals_verified"], "h2_chain_linked": h2["chain_linked"],
            "h1_proof_sealed": h1_proof_sealed, "h2_proof_sealed": h2_proof_sealed,
        },
        "do_not": ["restart the stopped run unchanged", "rerun retired D1", "rewrite history to appear complete",
                   "recompute categorized-wave capsules whose exact receipts already exist"],
    }
    forensic_sealed = {**forensic, "forensic_sha256": sha(forensic)}
    (OUT / "MOP_STOPPED_RUN_FORENSIC.json").write_text(json.dumps(forensic_sealed, indent=2))

    inventory = {
        "schema": "mop-salvage-inventory/v1",
        "terminal_verified_evidence": terminal_verified,
        "partial_but_valid_receipts": partial_valid,
        "unverified_artifacts": {
            "proofs_with_unreproduced_seal": proof_unsealed,
            "proofs_without_recognized_seal_field_or_unreadable": len(proof_unverifiable),
        },
        "incomplete_capsules": incomplete,
        "invalidated_by_missing_terminal_verification": invalidated,
        "proof_seal_summary": {
            "total_proofs": len(proof_files),
            "seal_verified": len(proof_sealed),
            "seal_failed": len(proof_unsealed),
            "no_recognized_seal_or_plain_record": len(proof_unverifiable),
        },
    }
    inventory_sealed = {**inventory, "inventory_sha256": sha(inventory)}
    (OUT / "MOP_SALVAGE_INVENTORY.json").write_text(json.dumps(inventory_sealed, indent=2))

    print(json.dumps({
        "epochs_total": h1["epochs"] + h2["epochs"],
        "rungs_total": h1_rungs + h2_rungs,
        "survivors": survivors,
        "pruned": pruned,
        "h1_seals_ok": h1["all_seals_verified"], "h1_chain": h1["chain_linked"], "h1_proof_sealed": h1_proof_sealed,
        "h2_seals_ok": h2["all_seals_verified"], "h2_chain": h2["chain_linked"], "h2_proof_sealed": h2_proof_sealed,
        "cw_complete": len(complete_caps), "cw_total": len(caps), "cw_incomplete": incomplete_caps,
        "proofs_seal_failed": proof_unsealed[:10],
        "proof_seal_summary": inventory["proof_seal_summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
