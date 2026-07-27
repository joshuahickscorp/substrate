"""Freeze the inherited scientific interpretation before any new work touches it.

Everything here is copied from sealed predecessor artifacts, not restated from memory. Where the mandate text
and the sealed artifact disagree, the artifact wins and the disagreement is recorded rather than smoothed.

House style: no dashes.
"""

from __future__ import annotations

import json

from fastforge.runs import io

INTEGRATED = io.ROOT / "integrated"


def read(name: str) -> dict:
    return json.loads((INTEGRATED / name).read_text())


def main():
    code = read("MOP_INTEGRATED_CODE_ACCOUNTING.json")
    ev = read("MOP_INTEGRATED_EVIDENCE_ACCOUNTING.json")
    nulls = read("MOP_INTEGRATED_NULL_MAP.json")
    forbidden = read("MOP_INTEGRATED_FORBIDDEN_CLAIMS.json")
    rounds = read("MOP_INTEGRATED_FORGE_ROUNDS.json")
    cross = read("MOP_SUBSTRATE_CROSS_DOMAIN_REPORT.json")
    audio = read("MOP_SUBSTRATE_AUDIO_REPORT.json")
    mut = read("MOP_PROOF_SCOPE_MUTATIONS.json")

    ts = rounds["verdicts"]["timescale_contributions"]
    res = rounds["results"]

    authority = {
        "schema": "mop-fast-state-forge-start-authority/v1",
        "successor_of": {
            "branch": "agent/mop-integrated-substrate-forge",
            "commit": "c570b87b96f806d16da9492d5ee6c08c3af854f0",
            "draft_pr": 32,
            "verified_local_and_remote": True,
        },
        "immutable_historical_authorities": {
            "collapse_branch": "agent/mop-accretion-collapse",
            "collapse_pr": 31,
            "rule": "not modified, merged, rebased, reset or interfered with by this program",
        },
        "successor_branch": "agent/mop-fast-state-plasticity-forge",
        "worktree": str(io.ROOT),
        "run_root": f"runs/substrate/{io.PROGRAM}",
        "proof_root": f"proof/substrate/{io.PROGRAM}",
        "stop_switch": "~/.mop_fast_state_forge_stop",
        "inherited_code_and_evidence": {
            "active_runtime_loc": code["active_runtime_loc"],
            "substrate_surface_loc": code["substrate_implementation_loc"],
            "maintained_python_excluding_proof_loc": code["maintained_python_excludes_proof"],
            "mandate_stated_maintained_python": 15788,
            "discrepancy_note": "the mandate states approximately 15,788 maintained Python LOC; the sealed "
            "integrated accounting measures 22,266 excluding proof serialization. The "
            "sealed measurement is binding and the 18,000 target is already exceeded at "
            "inheritance, so this program budgets net new code against 22,266.",
            "acceptance": "11/11",
            "proof_artifacts": ev["artifacts"],
            "proof_unique_objects": ev["unique_objects"],
            "proof_tamper_mutations_all_rejected": mut["mutations"]["all_rejected"],
        },
        "valid_temporal_domains": {
            "HAR_raw": "valid temporal domain",
            "SpeechCommands_v0.02": f"valid after the mel filterbank instrumentation repair "
            f"(gru {audio['gru_correct']}, bag {audio['bag_order_free']}, "
            f"shuffled {audio['gru_shuffled']}, headroom lcb {audio['temporal_headroom_lcb']})",
            "PAMAP2_window_classification": "invalid_no_temporal_headroom",
            "STARSS23_cached_eight_clip": "invalid for the required substrate sequence",
        },
        "closed_premises": {
            "architecture_E": "compression null, terminal",
            "architecture_F": "predictive objective falsified: no objective arm scored highest "
            f"({res['F_no_objective']['util']}) and shuffled time "
            f"({res['F_shuffled_time']['util']}) matched real multi horizon "
            f"({res['F_multi_horizon']['util']})",
            "medium_state": f"decorative, contribution {ts['medium']}",
            "shared_slow_state": f"harmful, contribution {ts['slow']}",
            "multi_horizon_objective": "no downstream benefit",
            "shuffled_time_predictive_objective": "indistinguishable from real time predictive objective",
            "learned_replay_or_retrieval": "closed",
            "learned_simulation": "closed",
            "rule": "these exact premises are not reopened by this program",
        },
        "surviving_signal": {
            "fast_state_contribution": ts["fast"],
            "AT_without_slow_state": {
                "util": res["AT1_no_slow"]["util"],
                "effect_vs_lstm_gdumb": res["AT1_no_slow"]["effect_vs_lstm_gdumb"],
                "status": "strongest substrate arm, lower confidence bound below SESOI 0.05, a near miss "
                "that this program does not reopen by adding seeds",
            },
            "persistent_cross_domain_core": {
                "return_recovery": cross["effects_vs_fresh_core"]["frozen_transferred_core"][
                    "har_return_vs_fresh"
                ],
                "second_domain_acquisition": cross["effects_vs_fresh_core"]["frozen_transferred_core"][
                    "speech_acq_vs_fresh"
                ],
                "verdict": cross["verdict"],
                "known_defect": "the prior matrix aliased arms: projection_only shared a code path with "
                "fresh_core, and slow_core_transfer shared one with fine tuned and full "
                "persistent. Repaired by this program before any new transfer inference.",
            },
        },
        "new_causal_premise": "a substrate can preserve transferable fast temporal dynamics while preventing "
        "cross domain interference by restricting slow change to domain local "
        "parameters and selectively controlling shared core updates",
        "premise_fails_if": "a strong conventional learner with separate domain heads or adapters remains "
        "superior after matched costs",
        "inherited_forbidden_claims": forbidden["forbidden"],
        "activation": False,
    }
    io.seal("MOP_FAST_STATE_FORGE_START_AUTHORITY.json", authority)

    binding = {
        "schema": "mop-fast-state-binding-nulls/v1",
        "binding_rule": "a null listed here is immutable. It may be superseded only by an appended authority "
        "that states the new evidence, never by rewriting or relabelling the original.",
        "inherited_nulls": nulls["nulls"],
        "inherited_null_effect_sizes": {
            k: res[k]["effect_vs_lstm_gdumb"] for k in sorted(res) if k != "lstm_gdumb"
        },
        "cross_domain_arm_effects": cross["effects_vs_fresh_core"],
        "cross_domain_verdict": cross["verdict"],
        "not_reopened": [
            "architecture E compression",
            "architecture F predictive consolidation",
            "shared medium state",
            "shared trainable slow workspace as a proposal",
            "learned replay, retrieval, simulation",
            "PAMAP2 window classification",
            "STARSS23 cached eight clip representation",
            "adding seeds to the A-T no slow near miss",
        ],
        "reopened_and_why": {
            "cross_domain_transfer": "the prior matrix contained aliased arm implementations, so its arm "
            "level attribution was never valid. The historical null verdict stands "
            "unchanged; only the implementation is repaired and rerun as a fixture.",
            "shared_slow_state_as_control": "retained only as a control arm, never as a proposed mechanism",
        },
    }
    io.seal("MOP_FAST_STATE_BINDING_NULLS.json", binding)

    md = f"""# Fast State Plasticity Forge, start authority

Successor of `agent/mop-integrated-substrate-forge` at `c570b87`, draft PR 32. Local and remote both verified
at that commit before any edit. The collapse branch `agent/mop-accretion-collapse` and PR 31 remain untouched
historical authorities.

## What is inherited as fact

* active runtime {code["active_runtime_loc"]} LOC, substrate surface
  {code["substrate_implementation_loc"]} LOC,
  maintained Python excluding proof {code["maintained_python_excludes_proof"]} LOC, acceptance 11/11
* {ev["artifacts"]} indexed proof artifacts, {ev["unique_objects"]} unique objects, every tamper mutation
  rejected
* two inherited temporal domains, HAR raw and Speech Commands, the latter valid only after the mel
  filterbank repair
* PAMAP2 window classification and the cached eight clip STARSS23 representation are invalid beds

## What is closed

Architecture E, Architecture F, shared medium state, shared trainable slow workspace as a proposal, multi
horizon and shuffled time predictive objectives, learned replay, learned retrieval, learned simulation. This
program does not reopen those premises.

## What survives

Fast temporal state is the only supported internal signal, contribution {ts["fast"]}. Medium state is
decorative at {ts["medium"]}. Shared slow state is harmful at {ts["slow"]}. A persistent cross domain core
improved return recovery and harmed second domain acquisition, but the arm level attribution behind that
statement was produced by an implementation that aliased arms, so it is not usable as an arm level fact.

## The new premise

Preserve dynamics, localize plasticity. A small owned fast temporal core may carry reusable dynamics while
domain local parameters absorb change. The premise fails if a strong conventional learner with separate heads
or adapters remains superior after matched costs.

## Discrepancy recorded rather than smoothed

The mandate states approximately 15,788 maintained Python LOC. The sealed integrated accounting measures
{code["maintained_python_excludes_proof"]}. The sealed measurement is binding.

Activation remains false.
"""
    io.seal_md("MOP_FAST_STATE_FORGE_START_AUTHORITY.md", md)
    print("authority sealed at", io.PROOF)


if __name__ == "__main__":
    main()
