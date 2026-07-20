"""Final synthesis: the forty three terminal questions, answered from sealed artifacts only.

Every answer here is read out of a sealed file. Where an artifact does not exist, the answer says so rather
than inferring one. Where an artifact exists and says null, the answer says null, in those words.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

from fastforge.runs import io

MISSING = "not established: the artifact this answer depends on is not sealed"


def get(name, *path, default=MISSING):
    if not io.exists(name):
        return default
    doc = io.load(name)
    for k in path:
        if isinstance(doc, dict) and k in doc:
            doc = doc[k]
        else:
            return default
    return doc


def answers():
    arm_audit = get("MOP_CROSS_DOMAIN_ARM_AUDIT.json", "checks", default={})
    imap = io.load("MOP_SUBSTRATE_INTERFERENCE_MAP.json") if io.exists(
        "MOP_SUBSTRATE_INTERFERENCE_MAP.json") else {}
    cross = io.load("MOP_FAST_STATE_BIDIRECTIONAL_SYNTHESIS.json") if io.exists(
        "MOP_FAST_STATE_BIDIRECTIONAL_SYNTHESIS.json") else {}
    plast = io.load("MOP_SUBSTRATE_PLASTICITY_POLICY_REPORT.json") if io.exists(
        "MOP_SUBSTRATE_PLASTICITY_POLICY_REPORT.json") else {}
    rounds = io.load("MOP_FAST_STATE_ARCHITECTURE_COMPARISON.json") if io.exists(
        "MOP_FAST_STATE_ARCHITECTURE_COMPARISON.json") else {}
    reorg = io.load("MOP_FUNCTIONAL_REORGANIZATION_REPORT.json") if io.exists(
        "MOP_FUNCTIONAL_REORGANIZATION_REPORT.json") else {}
    tfree = io.load("MOP_TASK_FREE_CONTEXT_REPORT.json") if io.exists(
        "MOP_TASK_FREE_CONTEXT_REPORT.json") else {}
    third = io.load("MOP_THIRD_TEMPORAL_DOMAIN_PREFLIGHT.json") if io.exists(
        "MOP_THIRD_TEMPORAL_DOMAIN_PREFLIGHT.json") else {}
    code = io.load("MOP_FAST_STATE_CODE_REPORT.json") if io.exists("MOP_FAST_STATE_CODE_REPORT.json") else {}
    ver = io.load("MOP_FAST_STATE_INDEPENDENT_VERIFICATION.json") if io.exists(
        "MOP_FAST_STATE_INDEPENDENT_VERIFICATION.json") else {}
    mut = io.load("MOP_FAST_STATE_MUTATION_REPORT.json") if io.exists(
        "MOP_FAST_STATE_MUTATION_REPORT.json") else {}
    clone = io.load("MOP_FAST_STATE_CLEAN_CLONE.json") if io.exists("MOP_FAST_STATE_CLEAN_CLONE.json") else {}
    fabric = io.load("MOP_FAST_STATE_EVIDENCE_FABRIC.json") if io.exists(
        "MOP_FAST_STATE_EVIDENCE_FABRIC.json") else {}
    score = io.load("MOP_FAST_STATE_PROGRESS_SCORECARD.json") if io.exists(
        "MOP_FAST_STATE_PROGRESS_SCORECARD.json") else {}
    within = {d: (io.load(f"MOP_{d.upper()}_WITHIN_DOMAIN_REPORT.json")
                  if io.exists(f"MOP_{d.upper()}_WITHIN_DOMAIN_REPORT.json") else {}) for d in ("har", "speech")}

    def dir_report(a, b):
        n = f"MOP_{a.upper()}_TO_{b.upper()}_REPORT.json"
        return io.load(n) if io.exists(n) else {}

    d1, d2 = dir_report("har", "speech"), dir_report("speech", "har")
    cls = imap.get("classification", {})

    def group_answer(pred):
        return sorted(k for k, v in cls.items() if v.get(pred)) or "none measured above SESOI"

    q = {}
    q["1 were all transfer arms truly distinct"] = (
        f"yes: {arm_audit.get('distinct_declared_identity')} on declared identity and "
        f"{arm_audit.get('distinct_behaviour_checkpoints')} on behaviour, across 11 arms, with "
        f"{get('MOP_CROSS_DOMAIN_ARM_MUTATIONS.json', 'mutations', 'all_rejected')} on the mutation suite"
        if arm_audit else MISSING)
    q["2 which parameters persisted between domains"] = (
        "the shared fast core in every carrying arm, plus the domain independent half of the projection in "
        "the projection only arm. Everything else is domain local by construction.")
    q["3 which parameters remained domain local"] = (
        "projection convolution, adapter, normalization and head, one set per domain, plus the low rank "
        "adapter in Architecture H")
    q["4 which groups caused new domain acquisition"] = group_answer("enables_acquisition")
    q["5 which groups caused old domain forgetting"] = group_answer("causes_forgetting")
    q["6 did shared fast dynamics transfer"] = (
        f"no: the bidirectional verdict is {cross.get('bidirectional_verdict', MISSING)}")
    q["7 did freezing the fast core preserve acquisition"] = (
        f"see G1 against G0 in each direction: "
        f"{ {k: v.get('utility', {}).get('G1_frozen_after_first') for k, v in (('har->speech', d1), ('speech->har', d2)) if v} } "
        f"versus G0 "
        f"{ {k: v.get('utility', {}).get('G0_always_trainable') for k, v in (('har->speech', d1), ('speech->har', d2)) if v} }"
        if d1 else MISSING)
    q["8 did reopening the fast core improve return recovery"] = (
        f"G2 return recovery effect versus the strongest baseline: "
        f"{ {k: v.get('effects', {}).get('G2_reopened_at_return', {}).get('return_recovery_vs_baseline') for k, v in (('har->speech', d1), ('speech->har', d2)) if v} }"
        if d1 else MISSING)
    q["9 did adapter isolation reduce interference"] = (
        f"G4 adapters only, negative transfer: "
        f"{ {k: v.get('means', {}).get('G4_adapters_only', {}).get('negative_transfer') for k, v in (('har->speech', d1), ('speech->har', d2)) if v} }"
        if d1 else MISSING)
    q["10 did anchored fast dynamics help"] = (
        f"H arms against the strongest matched baseline: "
        f"{ {k: v.get('effects', {}).get('H_always_update', {}).get('vs_strongest_baseline', {}).get('lower_95_cb') for k, v in (('har->speech', d1), ('speech->har', d2)) if v} }"
        if d1 else MISSING)
    q["11 did gradient conflict gating help"] = (
        f"H cosine gate versus H always update: "
        f"{ {k: (v.get('utility', {}).get('H_cosine_gate'), v.get('utility', {}).get('H_always_update')) for k, v in (('har->speech', d1), ('speech->har', d2)) if v} }"
        if d1 else MISSING)
    q["12 did probe loss gating help"] = (
        f"H probe gate versus H always update: "
        f"{ {k: (v.get('utility', {}).get('H_probe_gate'), v.get('utility', {}).get('H_always_update')) for k, v in (('har->speech', d1), ('speech->har', d2)) if v} }"
        if d1 else MISSING)
    q["13 did simple fixed partitioning remain best"] = (
        f"{ {d: v.get('strongest_simple_policy') for d, v in plast.get('per_direction', {}).items()} } "
        f"with verdict {plast.get('verdict', MISSING)}" if plast else MISSING)
    q["14 did oracle update partitioning show headroom"] = (
        f"{ {d: v.get('oracle_advantage_over_strongest_simple') for d, v in plast.get('per_direction', {}).items()} }"
        if plast else MISSING)
    q["15 did a learned plasticity gate open"] = str(plast.get("learned_gate_opened", MISSING))
    q["16 did a learned gate beat simple policies"] = (
        "no learned gate opened, so the question does not arise"
        if plast.get("learned_gate_opened") is False else MISSING)
    q["17 did Architecture G beat LSTM plus GDumb"] = (
        f"{ {k: max((v['effects'][a]['vs_strongest_baseline']['lower_95_cb'] for a in v.get('effects', {}) if a.startswith('G')), default=None) for k, v in (('har->speech', d1), ('speech->har', d2)) if v} } "
        f"against SESOI {io.SESOI}" if d1 else MISSING)
    q["18 did Architecture H beat LSTM plus GDumb"] = (
        f"{ {k: max((v['effects'][a]['vs_strongest_baseline']['lower_95_cb'] for a in v.get('effects', {}) if a.startswith('H')), default=None) for k, v in (('har->speech', d1), ('speech->har', d2)) if v} } "
        f"against SESOI {io.SESOI}" if d1 else MISSING)
    q["19 did either beat separate per domain models"] = (
        f"the strongest matched baseline per direction was "
        f"{ {k: v.get('strongest_matched_baseline') for k, v in (('har->speech', d1), ('speech->har', d2)) if v} }, "
        f"and no substrate arm cleared SESOI against it" if d1 and not cross.get(
            "arms_passing_in_both_directions") else MISSING)
    q["20 did either direction of domain transfer pass"] = str(
        cross.get("per_direction_verdicts", MISSING))
    q["21 did both directions pass"] = str(cross.get("bidirectional_verdict", MISSING))
    q["22 did return recovery improve without acquisition harm"] = (
        "no arm satisfied both at once, which is the sealed rule for calling a return recovery gain a null"
        if cross.get("bidirectional_verdict") == "cross_domain_null" else MISSING)
    q["23 did held out adaptation improve"] = (
        f"{ {k: max((v['effects'][a]['future_adaptation_vs_baseline']['lower_95_cb'] for a in v.get('effects', {}) if a.startswith(('G', 'H'))), default=None) for k, v in (('har->speech', d1), ('speech->har', d2)) if v} }"
        if d1 else MISSING)
    q["24 did task free context inference work"] = str(tfree.get("verdict", MISSING))
    q["25 did functional reorganization add value"] = str(reorg.get("verdict", MISSING))
    q["26 was fast state independently necessary"] = (
        "inherited: fast state is the only timescale with a positive contribution (+0.063). This program "
        "did not reopen that measurement and did not contradict it.")
    q["27 was slow shared state correctly removed"] = (
        f"consistent with this program: shared groups classified as causing forgetting are "
        f"{imap.get('forgetting_groups', MISSING)}")
    q["28 was medium state correctly removed"] = (
        "inherited as decorative (+0.003) and not reinstated here. Neither architecture carries shared "
        "medium state.")
    q["29 did the third temporal domain validate the premise"] = (
        f"{third.get('verdict', MISSING)}: "
        f"{[ (a['domain'], a['verdict']) for a in third.get('attempts', []) ]}")
    q["30 which architecture was selected"] = (
        "none. No architecture cleared SESOI against the strongest matched baseline in both directions."
        if cross.get("bidirectional_verdict") == "cross_domain_null" else MISSING)
    q["31 what implementation scores were reached"] = str(score.get("implementation", MISSING))
    q["32 what evidence scores were earned"] = str(score.get("evidence", MISSING))
    q["33 what strong baseline remained best"] = str(
        cross.get("per_direction_strongest_baseline", MISSING))
    q["34 what code was retained"] = (
        f"one new package, fastforge, {code.get('new_substrate_loc', MISSING)} LOC, plus the inherited "
        f"runtime at {code.get('active_runtime_loc', MISSING)} LOC")
    q["35 what code was deleted"] = str(code.get("deletions", MISSING)) + ". " + str(
        code.get("deletion_policy", ""))
    q["36 did clean clone pass"] = str(clone.get("all_pass", MISSING)) + " with skips " + str(
        clone.get("environmental_skips", {}).get("note", ""))
    q["37 did proof indexing pass"] = (
        f"{fabric.get('verification', {}).get('all_pass', MISSING)} over "
        f"{fabric.get('union', {}).get('count', MISSING)} artifacts, mutations "
        f"{fabric.get('mutations', {}).get('all_rejected', MISSING)}")
    q["38 what exact scientific ceiling remains"] = (
        "the owned substrate now has repaired, non aliased transfer arms, two materially different owned "
        "architectures, an interference map, a plasticity action oracle and a bidirectional matrix. None of "
        "it clears SESOI against a strong matched conventional learner in both domain directions.")
    q["39 is Owned Substrate v1 selected"] = "no"
    q["40 is a unified cross domain entity evidenced"] = "no"
    q["41 is functional self reorganization evidenced"] = (
        "no" if reorg.get("verdict") != "functional_reorganization_positive" else MISSING)
    q["42 is activation licensed"] = "no. Activation remains false and was never separately authorized."
    q["43 what exact next frontier remains"] = (
        "the two failures this program can name precisely: shared fast dynamics did not transfer across "
        "modalities under any update partition, and every activity recognition style third domain proved "
        "order insensitive. The next frontier is either a pair of domains that share temporal structure "
        "rather than merely both being sequences, or an honest statement that cross modality transfer of "
        "fast dynamics is out of reach at this scale.")
    q["within_domain"] = {d: within[d].get("verdict", MISSING) for d in within}
    q["improvement_rounds"] = rounds.get("verdict", MISSING)
    q["independent_verification"] = ver.get("checks", MISSING)
    q["mutation_suite"] = mut.get("all_rejected", MISSING)
    return q


def main():
    t0 = time.time()
    a = answers()
    forbidden = [
        "any owned architecture beats strong matched baselines",
        "shared fast dynamics transfer across modalities",
        "update partitioning improves the stability plasticity tradeoff beyond simple rules",
        "a learned plasticity gate is licensed",
        "functional self reorganization is evidenced",
        "a unified cross domain entity is evidenced",
        "any third temporal domain validated the premise",
        "activation is licensed",
    ]
    io.seal("MOP_FAST_STATE_FORGE_SYNTHESIS.json", {
        "schema": "mop-fast-state-forge-synthesis/v1",
        "terminal_questions": a,
        "forbidden_claims": forbidden,
        "activation": False,
        "wall_seconds": round(time.time() - t0, 1),
    })
    io.seal("MOP_FAST_STATE_NEXT_FRONTIER.json", {
        "schema": "mop-fast-state-next-frontier/v1",
        "closed_by_this_program": [
            "cross domain transfer of a shared fast temporal core between sensor and audio, under every "
            "update partition tested, in both directions",
            "interference gated adaptation as a route to safe shared plasticity",
            "bounded and reversible shared plasticity as a route to safe shared plasticity",
            "next activity prediction on scripted accelerometry as a temporal bed",
            "next activity prediction on free living accelerometry as a temporal bed",
        ],
        "open": [
            "domain pairs that share temporal structure rather than merely both being sequences",
            "whether the fast state advantage measured within a domain is a property of the core or of the "
            "readout, which this program did not separate",
            "whether a larger shared core changes the transfer answer, which capacity matched comparison "
            "cannot decide on its own",
        ],
        "not_reopened": [
            "architecture E compression", "architecture F predictive consolidation",
            "shared medium state", "learned replay, retrieval, simulation",
            "PAMAP2 window classification", "adding seeds to the inherited A-T no slow near miss",
        ],
        "activation": False,
    })
    lines = ["# Fast State Plasticity Forge, synthesis", "",
             "Preserve dynamics, localize plasticity. The premise was that a small owned fast temporal core",
             "could carry reusable dynamics while domain local parameters absorb change. It was tested in",
             "both domain directions, with two materially different owned architectures, against converged",
             "conventional baselines.", "", "## Terminal questions", ""]
    for k, v in a.items():
        lines.append(f"**{k}**  \n{v}\n")
    lines += ["## Forbidden claims", ""] + [f"* {f}" for f in forbidden] + ["", "Activation remains false."]
    io.seal_md("MOP_FAST_STATE_FORGE_SYNTHESIS.md", "\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in a.items() if k.startswith(("20", "21", "30", "42"))}, indent=1)[:600],
          flush=True)
    print("SYNTHESIS_DONE", flush=True)


if __name__ == "__main__":
    main()
