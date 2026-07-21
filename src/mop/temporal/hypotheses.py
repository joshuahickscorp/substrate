"""The eight central hypotheses and, for every possible result, what it does to each of them.
Preregistering the mapping is the point. Without it, a factorial produces numbers and then an author decides
afterwards which story they told.
House style: no dashes.
"""
from __future__ import annotations
HYPOTHESES = {
    "H1_recurrence": "a recurrent state transition provides value a stateless model cannot recover, even with "
                     "matched parameter count and explicit causal history",
    "H2_explicit_history": "the core effect is access to historical observations, and a stateless model given "
                           "a matched history reproduces it",
    "H3_capacity": "the core effect is mainly increased representation capacity",
    "H4_state_horizon": "persistent state helps only because it carries information over a minimum useful "
                        "temporal horizon",
    "H5_optimization": "recurrent models appear superior because they optimize more easily under the current "
                       "training authority",
    "H6_core_horizon_interaction": "larger cores help only when state persists sufficiently long",
    "H7_architecture_family": "the effect is specific to one recurrent implementation rather than to "
                              "persistent temporal computation generally",
    "H8_bed_specificity": "the effect is real on the two valid controlled beds and does not transfer to a "
                          "third natural stream",
}
# result key -> what it does to each hypothesis. Every entry is written before any principal cell runs.
PREREGISTERED_MAPPING = {
    "recurrent_beats_matched_history": {
        "supports": ["H1_recurrence"],
        "weakens": ["H2_explicit_history"],
        "closes": [],
        "unresolved": ["H3_capacity", "H4_state_horizon"],
    },
    "matched_history_matches_recurrent": {
        "supports": ["H2_explicit_history"],
        "weakens": ["H1_recurrence"],
        "closes": ["H1_recurrence"],
        "unresolved": ["H4_state_horizon"],
    },
    "capacity_monotonic_and_large": {
        "supports": ["H3_capacity"],
        "weakens": ["H1_recurrence"],
        "closes": [],
        "unresolved": ["H6_core_horizon_interaction"],
    },
    "capacity_flat_or_saturating": {
        "supports": [],
        "weakens": ["H3_capacity"],
        "closes": ["H3_capacity"],
        "unresolved": [],
    },
    "horizon_threshold_at_dependency_length": {
        "supports": ["H4_state_horizon"],
        "weakens": [],
        "closes": [],
        "unresolved": ["H6_core_horizon_interaction"],
    },
    "horizon_flat": {
        "supports": [],
        "weakens": ["H4_state_horizon", "H1_recurrence"],
        "closes": ["H4_state_horizon"],
        "unresolved": [],
    },
    "capacity_helps_only_at_long_horizon": {
        "supports": ["H6_core_horizon_interaction"],
        "weakens": [],
        "closes": [],
        "unresolved": ["H3_capacity"],
    },
    "capacity_and_horizon_independent": {
        "supports": [],
        "weakens": ["H6_core_horizon_interaction"],
        "closes": ["H6_core_horizon_interaction"],
        "unresolved": [],
    },
    "all_recurrent_families_agree": {
        "supports": ["H1_recurrence"],
        "weakens": ["H7_architecture_family"],
        "closes": ["H7_architecture_family"],
        "unresolved": [],
    },
    "one_recurrent_family_dissents": {
        "supports": ["H7_architecture_family"],
        "weakens": ["H1_recurrence"],
        "closes": [],
        "unresolved": ["H5_optimization"],
    },
    "unconverged_arms_explain_the_gap": {
        "supports": ["H5_optimization"],
        "weakens": ["H1_recurrence", "H3_capacity"],
        "closes": [],
        "unresolved": [],
    },
    "converged_everywhere_and_gap_remains": {
        "supports": [],
        "weakens": ["H5_optimization"],
        "closes": ["H5_optimization"],
        "unresolved": [],
    },
    "third_bed_agrees": {
        "supports": [],
        "weakens": ["H8_bed_specificity"],
        "closes": ["H8_bed_specificity"],
        "unresolved": [],
    },
    "third_bed_dissents": {
        "supports": ["H8_bed_specificity"],
        "weakens": [],
        "closes": [],
        "unresolved": ["H1_recurrence"],
    },
    "third_bed_invalid": {
        "supports": [],
        "weakens": [],
        "closes": [],
        "unresolved": ["H8_bed_specificity"],
    },
    "readout_capacity_reproduces_the_effect": {
        "supports": ["H3_capacity"],
        "weakens": ["H1_recurrence"],
        "closes": [],
        "unresolved": [],
    },
    "readout_capacity_flat": {
        "supports": [],
        "weakens": ["H3_capacity"],
        "closes": [],
        "unresolved": [],
    },
}
STATES = ("open", "supported", "weakened", "closed", "unresolved")
def apply(results: list[str]) -> dict:
    """Fold a set of observed result keys into a state per hypothesis. Support and closure both accumulate."""
    state = {h: {"supports": [], "weakens": [], "closes": [], "unresolved": []} for h in HYPOTHESES}
    unknown = [r for r in results if r not in PREREGISTERED_MAPPING]
    for r in results:
        m = PREREGISTERED_MAPPING.get(r)
        if not m:
            continue
        for kind in ("supports", "weakens", "closes", "unresolved"):
            for h in m[kind]:
                state[h][kind].append(r)
    out = {}
    for h, s in state.items():
        if s["closes"]:
            v = "closed"
        elif s["supports"] and not s["weakens"]:
            v = "supported"
        elif s["supports"] and s["weakens"]:
            v = "mixed"
        elif s["weakens"]:
            v = "weakened"
        elif s["unresolved"]:
            v = "unresolved"
        else:
            v = "open"
        out[h] = {"premise": HYPOTHESES[h], "state": v, "evidence": s}
    return {"hypotheses": out, "unknown_result_keys": unknown,
            "observed_results": sorted(set(results))}
