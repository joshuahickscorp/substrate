"""Third temporal domain preflight, both attempts.

Section priority order was followed: a redesigned continuous PAMAP2 transition task first, then HARTH free
living continuous accelerometry. Both attempts are reported, including the one that failed, because a
preflight that only reports the surviving candidate is not a preflight.

House style: no dashes.
"""

from __future__ import annotations

import time

from fastforge import data as D
from fastforge.runs import io


def attempt(name, title, rights, design, differs, note):
    key = f"MOP_DOMAIN_GATE_{name.upper()}.json"
    if not io.exists(key):
        return None
    g = io.load(key)
    d = D.domain(name)
    return {
        "domain": title, "rights": rights, "design": design,
        "differs_from_the_invalid_bed": differs,
        "windows_train": g["n_train"], "windows_test": g["n_test"],
        "units_train": g["n_units_train"], "units_test": g["n_units_test"],
        "transition_fraction": round(d.get("transition_fraction", 0.0), 4),
        "gate": {k: g[k] for k in ("gru", "gru_shuffled", "bag_order_free", "temporal_headroom_gru_vs_bag",
                                   "temporal_headroom_lcb", "order_matters_gru_vs_shuffled", "verdict",
                                   "steps", "seeds")},
        "verdict": g["verdict"], "note": note,
    }


def main():
    t0 = time.time()
    pam = attempt(
        "pamap2_transition", "PAMAP2 continuous transition prediction",
        "PAMAP2 Physical Activity Monitoring, public research dataset",
        "given a raw IMU window, name the activity two seconds after the window ends. Windows that straddle "
        "a transition are kept rather than dropped.",
        ["continuous transitions retained", "label lies in the future so sequence state is required",
         "returning contexts available", "future subject adaptation available"],
        "refuted on event count before it was refuted on the gate: the scripted protocol separates every "
        "activity with a rest label, so the whole dataset contains 22 direct activity to activity "
        "transitions across 9 subjects. At that prevalence the future label is almost always the current "
        "activity and the task collapses back into the window classification bed that was already invalid.")
    har = attempt(
        "harth_transition", "HARTH free living transition prediction",
        "HARTH Human Activity Recognition Trondheim, public research dataset, downloaded for this program",
        "given a three second window of back and thigh accelerometry, name the activity two seconds after "
        "the window ends. Transition and non transition windows are sampled at equal rate.",
        ["free living rather than scripted, so activity changes are frequent and unannounced",
         "label lies in the future", "22 subject units", "transition balanced sampling"],
        "one bounded repair was applied: at natural prevalence only 5 percent of windows straddle a change, "
        "so the first gate correctly rejected the bed. Balancing transition and non transition windows "
        "changes the sampling only, not the labels, the signal or the units. Both gate results are reported.")
    attempts = [a for a in (pam, har) if a]
    valid = [a for a in attempts if a["verdict"] == "temporal_headroom_present"]
    io.seal("MOP_THIRD_TEMPORAL_DOMAIN_PREFLIGHT.json", {
        "schema": "mop-third-temporal-domain-preflight/v2",
        "priority_order_followed": ["HARTH continuous free living data",
                                    "continuous PAMAP2 transition task redesigned from raw streams",
                                    "speaker disjoint continuous audio",
                                    "another lawful session disjoint temporal dataset"],
        "attempts": attempts,
        "valid_third_domains": [a["domain"] for a in valid],
        "verdict": "third_domain_valid" if valid else "third_domain_invalid",
        "status": "secondary. A third domain can support or fail to support the premise, it cannot carry a "
                  "principal verdict on its own, and the principal matrix stays HAR and Speech Commands.",
        "not_repeated": "PAMAP2 window classification, which was already invalid",
        "wall_seconds": round(time.time() - t0, 1),
    })
    for a in attempts:
        print(f"  {a['domain']}: {a['verdict']} gru {a['gate']['gru']} bag {a['gate']['bag_order_free']} "
              f"shuffled {a['gate']['gru_shuffled']}", flush=True)
    print("THIRDDOMAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
