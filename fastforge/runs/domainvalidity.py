"""Domain validity authority: the temporal order gate for every domain this program touched.

Includes the bounded repair to the order free control. The first control reused the projection whose Conv1d
has a kernel width of five, so it read local temporal order and was not order free. It then outscored every
recurrent model on both principal domains, which would have been read as those domains being order
insensitive. The defect was in the control. Both the pre repair and post repair numbers are recorded.

House style: no dashes.
"""

from __future__ import annotations

import time

from fastforge.runs import io

DOMAINS = ["har", "speech", "har_stream", "speech_stream", "pamap2_transition", "harth_transition"]
PRE_REPAIR = {
    "pamap2_transition": {"gru": 0.7855, "bag_order_free": 0.7888, "gru_shuffled": 0.7924},
    "harth_transition": {"gru": 0.6919, "bag_order_free": 0.7015, "gru_shuffled": 0.707},
    "speech_stream": {"gru": 0.7238, "bag_order_free": 0.3139, "gru_shuffled": 0.2381},
}


def main():
    t0 = time.time()
    gates = {}
    for d in DOMAINS:
        key = f"MOP_DOMAIN_GATE_{d.upper()}.json"
        if io.exists(key):
            g = io.load(key)
            gates[d] = {
                k: g[k]
                for k in (
                    "gru",
                    "gru_shuffled",
                    "bag_order_free",
                    "temporal_headroom_gru_vs_bag",
                    "temporal_headroom_lcb",
                    "order_matters_gru_vs_shuffled",
                    "verdict",
                    "steps",
                    "seeds",
                    "n_train",
                    "n_test",
                    "n_units_train",
                    "n_units_test",
                    "channels",
                    "classes",
                    "unit",
                )
            }
            if d in PRE_REPAIR:
                gates[d]["pre_repair_control"] = PRE_REPAIR[d]
    for g in gates.values():
        g["beats_order_free_control"] = g["temporal_headroom_lcb"] >= 0.03
        g["order_matters"] = g["order_matters_gru_vs_shuffled"] > 0.02
        g["status"] = (
            "temporal"
            if g["beats_order_free_control"] and g["order_matters"]
            else "marginal: order matters but an order free reader is nearly as good"
            if g["order_matters"]
            else "order insensitive"
        )
    valid = [d for d, g in gates.items() if g["verdict"] == "temporal_headroom_present"]
    marginal = [d for d, g in gates.items() if g["status"].startswith("marginal")]
    io.seal(
        "MOP_DOMAIN_VALIDITY.json",
        {
            "schema": "mop-domain-validity/v1",
            "marginal_domains": marginal,
            "marginal_meaning": "shuffling time costs the recurrent model real accuracy, so the bed is not "
            "order insensitive, but an order free reader gets close enough that the bed cannot carry a "
            "strong claim about temporal dynamics at this capacity. Any result on a marginal bed is "
            "reported with that bound attached.",
            "inherited_status_note": "HAR raw was validated as a temporal bed by the inherited authority "
            "under a different estimator family. Under this program's owned architecture and a genuinely "
            "order free control it is marginal. The inherited record is not rewritten; this is an "
            "additional measurement under a new authority.",
            "gate_rule": "a bed is temporal only if the recurrent estimator beats an order free control "
            "with a "
            "lower confidence bound of at least 0.03, and beats itself on shuffled time by more "
            "than 0.02. Both must hold.",
            "bounded_repair": {
                "what": "the order free control reused the projection, whose Conv1d kernel of width five "
                "reads "
                "local temporal order",
                "why_it_mattered": "the defective control outscored every recurrent model on both principal "
                "domains, which would have been read as the domains being order insensitive",
                "fix": "BagProjection, a per timestep linear map with no temporal operator, so pooling over "
                "time is genuinely order invariant",
                "what_did_not_change": "the shuffled time comparison, which never depended on the control "
                "and "
                "which is what actually decided the two accelerometry beds",
            },
            "gates": gates,
            "valid_domains": valid,
            "invalid_domains": [d for d in gates if d not in valid],
            "principal_domains": [d for d in ("har", "speech") if d in valid],
            "wall_seconds": round(time.time() - t0, 1),
        },
    )
    for d, g in gates.items():
        print(
            f"  {d:20s} {g['verdict']:28s} gru {g['gru']:.4f} bag {g['bag_order_free']:.4f} "
            f"shuffled {g['gru_shuffled']:.4f} order {g['order_matters_gru_vs_shuffled']:+.4f}",
            flush=True,
        )
    print("DOMAINVALIDITY_DONE", flush=True)


if __name__ == "__main__":
    main()
