# MOP Substrate Moldability Battery

Modability tasks: continual acquisition, retention, return-to-old-context, abrupt/gradual shift, future held-out adaptation, interference, limited-memory/compute adaptation, cross-context and cross-domain transfer. Controls: known-positive (joint upper bound), known-null (naive fine-tune), no-headroom (har_shift), leakage (group-disjoint splits), interference (class-incremental forgetting).

## Calibration (5-seed preflight)
Known-positive PASS (joint dominates naive on all eligible domains). No-forgetting control PASS (har_shift correctly invalid_no_forgetting). Residual headroom beyond the STRONG baseline (GDumb) present on emnist/cifar100/har_class. Caveat: headroom is budget-dependent; all principal arms run at the same budget so any substrate advantage is at matched budget. A tie is a null.
