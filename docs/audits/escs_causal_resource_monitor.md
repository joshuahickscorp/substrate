# ESCS causal/resource monitor adapter audit

Status: activation-disabled mechanics scaffold; no official run or proof receipt.

## Boundary

`src/mop/escs/causal_resource_monitor.py` is a pure downstream observer of canonical
`EventLedger.payload()` and `LifecycleLedger.payload()` snapshots. It reconstructs both ledgers, verifies
their hash chains and cross-ledger event references, and refuses over-cap, future-dated, evaluator-bearing,
or malformed snapshots before deriving a claim. It never appends an event or charge and retains no state
between calls.

The adapter emits only two inert `ClaimMessage` types:

- `resource_anomaly`: an exact abstract-work or retained-byte-time threshold observation with snapshot,
  charge, and event provenance;
- `same_parent_resource_contrast`: a factual/simulated hypothesis pair with the same immutable parent set,
  the same self-hashed two-arm pairing contract, observational/resource-control arm labels, and directly
  attributed lifecycle work. Its payload says explicitly that this is a simulated counterfactual candidate,
  not a realized causal effect.

Both claim types use the shared ESCS message integrity and evidence-taint mechanics. They carry zero
calibrated confidence, no predicted utility, immediate expiry, and the complete monitor/config/control
authority needed to reproduce their bytes. Observation-only resource work cannot be promoted into a claim
without same-branch hypothesis provenance.

## Authority and information controls

Activation, dispatch, commitment, mutation, external effect, retry, resource relief, and independent-trigger
authority are all `false` in the config, result, and every claim payload. Scientific promotion and official-run
status are also false. Construction rejects attempts to enable activation or promotion, and the result exposes
no action method.

The monitor reads envelope topology, declared pairing provenance, event kind/branch/time/evidence, charge
provenance, and exact abstract work counters. It does not branch on semantic event payload values. Snapshot
bytes still bind output hashes, and the recursive evaluator/future key rejection is a schema guard rather than
a proof against evaluator data hidden in arbitrary values; a real experiment still requires a source-isolated
sanitized projection. Any event or charge ending after the caller-supplied `observed_through_tick` invalidates
the call, and every charge must follow its referenced event and obey the factual/counterfactual branch rule.
This establishes internal tick consistency, not independent proof that monitoring was prospective.

## Exact controls

Every control is deterministic, provenance-bearing, bounded, and remains nonpromotable:

- `noisy`: adds `1 + int(charge_sha256[:8], 16) % noise_ceiling` abstract work to each in-window sample;
- `poison`: adds the exact configured increment to the lexicographically smallest in-window charge digest;
- `stale`: moves the analysis frontier back by the configured tick count and records every excluded current
  row;
- `shuffle`: rotates both abstract work and retained-byte-time values left by one across in-window charges
  while retaining both target and source charge digests. Fewer than two rows is refused, so the shuffle
  cannot silently become identity.

Separately, poisoned ledger bytes or reordered ledger entries fail canonical replay before control logic runs.

## Bounds and accounting

Default limits are 128 event entries, 128 lifecycle entries, 1 MiB per snapshot, a 32-tick horizon, eight
claims, sixteen provenance IDs per claim, 128 same-parent comparisons, and 4,096 abstract monitor work units.
Hard module ceilings prevent a caller from configuring unbounded retained state, scans, or pair comparisons.
The reported work units are a declared deterministic accounting proxy, not Python instruction, hashing, or
serialization counts. Retained byte-time stays separate from operation-like work.

The result reports monitor work as `indexing_and_graph_maintenance` with
`monitor_work_charge_applied=false`. A future caller must append that charge exactly once if the monitor is
ever integrated; this scaffold has no ledger-mutation authority and makes no no-gap accounting claim.

## Evidence and nonclaims

The design inherits only the mechanics boundary of `proof/P9_CAUSAL_MONITORING_PREFLIGHT.json` and
`docs/P9_CAUSAL_MONITORING_PREFLIGHT.md`: bounded, prospective-visible telemetry and same-parent controls.
It does not inherit P9's structural-fixture result as evidence for ESCS, natural workloads, physical failures,
causal benefit, capability, efficiency, cognition, sentience, or energy. ESCS canonical records remain
provenance rather than faithful causal explanations without registered real interventions.

Focused verification:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider \
  tests/unit/test_escs_causal_resource_monitor.py
10 passed
```

The tests cover deterministic input immutability, shared claim validation, false authority, future/evaluator
rejection, entry/work bounds, exact noisy/poison/stale/shuffle transforms, poisoned/reordered snapshot
rejection, lifecycle branch/time joins, same-parent and pair-contract necessity, provenance abstention, and
nonidentity shuffle enforcement.
