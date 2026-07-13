# EDCM-1 to ESCS adapter audit

Status: deterministic mechanics adapter, activation-disabled
Evidence ceiling: EDCM-1 structured generated mechanics only
Scientific promotion: forbidden

## Boundary

`mop.escs.edcm_adapter` is a strict compatibility layer, not a new experiment and not a runtime
activation path. It converts the already-structured, future-blind EDCM-1 v3 interface into native ESCS
contracts without treating EDCM as evidence for raw event formation, useful learned dispatch, natural
perception, or an integrated efficiency advantage.

The deterministic factual mapping is:

1. `VisibleObservation` becomes one factual `ObservationEvent` and one zero-confidence
   `OBSERVED_CANDIDATE` `HypothesisEvent` that explicitly abstains from activation.
2. A clean bounded `PreparedDecision`/`Resolution` becomes typed `ClaimMessage` records, an
   `ActionIntent`, and an `EXTERNAL_ACTION` `CommitmentEvent` recorded before any consequence.
3. A later `VisibleTransition` becomes a `ConsequenceEvent` bound to the exact commitment. A translated
   successor observation may be included as an additional causal parent.

All translated records stay on `branch:factual` and retain `scripted-mechanics-only` evidence taint.
Counterfactual lesions, delayed-channel controls, dropped or reordered messages, evaluator transitions,
and simulated branches do not cross this adapter.

## Strictness

The adapter rejects incompatible or extra schemas; malformed content hashes; changed world, event,
specialist, referent, provenance, action, or commitment identity; unknown specialists; duplicate or
noncanonical coalitions; more than the fixed initial round plus one verifier round; nonfinite values;
ambiguous JSON with duplicate fields or nonfinite constants; symlink authority inputs; message-byte
disagreement; and future/evaluator-only fields such as hidden change points, action rotation, physical
action, niche/noise labels, future consequences, oracle labels, or ground truth.

EDCM's `pre_bus_state_sha256` remains an evaluator-side fork-audit value. Its form is checked, but it is
neither used as actor state nor copied into factual ESCS provenance. The clean resolution's delay-control
cache is accepted only when it exactly equals the current planner message, then excluded from the factual
translation.

## Authority and activation

Activation requires a terminal, nonexploratory, self-hashed `mop-edcm1-receipt/v3` and a terminal
`mop-edcm1-verification-artifact/v1` produced by full deterministic regeneration. The loader joins the
verification artifact to the exact producer and checkpoint bytes and revalidates the current official
config and implementation manifest/files. A failed complementarity gate is an invalid bed, not a routing
null.

Those checks are necessary but deliberately insufficient: `ADAPTER_ACTIVATION_ENABLED` is frozen to
`False`, every activation assessment contains `adapter-activation-disabled`, and `activate()` always raises.
No official EDCM producer or verification artifact existed while this adapter was built, so the current
activation assessment also lacks verified result authority.

## Accounting

Every translation returns a self-hashed `TranslationAccounting` record. It preserves every EDCM
`AbstractWork` component and the exact official weight vector, recomputes the source work total, records
canonical source and target byte sizes, and charges each source and target byte once for serialization and
once for hashing plus explicit field-validation operations. The resulting native ESCS `WorkVector` must
equal source work plus adapter work exactly. The record can append one hash-linked `LifecycleCharge` to a
caller-supplied ledger. Retention after translation remains a caller-owned byte-time charge; it is not
silently inferred.

These are deterministic semantic counters, not FLOPs, energy, latency, or proof that Python execution was
fully metered by hardware.

## Validation

Focused tests cover deterministic observation, claim, commitment, and consequence translation; ESCS event
ledger replay; ESCS receiver-side claim validation; lifecycle-work reconciliation; self-hash tampering;
message lesions and factual mutations; evaluator leakage; bounded rounds; current-authority schema failure;
failed complementarity; and activation refusal with or without a verified authority object. No official
experiment is launched by the module or its tests.
