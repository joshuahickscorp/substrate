# Independent re-verification of the completion-claim surface (2026-07-10)

Session-authored audit narrative. This document is NOT a generated proof: it is the record of an
independent sibling session that RE-EXECUTED the verdict-determining computations behind the prior
completion claims, rather than re-hashing receipts. Where a number below is load-bearing it was
recomputed from raw inputs in scratch (no repo file was written during verification); the one
durable code artifact is the CM7 recomputation regression test named at the end.

Method: eight receipt chains were re-executed in parallel by independent verification agents under
hard guardrails (no repo writes, light lane only, one live heavy P4 job respected). Standard:
recompute the numbers that decide each verdict from the rawest retained input, compare to the
canonical receipt, and attack the verifier itself (mutation tests) where safe.

## Verdicts: 8 of 8 chains REPRODUCED

1. CM7 objective-selection null (the branch-retiring result). From the raw 25-score table alone,
   all 12 paired comparisons, Holm-adjusted p-values, simultaneous Bonferroni-t lower bounds
   (t-critical agreement to 13 significant figures via independently implemented math), winner
   identity, all three killing gates, compute matching (max fractional FLOP deviation 7.74e-5),
   and promotes=false reproduce exactly. Chain hashes match; the composite embeds the raw receipt
   byte-identically; a scratch-mutated raw receipt is rejected at three separate code points.
   The null is stronger than "winner fails to clear": both untrained controls beat every learned
   objective (random_target 0.725, frozen_random 0.709, best learned 0.684). Airtight.
2. F-series fresh-seed verifiers. F10's refutation reproduces bit-for-bit (fresh-seed transfer
   mean 0.0067, CI [-0.0354, 0.0488], sign flip); F4, F13, F18 passes reproduce exactly from fresh
   re-executed experiments at the verifier seeds; contract audit re-run in-process, 20/20.
   Coverage limit: 4 of 12 verifier experiments re-executed; the other 8 checked via receipt flags
   and shared machinery only.
3. Wave E0. Fully re-executable: regeneration from seeds to scratch is BYTE-IDENTICAL to the
   canonical receipt (composite 94603d36 reproduced), so all 72 metric booleans re-derive from
   scratch. One mutation rejection re-demonstrated live.
4. P6/P7/P9 preflights + throttle fail-closed dry-run. P6 preflight regenerated to scratch
   byte-for-byte (payload 83e7af3a). P7 numbers re-derived from embedded per-arm data. P9
   deterministic workload evidence reproduced exactly; its host timings are one-shot and
   unverifiable by re-execution in principle, as the receipt itself scopes. Throttle denial
   reproduces fail-closed (telemetry-dependent reason count differs as expected).
5. Extended-compute requirements matrix. Driver --check re-runs the full build and byte-compares:
   passed first run. 291 rows, cat8=0, cat9=0, zero hardware_required independently recounted.
6. Exhaustion + frontier ledgers. 177-row ledger rebuilt end-to-end in memory from raw inputs with
   zero entry diffs; frontier consistent; zero measured-hardware rows.
7. Historical confirmatory closures (e7 sparse, ex2 planning, ex5 local rules). Exactly
   re-executed (full scale, outputs redirected to scratch): e7 reproduced including every per-seed
   number. Their receipts still embed no source hashes, so they bind to current sources, not to
   the historical sources that first produced them; that limitation stands.
8. Completion-claim accounting. 162+33+4=199 checkboxes re-counted at the frozen commit with the
   same legend exclusion; the 14 groups form an exact duplicate-free partition; class totals match.

## Defect ledger (none verdict-affecting today; all worth closing)

D1. proof/COMPLETION_CLAIM_AUDIT.json is hand-authored (no generator) and UNTRACKED in git. Its
    self_verification is self-asserted. Roughly 70-80 percent of its fields are mechanically
    derivable and should be driver-ized; until then it should at minimum be committed so its
    contents are version-bound.
D2. scripts/custom_substrate_finalize.py line 466 hardcodes all_ok true in the CM7 independent
    verifier regardless of gate outcomes. Not verdict-affecting (promotion and problems[] carry
    the real signal) but the field overstates what it attests. Fix requires a coordinated receipt
    regeneration because the chain binds the script hash; queued, not patched piecemeal.
D3. scripts/verify_expansion_wave0.py skips its 9-mutation battery whenever base errors are
    non-empty and matches rejections by substring. Same coordinated-regeneration caveat as D2.
D4. All 13 records in the requirements gate-evidence catalog carry empty gate_roles, so no
    existing evidence object could ever satisfy the typed cat-8/cat-9 gates. Today this biases
    against a Studio conclusion (conservative in the safe direction), but it means the hardware
    gate is not currently satisfiable by any receipt; the role taxonomy needs wiring.
D5. proof/FORM_SUBSTRATE/SCORECARD.json still includes refuted F10 capability numbers in its
    density accounting, and its leg-level all_ok means "obligations complete", not "claim
    verified"; the naming invites over-reading (the honest state lives in verdict_gate_ok and
    PRE_STUDIO_BOUNDARY.json).
D6. The exhaustion class freshly_executed_verified is mechanically a receipt-read (manifest
    structure, status ok, finite metrics), not a re-execution, and carries no timestamp bound;
    e5_curiosity's fresh evidence lacks the orchestrator attempt receipt and used a seed that
    differs from the documented reproduction command.
D7. proof/VJEPA_SCALE_ATLAS_LOCAL.json carries uncommitted working-tree edits that strengthen a
    retired receipt's controls post-hoc (matched_stimulus_hashes false to true, limitation text
    removed). The strengthening reflects real verification work recorded in STATUS.md, but it was
    applied by editing a generated receipt rather than regenerating through its driver, and it is
    unversioned. Either regenerate through the driver or document the edit in the receipt itself.
D8. The encoder retirement (configs/legacy_encoder/, deletion of the vitg/vith selectors) and the
    claim-audit artifact exist only as uncommitted working-tree state; the retirement is real but
    unversioned until the in-flight session commits.
D9. scripts/local_execution_throttle.py list subcommand crashes on dataclass serialization
    (decide/run unaffected; P4's live leg is untouched). Do not patch while the governed P4 leg is
    running; the fix is queued for the lane gap.
D10. The three closure scripts hardcode canonical output paths; a naive re-run silently overwrites
    canonical receipts. They need an --out flag before anyone re-runs them casually.

## What this means for the standing goal

The receipt discipline held under independent re-execution: positives, nulls, and refutations all
reproduce, hardware necessity is zero everywhere, and the two dishonesty-shaped events found
(post-hoc receipt strengthening, hand-authored audit) are process defects, not result defects.
The program state is what the atlas says it is: weighted realization 5.94 of 10, zero facets at
10, one heavy-lane item running (P4 at 40 of 60 seed-runs at re-verification time), fifteen of
sixteen local queue items pending, and no receipt anywhere earning a Studio boundary. Local
resources are NOT yet exhausted; the boundary claim remains honestly unearned.

Durable artifact added with this audit: tests/unit/test_cm7_independent_recompute.py, a regression
gate that re-derives the CM7 verdict-determining statistics from the raw workbench receipt with
independent math and fails if the canonical chain ever drifts from what the raw data supports.
