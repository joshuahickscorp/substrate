# ISSUES

Degraded / deferred / expected-failure items, each with reason + unblock step.
Empty of hard failures means a clean run.

## Active local admission blocker (category 1, not a hardware boundary)
- P5 smoke has no scientific run receipt. The current governed preflight failed closed across three
  samples with 7.039 to 7.243 GB available against the unchanged 10.0 GB requirement, while the host
  was on battery power. No child ran. Unblock: connect AC power, free enough local memory for three
  healthy samples, then run the exact command recorded at the top of `STATUS.md`. Categories 8 and 9
  remain empty, so the evidence does not justify a Studio purchase.

## Known trap (2026-07-10)
- `scripts/cache_factorized_encoder.py` (and any compose()-driven cache script) does not
  implement `--help`; the token is swallowed and a full default 288-clip encode starts as a
  side effect. When piped to `head`, the process wedges on a closed stdout pipe after store
  preallocation, leaving an uncitable partial store that turns the doctor red. One such
  runaway was killed and its receiptless partial removed this session. Unblock: add an early
  argparse/usage guard to the compose()-driven scripts; until then read the module docstring
  instead of passing --help.
- `load_encoder` silently returns the `frozen_random` fallback backend when the composed
  config omits `prefer_real`/`require_real`. Any ad hoc script that forgets those overrides
  can produce random-weight outputs labeled by intent as real. The established cache scripts
  refuse this by asserting `enc.spec.backend`; every new consumer must do the same
  (`+encoder.prefer_real=true +encoder.require_real=true` plus a backend assertion). Caught
  fail-closed in the archived inherited-scale identity generator this session.

## Deferred (environment, not a defect)
- Real V-JEPA latent caching: encoder weights not fetched in this session (no model
  download). Synthetic-latent path is operational and all downstream is built/tested on
  it. Unblock: `uv pip install -e ".[encoder]"` + network, then
  `python scripts/cache_latents.py encoder=vjepa2_vitl_fpc64_256 +source=<videos>`.
- Tier R campaign legs (environment rollouts and capstone): queued with enabled=false and require
  the declared real environment. Unblock: connect the existing local action adapter to the exact
  rendered or substrate evidence required by the row, then enable only that registered leg.
- V-JEPA 2.1 dense task data: the official ViT-B checkpoint is retained, hash-verified,
  strict-loaded, and finite at 8 and 64 frames. E6 and DR14 now wait on a rights-clean natural
  tensor manifest plus matched learned/random caches, not model publication or larger variants.
- Real natural-video decode: `substrate/video.py` is backend-agnostic and lazy-imports
  torchvision/decord; neither is a hard dep. Unblock: `uv pip install -e ".[video]"`. The
  preprocessing core is tested today; full decode still requires a configured video environment
  and rights-clean real clips.

## Deferred (acquisition lane, gated by design, not a defect)
- Heavy dataset downloads: every remote source in `registry/datasets.yaml` is DRY-RUN by
  default in `scripts/studio_pipeline.py acquire`. Real bytes need `--execute --budget-gb N`
  and, for any source with terms, `--accept-license`. On the current device the downloader has
  no credentials/fetcher for remote methods, so remote sources record a clean `blocked` status
  (never a partial). Unblock: provide the required credentials and fetch tooling, then run
  `acquire --execute` within the declared budget.
- Manual-license sources (ssv2, ego4d_subset): status `manual`; the planner will not select
  them unless the profile allows manual auth AND `--accept-license` is passed. Unblock: complete
  the signed terms (see each source's `auth_steps` / the license ledger), then re-plan.
- Blocked / deferred sources (laion_tiny_meta blocked; ego4d_full, ego_exo4d_subset deferred):
  full Ego4D and Ego-Exo4D are beyond a 1 TB local disk and are NEVER planned by default; LAION
  is withdrawn pending a safety-reviewed re-release. These are intentional, surfaced in the
  license ledger, and require dedicated storage / a re-release before they could be considered.
- Real remote fetch + archive extraction: the downloader detects unsafe archive members
  (path traversal / absolute paths) but the actual streaming fetcher is an external callback
  (not implemented on this device, by design). Unblock: supply a `fetch_fn` with credentials.

## Degraded
- faiss 1.14 + torch on Apple Silicon: `faiss.search()` HARD-SEGFAULTS (rc 139, dual OpenMP
  runtime) when run after torch is imported, and a segfault cannot be caught in-process.
  MITIGATED, not a live risk: `KVIndex` probes faiss.search safety once in a subprocess
  (`shell.buffer.faiss_search_safe`) and falls back to EXACT brute-force retrieval if unsafe;
  the buffer default index is now `brute`. Correctness is never affected (brute is exact); only
  large-scale retrieval speed. Unblock: re-probe in the target environment with a faiss build that
  does not share the conflicting OpenMP runtime.

## Methodological defect (found by the axis-ceiling meta-control audit, 2026-07-02): GATE FIXED, arm follow-up remains
- FIXED (Round 2, 2026-07-03): the vacuous `needs_real` GATE in `src/mop/diagnostics/substrate_ablation.py` is
  retired. `needs_real` now gates on the shuffled chance floor (`real - shuffled > margin`), the honest
  latent-level meaning (decodable-above-chance, not encoder-specificity, which needs a random-init-ENCODER
  comparison). `frozen_random`/`delta_frozen_random` are kept but truthfully labeled known-vacuous-for-linear
  metrics; a genuinely-lossy `rank_reduced` control was added. Three consumer verdicts flipped, each validated
  correct with NO manufactured positive: `a_perception` A1 null_supported True->False, `s_semiotics` S1
  grounded_index False->True (stricter MI+RSA gate), S10 null_supported True->False. Full gates green.
  Evidence: `runs/mot/falsification_vacuous_fix.diff`, `runs/mot/meta_control_audit.json`. Falsification 9 -> 10.
- REMAINING (scoped follow-up, deferred to keep the suite honest): the SAME class of vacuity one level down,
  where experiments read `frozen_random_projection` DIRECTLY as an ablation arm (S3 `gain_vs_frozen_random`,
  S5, S6 `heldout_above_frozen_random`, plus b_biology/c_cogsci/d6/y4/i_infotheory/n_neuro/p_philosophy). Their
  frozen-random arm is also transparent to their linear metrics, so those nulls partly rest on a vacuous arm.
  Fixing that moves ~10 experiments at once and risks the manufactured-positive failure mode, so it is a
  dedicated reviewed refactor, not a same-session edit. Replace the direct frozen-random arm with the shuffled
  floor and/or the new `rank_reduced` lossy control.

## Expected failures (xfail, kept in the suite)
(none currently; any flaky-on-Metal exact assertion is converted to a tolerance from the
 determinism utility rather than xfailed.)
