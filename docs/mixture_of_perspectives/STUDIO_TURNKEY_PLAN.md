# STUDIO TURNKEY PLAN: make the M1 Ultra runs one-command, de-risked, and maximalist

The M3 Pro cannot run the Studio science (no real corpora, no 128 GB residency, no 1.7 TB dense cache),
but it CAN make the Studio's first days turnkey: every Studio script SMOKE-VALIDATED end-to-end here so
it cannot crash on expensive Studio time, preregistration FROZEN before any Studio result exists, and
one-command entry points for the whole spine. This plan is that work. House style: no em or en dashes.
A tie is a null. Every Studio positive still gets an independent adversarial verification pass.

Governing principle: the worst Studio outcome is a script that crashes on wave 1 after a multi-hour
encode. Every item below either (a) exercises a Studio script's plumbing at smoke scale on the laptop so
that crash happens HERE for free, or (b) freezes a preregistration so a Studio result cannot be
back-fit. Priority = de-risk before decorate.

## Status (2026-07-03, M3 Pro)

- [x] T1.1 DR1 smoke: `scripts/studio/dr1_smoke.py` proves the caption acceptance gate PASSES on
      captions carrying both factors (score 1.0 vs chance 0.5) and REFUSES when a factor is not
      caption-recoverable. The spine's binding pre-encode gate is de-risked.
- [x] T1.2 PR9 + atlas re-smoked on HEAD: PR9 fires reinit (reinit_count_total 28, any_zero_reinit
      false) and returns the honest inadmissible-NULL on the tiny stream; atlas runs and WITHHOLDS the
      universal claim on incomplete caches rather than faking it.
- [x] T1.3 Studio rehearsal on HEAD: 9/9 stages pass (doctor, video_corpus, source_validation,
      decode_preprocess, cache_creation, cache_integrity, full_grid_dryrun, miniature_campaign,
      microbench). The whole workflow is validated against current HEAD.
- [x] T2.1 Preregistration frozen: 7 NULL/survival cards authored from real run data (facet 12, b5,
      ex2 survival, e7 survival, ex5, ex13, ex15), all north_star-clean, ledgered in check_docs.
- [ ] T3.1 facet-12 real-corpora + readout-adapter scaffold (next).
- [ ] T4.1 encode auto-select; T4.2 ViT-H/g encoder configs.
- [ ] T5.1 missing-facet entry points.

## Tier 1: de-risk the spine scripts (highest leverage)

- T1.1 DR1 smoke. `scripts/studio/dr1_curate_bound_video.py` is the spine's #1 script and has NO smoke
  mode (PR9 and atlas do). Add a `--smoke` flag that bypasses the >= 32 GB RAM guard and the
  one-encoder pgrep guard and swaps a tiny stub encoder for V-JEPA, then run the FULL pipeline on a tiny
  real-frame fixture: curate -> caption acceptance gate -> encode leg (stub) -> merge shards
  (clip_stems / clip_cells sidecars) -> A6 residual guard. Also run the NEGATIVE fixture (a factor NOT
  caption-recoverable) to confirm the gate REFUSES (a tie is a null), so the gate is proven to gate.
- T1.2 Re-smoke PR9 and atlas against current HEAD; fix any drift (they carry smoke modes already).
- T1.3 Re-run the `studio_pipeline.py` local-max rehearsal on HEAD (last validated at 112053b); confirm
  12/12 stages still pass so the conveyor is trusted before Studio time.

## Tier 2: freeze preregistration (zero compute, pure honesty)

- T2.1 Author the missing NULL_CARDS: the ex-series (ex1, ex4, ex5, ex6, ex7, ex11, ex13, ex14, ex15,
  ex18), the refuted candidate positives, b5_degeneracy, and facet 12. Each card states the preregistered
  null, the required control, the decision threshold, and the honest current verdict, so no Studio result
  can be back-fit to a moved goalpost. Wire the registry `null_card` pointers to the new files.

## Tier 3: scaffold the re-tests this session discovered

- T3.1 Facet-12 real-corpora + readout-adapter script (the licensed re-test from ROLLOUT_LANE_RESULT.md
  section 11): extend the decodability test with a `--clip-dir` real-video path and an ADAPTER arm (fit a
  linear map predictor-space -> encoder-space on visible slots, apply to rollout), preregistered and
  smoke-tested, so the Studio runs the facet-12 licensed re-test with one command on real moving video.

## Tier 4: encode path and conveyor polish

- T4.1 Encode auto-select. The WAVE-0 microbench measured CPU 13.69 s/clip vs MPS 821 s/clip (paged at
  18 GB) on the M3 Pro. Wire the encode path to run a tiny microbench and pick the winner automatically,
  so the Studio does not hand-choose the device (and re-measures MPS at 128 GB where it may win).
- T4.2 Verify the ViT-H / ViT-g encoder configs and the acquisition commands are Studio-ready (facet 7
  atlas encoder-scale curve); both are config-only stubs today.

## Tier 5: one-command entry points for the missing facets (13 to 17)

- T5.1 Scaffold the cheapest missing facet entry points with preregistered nulls and matched controls,
  so the Studio has a runnable stub per facet rather than a design note. Lowest priority: the spine
  (DR1, PR9) must land first and these lanes ride its artifacts.

## Order and gates

Tier 1 first (de-risk), then Tier 2 (freeze), then Tier 3, 4, 5. Every change ships lint + type + test +
docs green and a plain commit. The Studio inherits: smoke-proven spine scripts, a frozen preregistration,
turnkey re-test scripts, an auto-selecting encode path, and this plan as the running checklist. Progress
is tracked in `STUDIO_RUN_REPORT.md`.
