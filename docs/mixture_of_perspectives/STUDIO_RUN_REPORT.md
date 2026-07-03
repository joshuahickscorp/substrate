# STUDIO RUN REPORT: the accumulating scoreboard for the goal loop

This is the single accumulating artifact the goal loop (`STUDIO_GOAL_PROMPT.md`) writes each wave, so
a dead session loses at most one wave and the next session resumes from here. House style: no em or en
dashes. A tie is a null. No positive enters a doc without an independent adversarial verification pass.

IMPORTANT PROVENANCE: waves 0 to 3 in the log below ran on the M3 Pro laptop (18 GB), NOT the M1 Ultra
Mac Studio. They executed exactly the goal-loop work that is feasible without the Studio's resources
(the facet-12 predictor lane, the b5 closure, the machine-agnostic half of WAVE 0). The Studio-specific
WAVE-0 steps are marked PENDING below and MUST run on the M1 Ultra before the spine. A session on the
real Studio resumes from the "Next move" section.

## WAVE 0 status (partial: machine-agnostic half done on M3 Pro)

- [x] Full gates green on this box: acceptance 10/10 (full pytest suite, ruff lint + format, mypy 139
      files, E1 forget-then-retain gate, diagnostics, I4 table, campaign queue dry-run, one toy Tier-C
      leg, registry 109 experiments). check_docs green.
- [x] STUDIO_RUN_REPORT.md created (this file) and added to `scripts/check_docs.py` CANONICAL_MD.
- [x] Encode microbench, M3-Pro baseline: MPS vs single-worker CPU on 8 real clips (V-JEPA 2 ViT-L,
      64 frames / 256 px, transformers path). RESULT: CPU single-worker 13.69 s/clip (stable, 8 clips);
      MPS ran but took 821 s/clip on one clip (about 60x slower, memory-pressured at 18 GB). NOTE: the
      hard "Invalid buffer size" MPS overflow documented in STUDIO_HANDOFF did NOT reproduce this
      session on this path; MPS instead completed but paged badly at 18 GB. CPU is the clear M3-Pro
      winner. The Studio MUST re-measure MPS at 128 GB and 14 to 16 parallel CPU workers (128 GB
      removes the paging pressure and could make MPS the winner: MEASURE, never assume).
- [ ] PENDING (M1 Ultra): rebuild the 64-clip real cache to >= 1000 clips. The 64-clip cache
      (`data/cache/vjepa2_vitl_fpc64_256_real`, count 64, acc 1.0) is present and sufficed for all
      M3-Pro-feasible science; the >= 1000-clip rebuild needs the Studio's encode throughput.
- [ ] PENDING (M1 Ultra): the Studio-scale MPS-vs-16-worker-CPU microbench above at 128 GB.

WAVE 0 is therefore COMPLETE for the machine-agnostic half and its two remaining steps are the two
genuinely Studio-gated measurements. A session on the M1 Ultra completes those, then proceeds to the
spine.

## Axis scoreboard (the four north-star axes)

| Axis | Laptop proven | Studio ceiling | Status |
|---|---:|---:|---|
| Falsification | 10 | 10 | Held and completed (vacuous gate retired, 4 over-claims killed). B5 multi-seed re-encode is the one Studio caveat. |
| Abstraction | 6 | 9 | Four controlled wins (systematicity, pairwise + 3-way cross-substrate analogy), all on the shape factor. Beyond 6 needs DR1 real video (gates b1/b2/b3). |
| Density | 6 | 9 | Matched-compute mixture win on a constructed task (10/10 seeds). Natural-complementarity at scale needs DR1. |
| Moldability | 5 | 8 | Frozen-encoder-capped (proven). PR9 real stream then Process C are the levers. |
| Overall (mean) | ~6.75 | ~9.0 | Studio theoretical ~9.3 including the Part-2 frontier. |

Source of the laptop numbers and their adversarial proofs: `AXIS_CEILING_RESULT.md`,
`HANDOFF.md` verdict ledger.

## Facet scoreboard (Part 2, the Studio-native frontier)

| Facet | Score | Status |
|---|---:|---|
| 12 world-model rollouts | 0 | MEASURED and walled provisionally on the M3 Pro (`ROLLOUT_LANE_RESULT.md`): the frozen predictor carries a real but sub-usable rollout signal; motion-tracking is a wall/null-by-ill-posedness on synthetic near-static clips. Licensed re-test: real moving video + a readout adapter. |
| 13 closed-loop / active | 3 | Not started; needs live-encoder throughput (Studio). |
| 14 real corpora | 0 | Not started; needs 8 TB disk + licensing (Studio). |
| 15 perspective ecology | 2 | Not started; needs 10+ resident encoders (128 GB). |
| 16 developmental daemon | 0 | Not started; needs week-scale always-on (Studio). |
| 17 trainable capacity | n/a | A doctrine decision, deliberately not graded. |

## Open levers (all now Studio-gated; the M3-Pro-feasible backlog is CLOSED)

| Lever | Axis | Resource gate |
|---|---|---|
| DR1 real bound-attribute video cache (+ caption acceptance gate, local-VLM arm) | abstraction, density, coverage, plurality | 21 s/clip encode throughput, real corpora, 8 TB disk. THE spine unblocker. |
| PR9 continual-backprop on a real long stream (certificate-guarded) | moldability | 128 GB-resident long stream; decides Process C licensing. |
| Process C (1 to 10M object-centric module on dense tokens) | moldability | dense cache + GPU training. Gated by PR9 kill-switch. |
| Dense-token cache (~1.7 TB) | facet 8 | 8 TB disk. |
| Encoder-scale atlas (ViT-H/g + DINOv2 + VideoMAEv2) | facet 7 | disk + 128 GB residency; pull ViT-H/g stubs. |
| B5 multi-seed re-encode + 30-seed retrofits | falsification, statistical power | pure compute at Studio scale; LAST per doctrine. |
| Facets 13 to 17 (closed-loop, corpora, ecology, daemon) | Part-2 frontier | ride the spine's artifacts; each Studio-gated. |

M3-Pro-feasible backlog, CLOSED this session: facet 12 measured (waves 1 to 2); the pre_studio
candidate positives resolved (e7_sparse architectural, ex5 Adam-artifact refuted, ex2 promoted, b5
underpowered null, wave 3); no open non-vacuous frozen-random gap remains (`DOCTRINE_SYNTHESIS.md`).

## s/clip benchmarks

| Path | s/clip | Notes |
|---|---|---|
| M3-Pro CPU, single worker | 13.69 | 8 real clips, V-JEPA 2 ViT-L, 64 frames / 256 px, transformers path; stable ~13.5 s each. |
| M3-Pro MPS | ~821 | ran on one clip (no hard buffer error this session), 60x slower than CPU, memory-pressured at 18 GB. |
| Studio (M1 Ultra), 14 to 16 CPU workers | PENDING | ~2 s/clip aggregate projected off 13.69 single-worker; MEASURE. |
| Studio (M1 Ultra), MPS at 128 GB | PENDING | 128 GB removes the paging pressure; could win; MEASURE, never assume. |

## Wave log

- WAVE 0 (M3 Pro, 2026-07-03): gates 10/10, this scoreboard created, encode microbench recorded; the
  1000-clip rebuild and the 128 GB MPS microbench remain PENDING for the M1 Ultra. Artifact: this file.
- WAVE 1 (M3 Pro, 2026-07-03, commit d4abb5a): facet 12 DR13 predictor-fidelity on the real V-JEPA 2
  predictor. VERDICT NULL on usability (real beats all 3 controls by seed CI at every horizon but by
  only 5 to 7 percent; sub-usable). Instrument `scripts/mop_dr13_predictor_fidelity.py`. Adversarially
  verified 6/6. `ROLLOUT_LANE_RESULT.md`.
- WAVE 2 (M3 Pro, 2026-07-03, commit f447707): facet 12 decodability-retention (does the rollout track
  object position, beating persistence?). VERDICT WALL / null-by-ill-posedness (synthetic clips move
  sub-patch; the build's motion gate was falsified by the adversarial re-derive). New finding: position
  survives the rollout in a shifted sub-space (readout-adapter target). `ROLLOUT_LANE_RESULT.md` s11.
- WAVE 3 (M3 Pro, 2026-07-03, commit 4c818e4): stack-informed ORIENT audit + closed b5_degeneracy (the
  last open non-vacuous frozen-random gap) as an underpowered null. Net verdict: axis-moving M3-Pro
  science is exhausted; every remaining axis-mover is Studio-gated. `DOCTRINE_SYNTHESIS.md`,
  `STUDIO_HANDOFF.md` update.

- WAVE 4 (M3 Pro, 2026-07-03, turnkey plan Tier 1 to 2): de-risked the Studio spine before it spends
  Studio time. DR1 caption acceptance gate smoke (`scripts/studio/dr1_smoke.py`, passes on carried
  factors, refuses on a non-recoverable one); PR9 + atlas re-smoked on HEAD (reinit fires, honest
  nulls, atlas withholds on partial caches); the studio rehearsal passes 9/9 stages on HEAD; and
  preregistration frozen with 7 NULL/survival cards (facet 12, b5, ex2, e7, ex5, ex13, ex15). Plan and
  running checklist: `STUDIO_TURNKEY_PLAN.md`.
- WAVE 5 (M3 Pro, 2026-07-03, turnkey plan Tier 3 to 4): facet-12 readout-adapter scaffold
  (`scripts/mop_dr13_readout_adapter.py`, `--clip-dir` real video), preregistered + smoke-run: the
  adapter halves the visible-slot representational gap but does not transfer to the rollout (naive
  visible-slot adapter insufficient; Studio fits on rollout predictions). ViT-H / ViT-g encoder
  readiness verified (configs correct, weights are stubs to pull, vitg needs a prefer_real flag);
  turnkey pull + flip steps recorded in STUDIO_TURNKEY_PLAN.md. Remaining: T4.1 encode auto-select,
  T5.1 facet entry points (lower priority, ride the spine).

- WAVE 6 (M3 Pro, 2026-07-03, potential re-audit + gap closure): an adversarial re-audit put honest
  M3-Pro completion at 86 percent (science ~90, enablement ~72) and caught one over-claim (the facet-12
  verifier was cited but absent from the repo) plus a cross-cutting durability hole (the evidence base
  was 100 percent gitignored). Closed: the load-bearing verdict evidence is now git-tracked via a
  targeted .gitignore negation; `scripts/mop_dr13_verify.py` restored (6/6 PASS in-repo); Tier 4.1 encode
  auto-select built (`scripts/mop_encode_autoselect.py`); PR9 bare-smoke cache fallback and the stale
  rehearsal-stage phrasing fixed. Only axis-moving science (Studio-gated) and the T5 facet stubs remain.

## Next move (per the goal-loop protocol)

On the M1 Ultra, in order:
1. Finish WAVE 0: run the MPS-vs-16-worker-CPU encode microbench at 128 GB (record s/clip + winner);
   rebuild the real cache to >= 1000 clips; commit.
2. SPINE, standing order: DR1 first (the single highest-leverage artifact; its caption acceptance gate
   prevents wasted encode), PR9 second (certificate-guarded; decides Process C), dense cache + atlas
   encode ride the same conveyor, B5 multi-seed and seed retrofits LAST.
3. One Part-2 item is already run (facet 12, walled provisionally); the licensed re-test rides DR1's
   real moving video plus a readout adapter. The other Part-2 lanes (13 to 17) ride the spine artifacts.

Every wave: ORIENT (reread this scoreboard), SELECT highest axis-delta-per-hour, PREREGISTER in code,
BUILD + RUN inside `studio-m1ultra`, VERIFY (independent adversarial pass on every positive), LEDGER
here + commit, STOP-CHECK. No wave ends without a committed artifact.
