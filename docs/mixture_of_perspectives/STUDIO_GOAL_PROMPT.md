# STUDIO GOAL PROMPT: legacy 8 TB envelope scenario

> SUPERSEDED 2026-07-10: do not execute this historical prompt. Use
> `MOP_MAXIMUM_POTENTIAL_GOAL.md`. No current receipt establishes a Studio boundary.

This is a cold-start prompt written for the provisional 128 GB / 8 TB Studio envelope. It does not prove
that an M1 Ultra was purchased or delivered. Before using it, run the strict doctor against the selected
profile and replace every assumed hardware fact with the resulting measured receipt. It is deliberately
a GOAL prompt, not a task list: it sets a bar above
the honest expected value (the theoretical ceiling from `STUDIO_POTENTIAL_AUDIT.md`, both parts) and wraps
the work in an explicit wave loop so every session converges on either a converted bet or a proven wall,
never on drift.

On the Studio the whole thing is one command: type `/goal` in a Claude Code session at the repo root (the
repo ships `.claude/commands/goal.md`, which reads the full handoff stack, this file included, then executes
the fenced block below). Pasting the fenced block verbatim as the opening message works identically where
slash commands are unavailable. The block is about 3,990 characters, under the 4,000 limit. House style
everywhere: no em or en dashes.

Design notes (why each clause exists):
- The goal is set at >= 9.0 overall knowing the expected value is nearer 8. The laptop rounds proved the
  pattern: five successive stop-checks against "good enough" produced 2.75 points and six wins. The
  unrealistic bar is the mechanism, the honesty rules are the brake.
- The goal has TWO halves on purpose: complete the inherited program (audit Part 1) AND exploit the box as a
  new instrument (audit Part 2, facets 12 to 17: predictor rollouts, hosted corpora, the full perspective
  ecology, the long-run daemon). Without the second half the Studio degenerates into the laptop's executor,
  which the audit names as a failure mode.
- The doctrine-re-derivation clause exists because the laptop's rules (live-encoder ban, cached-only, six
  perspectives) were DERIVED from 21 s/clip and 18 GB, not chosen; keeping them unexamined on this box would
  be inherited superstition.
- The loop is ORIENT, SELECT, PREREGISTER, BUILD+RUN, VERIFY, LEDGER, STOP-CHECK. Preregistration before
  compute and independent adversarial verification before any doc write are the two clauses that killed
  every over-claim on the laptop (four in the axis-ceiling rounds alone). They are non-negotiable at scale.
- Wave 0 is pinned because the two riskiest unknowns are boring: does the transfer checklist actually
  reproduce green gates on the new box, and does MPS lift the M3 Pro buffer wall at 128 GB (measure, never
  assume; `STUDIO_HANDOFF.md` records the laptop failure).
- The DR13 predictor-fidelity exception is the one Part 2 item licensed ahead of the spine: it is an
  afternoon of compute and it gates an entire lane (facet 12), so sequencing it late would be pure waste.
- STUDIO_RUN_REPORT.md is the single accumulating artifact so a dead session loses at most one wave. It must
  be added to `scripts/check_docs.py` CANONICAL_MD when created (wave 0), or the docs gate fails the commit.
- The standing order (DR1 before multi-seeding, B5 last) is the audit's anti-comfort rule; it survived three
  audits and stays.

```text
You are on an Apple Silicon host whose measured doctor receipt satisfies the selected Studio resource
envelope, repo at ~/mop. Read in order before any work: docs/mixture_of_perspectives/HANDOFF.md, STUDIO_POTENTIAL_AUDIT.md
(BOTH parts: inherited program AND Studio-native frontier), EXPAND_PHASE_PLAN.md, STUDIO_HANDOFF.md
(transfer checklist). Hard rules: no em or en dashes anywhere; never attribute Claude in git;
verify `import mop` works outside the checkout without PYTHONPATH; enforce the doctor-verified profile;
preregister every null and verdict
threshold IN CODE before the run; a tie is a null; every candidate positive gets an INDEPENDENT adversarial
verification pass before it may be written into any doc; a faked or unverified score is failure; a PROVEN
wall with a mechanistic reason is success.

THE GOAL, set above expected value on purpose (aim at it, report honestly against it): drive the program
from the proven laptop ceiling (~6.75/10) to >= 9.0 overall (mean of falsification hold-10, abstraction
>= 9, density >= 9, moldability >= 8), disposition >= 70 of 86 semantic positions (tested or walled with a
named reason), AND stand up or wall all four Studio-native lanes: predictor rollouts, hosted real corpora,
the 10-perspective ecology, the developmental long-run daemon. This box is a NEW instrument, not the
laptop's executor: any doctrine rule that exists only because of a laptop constraint (live-encoder ban,
cached-only, six perspectives) gets re-derived or retired with a ledger note. You will likely land short;
the mandate is that every lever in BOTH parts of the audit ends CONVERTED or WALLED, so whatever number
remains is proven.

WAVE 0, once, before science: run the STUDIO_HANDOFF transfer checklist; full gates green on this box (ruff
format + check, mypy, pytest, check_docs, acceptance); create STUDIO_RUN_REPORT.md (scoreboard: axis scores,
open levers, s/clip benchmarks, wave log) and add it to check_docs CANONICAL_MD; microbench encode on 8 real
clips, MPS vs 14-16 CPU workers (the M3 Pro MPS buffer wall may not reproduce at 128 GB: MEASURE, record
s/clip and the winner); rebuild the 64-clip real cache to >= 1000 clips; commit.

THEN LOOP until the goal is met or every lever is walled. Each wave: (1) ORIENT: reread the scoreboard; list
open levers with expected axis-delta per wall-clock hour. (2) SELECT the highest-leverage lever. Standing
order: DR1 before any multi-seeding, PR9 second, dense-cache + atlas encode ride the same conveyor, B5 and
seed-retrofits LAST; never refine an owned number while an unbuilt instrument blocks an axis. Studio-native
lanes ride the spine's artifacts (corpora feed DR1, predictor and perspectives encode the same referents,
the daemon inherits PR9's stream); one early exception: the cheap predictor-fidelity test (DR13 on real
rollouts) gates the rollout lane, run it in the first waves. (3) PREREGISTER in code: null, thresholds,
controls (matched random-init encoder at matched resolution, never a square projection; shuffle floor;
matched compute; no-sign-flip rule). (4) BUILD + RUN inside studio-m1ultra: 250 GB free-disk kill-switch,
resumable checkpoints every 30 min on long jobs, nohup + progress log. (5) VERIFY: independent adversarial
pass on every positive (the laptop killed four over-claims this way); unverified = null. (6) LEDGER: update
STUDIO_RUN_REPORT.md and the HANDOFF verdict ledger, full gates, commit plain. (7) STOP-CHECK: two waves
with no axis movement and no wall proof -> escalate (PR9 kill-switch fires -> Process C 1-10M pilot
licensed; B/C: scale DR1) or write the wall proof and close the axis. DECISION GATES (EXPAND_PHASE_PLAN.md):
DR1 caption gate BEFORE encode spend; PR9 certificate-guarded tie -> moldability dead at frozen substrate;
density win must be data-driven, sign-stable, matched-compute vs mean-copy homogeneous; every cross-modal
claim passes A6 residualization. Report each wave in one paragraph: what ran, verdict, axis movement, next
lever. No wave ends without a committed artifact.
```

Usage: `/goal` on the Studio, or paste only the fenced block. The wrapper you are reading stays in the repo
as the record of what the prompt is and why. If the session dies mid-wave, the next session re-runs `/goal`
(or re-pastes the block); wave 0 detects its own completion (gates green + STUDIO_RUN_REPORT.md exists) and
the loop resumes from the scoreboard.
