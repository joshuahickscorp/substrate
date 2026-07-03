# STUDIO GOAL PROMPT: the iterative maximization loop for the M1 Ultra

This is the cold-start prompt for the first session ON the Mac Studio (M1 Ultra, 20-core CPU, 48 to 64-core
GPU, 128 GB unified memory, 8 TB SSD). It is deliberately a GOAL prompt, not a task list: it sets a bar above
the honest expected value (the theoretical ceiling from `STUDIO_POTENTIAL_AUDIT.md`) and wraps the work in an
explicit wave loop so every session converges on either a converted bet or a proven wall, never on drift.
Paste the fenced block below verbatim as the opening message of the Studio session. It is about 3,570
characters, under the 4,000 limit. House style everywhere: no em or en dashes.

Design notes (why each clause exists):
- The goal is set at >= 9.0 overall knowing the expected value is nearer 8. The laptop rounds proved the
  pattern: five successive stop-checks against "good enough" produced 2.75 points and six wins. The
  unrealistic bar is the mechanism, the honesty rules are the brake.
- The loop is ORIENT, SELECT, PREREGISTER, BUILD+RUN, VERIFY, LEDGER, STOP-CHECK. Preregistration before
  compute and independent adversarial verification before any doc write are the two clauses that killed
  every over-claim on the laptop (four in the axis-ceiling rounds alone). They are non-negotiable at scale.
- Wave 0 is pinned because the two riskiest unknowns are boring: does the transfer checklist actually
  reproduce green gates on the new box, and does MPS lift the M3 Pro buffer wall at 128 GB (measure, never
  assume; `STUDIO_HANDOFF.md` records the laptop failure).
- STUDIO_RUN_REPORT.md is the single accumulating artifact so a dead session loses at most one wave. It must
  be added to `scripts/check_docs.py` CANONICAL_MD when created (wave 0), or the docs gate fails the commit.
- The standing order (DR1 before multi-seeding, B5 last) is the audit's anti-comfort rule; it survived three
  audits and stays.

```text
You are on the Mac Studio (M1 Ultra, 20-core CPU, 48-64 core GPU, 128 GB unified memory, 8 TB SSD), repo at
~/mop. Read in order before any work: docs/mixture_of_perspectives/HANDOFF.md, STUDIO_POTENTIAL_AUDIT.md,
EXPAND_PHASE_PLAN.md, STUDIO_HANDOFF.md (transfer checklist). Hard rules: no em or en dashes anywhere; never
attribute Claude in git; PYTHONPATH=<repo>/src for import mop; enforce profile studio-m1ultra
(src/mop/studio/profiles.py); preregister every null hypothesis and verdict threshold IN CODE before the run;
a tie is a null; every candidate positive gets an INDEPENDENT adversarial verification pass before it may be
written into any doc; a faked or unverified score is program failure; a PROVEN wall with a mechanistic reason
is program success, exactly like the laptop's 6.75.

THE GOAL, set above expected value on purpose (aim at it, report honestly against it): drive the program from
the proven laptop ceiling (~6.75/10) to >= 9.0 overall, where overall = mean of falsification (hold 10),
abstraction (>= 9), density (>= 9), moldability (>= 8), and disposition >= 70 of the 86 semantic positions
(tested, or terminally walled with a named reason). You will likely land short; the real mandate is that
every lever in STUDIO_POTENTIAL_AUDIT.md ends CONVERTED or WALLED, so whatever number remains is proven.

WAVE 0, once, before any science: run the STUDIO_HANDOFF transfer checklist; full gates green on this box
(ruff format + check, mypy, pytest, check_docs, acceptance); create STUDIO_RUN_REPORT.md (scoreboard: axis
scores, open levers, s/clip benchmarks, wave log) and add it to check_docs CANONICAL_MD; microbench encode on
8 real clips, MPS vs 14-16 parallel CPU workers (the M3 Pro MPS buffer wall at 64f/256px ViT-L may or may not
reproduce at 128 GB: MEASURE it, record s/clip and the winner); rebuild the 64-clip real cache to >= 1000
clips as the first artifact; commit.

THEN LOOP until the goal is met or every lever is walled. Each wave: (1) ORIENT: reread the scoreboard; list
open levers with expected axis-delta per wall-clock hour. (2) SELECT the highest-leverage lever. Standing
order: DR1 before any multi-seeding, PR9 second, dense-cache + atlas encode ride the same conveyor, B5 and
seed-retrofits LAST; never refine an owned number while an unbuilt instrument blocks an axis. (3) PREREGISTER
in code: null, thresholds, controls (matched random-init encoder at matched resolution, never a square
projection; shuffle floor; matched compute; no-sign-flip rule). (4) BUILD + RUN inside studio-m1ultra:
min-free-disk 250 GB kill-switch, resumable checkpoints every 30 min on jobs over an hour, nohup + progress
log for anything long. (5) VERIFY: independent adversarial pass on every positive; the laptop killed four
over-claims this way; unverified positive = null. (6) LEDGER: update STUDIO_RUN_REPORT.md and the HANDOFF
verdict ledger, full gates, commit plain. (7) STOP-CHECK: if two consecutive waves moved no axis and produced
no wall proof, escalate (Track A: PR9 kill-switch fires -> Process C 1-10M pilot is licensed; Tracks B/C:
scale DR1) or write the wall proof and close the axis. DECISION GATES from EXPAND_PHASE_PLAN.md: DR1 caption
acceptance gate BEFORE encode spend; PR9 certificate-guarded tie -> moldability dead at frozen substrate;
density win must be data-driven, sign-stable, matched-compute vs mean-copy homogeneous; every cross-modal
claim passes A6 residualization. Report each wave in one short paragraph: what ran, verdict, axis movement,
next lever. No wave ends without a committed artifact.
```

Usage: paste only the fenced block. The wrapper you are reading stays in the repo as the record of what the
prompt is and why. If the session dies mid-wave, the next session re-pastes the same block; wave 0 detects
its own completion (gates green + STUDIO_RUN_REPORT.md exists) and the loop resumes from the scoreboard.
