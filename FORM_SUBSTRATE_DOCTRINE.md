# FORM SUBSTRATE DOCTRINE

The methodological constitution of the Form Substrate Program. Everything here was earned, mostly by
negative results, across the V-JEPA, biology-lever, and Mixture-of-Perspectives eras. The paradigm
changed. The discipline did not. It tightened.

House style: no em or en dashes. Companion: FORM_SUBSTRATE_PROGRAM.md (what we are building),
FORM_SUBSTRATE_EXPERIMENTS.md (the bank), PERFORMANCE_DENSITY_DOCTRINE.md (cost law),
OPERATIONAL_AWARENESS.md (awareness law), PARADIGM_MIGRATION.md (what this doctrine inherits).

---

## 1. The spine (unchanged, non-negotiable)

Every experiment carries, before it runs:

```text
baseline . ablation . metric . null . control . gate . taxonomy slot
```

Enforced in code, not culture: experiments/base.py refuses a subclass missing the contract;
registry/experiments.yaml is the preregistration (null, headline metric, falsifier, taxonomy slot
committed before the run); mop.devel.registries.validate_experiment makes registry and code
inseparable; the north_star rail scans all free text.

## 2. The standing controls

Wired at harness level. A result that has not faced them is a claim, not a result.

1. Non-vacuous substrate control. The constitutional lesson of this repo: an invertible
   frozen-random PROJECTION is vacuous for every probe metric (the probe absorbs its inverse; the
   delta is forced to 0.000, measured). The only admissible substrate controls are a random-init
   same-architecture ENCODER at matched resolution, a lossy rank-reduced map, or a trained-shell
   dynamics metric. Form-substrate translation: every substantive form arm declares control_for
   pairing in FormMeta, and form_audit names missing controls before any evidence is read.
2. Matched compute (diagnostics/compute.py): iteration is not allowed to be unrolled depth; routing
   is not allowed to be extra FLOPs.
3. Matched capacity (shell/capmatch.py): sparse is not allowed to be a parameter count accident.
4. Tuned baseline: renamed biology and renamed routing must beat tuning, not defaults.
5. Shuffled floors: shuffled referents for alignment and memory, shuffled labels for probes,
   action-shuffle for consequence claims, shuffled anchors for maps.
6. Noisy-TV guard (diagnostics/noisy_tv.py): any uncertainty or curiosity signal must ignore
   irreducible noise (e4 died here 30 of 30).
7. Seed stability (harness sweep + riskcov seed CI): sign flips publish as instability, never as wins.
8. Difficulty calibration (diagnostics/difficulty_calibration.py): a tie is meaningless until a
   known-separable reference certifies the regime is non-ceiling. The binding constraint of the whole
   corpus was the test bed; this gate exists so it can never silently be again.

## 3. Referent discipline (the new first law)

The form substrate adds one law on top of the spine: no referents, no evidence.

- Every form batch carries referent ids; duplicates are refused at intake (substrate/form.py).
- Every alignment claim must beat shuffled-referent anchors, not just raw transfer.
- Every cross-form memory claim must beat form-local nearest neighbor and shuffled referents at
  matched memory slots.
- A matrix with one form kind is mechanically valid and scientifically weak; form_audit says so.
- Referent provenance rides the cache manifests (substrate/cache_manifest.py); a cache without a
  manifest is not citable.

## 4. The promotion pipeline

A number becomes a claim becomes a result only through this sequence, enforced by the studio
governance layer:

```text
preregister (registry row)
-> run (harness, receipts to runs/)
-> verdict (PUBLISH-POSITIVE, DOWNGRADE-TIE, SUBSTRATE-BOUND, SEED-UNSTABLE, CAPACITY-ARTIFACT)
-> null card (proof/NULL_CARDS/, cites its atlas probe_dependency)
-> independent adversarial verification (a positive without a verifier receipt is refused)
-> ledger (claim_plan forces verdict-gate then artifact-index then ledger, in that order)
```

Negatives are first-class: they file into the ten-slot failure taxonomy (proof/FAILURE_TAXONOMY.md)
and the obituaries. "Not decodable" is a substrate bound, not an embarrassment.

## 5. Trainable-substrate law (inherited brake, restated for forms)

Doc 15's skepticism doctrine binds the F-series without modification. Any claim that training the
substrate (rather than the shell) is necessary must clear ALL of:

- a LOCATED failure of the frozen inherited substrate on real, non-ceiling content that a
  matched-capacity shell cannot close,
- a NAMED property to install, with a preregistered margin over the best frozen arm,
- matched total compute, random-init same-architecture control, larger-shell-on-frozen control,
- old-form retention (a rewrite that forgets its old forms is a regression, not development),
- an explicit license receipt (the process_c_gate pattern: launch_allowed true or no training run).

F7, F8, and F16 are the only doors to trainable substrates, and they open on evidence, not desire.

## 6. Evidence ladder and provenance

R0 preregistered, R1 ran privately, R2 survived controls, R3 seeded and reproducible, R4
independently re-run, R5 externally citable. Provenance tags, richest first: natural-video,
real-encoder, structured-synthetic, provisional. A structured-synthetic positive is a mechanics
result; only natural-referent content can promote a capability claim to program-level.

## 7. Language law

The sentience rail (north_star.py) is part of the doctrine, not a formality. Engineering vocabulary
only; affirmative mentalistic claims are refused at render time. Prose escalation is a bug class:
"cross-form transfer improved by 0.12" may not become "the system understands." The same discipline
applies to performance ("1.3x capability per FLOP on F13's bed," never "blazingly fast") and to
awareness (OPERATIONAL_AWARENESS.md language contract).

## 8. What this doctrine promises

The program gets more ambitious only by becoming more falsifiable. Every poetic sentence in
FORM_SUBSTRATE_PROGRAM.md has an experimental translation in FORM_SUBSTRATE_EXPERIMENTS.md, every
experiment has a null, every null has a home in the taxonomy, and every positive has to survive the
controls that killed its predecessors. The fossil record of dead mechanisms is the proof that the
survivors mean something.
