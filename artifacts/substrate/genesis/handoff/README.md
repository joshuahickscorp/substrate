# Tangible sandbox handoff

External activation is `false`. Nothing in this handoff changes that, and no
step below is authorised to.

This directory tells the next program what it inherits, what it must not
repeat, and what would actually move the question forward.

## What you inherit

A cognitive material foundation that runs, is instrumented, and can be
falsified:

- eleven candidate materials and twelve controls, each a separate
  implementation, all twenty-three producing distinct durable state under an
  identical probe sequence
- a material interface that makes label leakage structurally impossible: a
  material receives observations and probes and is handed only scalar verdicts
- fourteen sealed challenge families with a public commitment over generator
  source, seed namespace and configuration
- twelve mechanism canaries, each with a paired negative test
- a frozen analysis: developmental history as the independent unit, paired
  difference, history-level bias-corrected bootstrap, Holm correction
- a freeze whose seed namespaces are derived from the freeze document's own
  digest, so no principal instance is computable before the freeze exists

## The two instruments that matter most

Everything else is secondary to these. Run them first, every time.

- `record_store_null` — a policy that copies observed labelled fields. It must
  score at or below chance on every family. If it does not, the measure is
  testing storage rather than development.
- `reference_learner` — implements each family's intended solution path from
  the observation stream alone, touching no sealed answer or seed. It must
  score near 1.0 on every family. If it does not, that family is unanswerable
  from experience and any failure on it is an artefact, not a result.

Both instruments found real defects during this campaign that no amount of
code review had caught: one family leaked a copyable answer, and one was
unanswerable from experience while a constant guess scored perfectly. Neither
was visible in the code. Both were obvious the moment the instruments ran.

## What this campaign actually found

The strongest equally plastic control reached 0.4246 against a chance level of
0.125. The best candidate reached 0.1634. Random growth — the counterfeit that
grows structure without verified value — reached 0.1670 and edged out the best
candidate. The oracle reached 1.0 and the reference learner 0.7866, so the
tasks were solvable and the measure had resolution across the full range.

The candidates did not use it.

## What not to repeat

**Do not build eleven materials in one session.** They were implemented once
each and never tuned. A null on an untuned implementation says almost nothing
about its architecture. Either build fewer and develop them properly, or state
the limitation as loudly as this campaign does.

**Do not trust proposal counts to match by accident.** The candidates emitted
one proposal per cycle while the control emitted thousands, under the same
budget. That asymmetry was found by inspection, not by any gate, and it biases
toward a false null. Count attempts explicitly and publish the counts.

**Do not let a result-dependent criterion sit among the prerequisites.** The
counterfeit check was written as a prerequisite, which would have forced
Outcome C the moment a counterfeit tied the best candidate. Separate statements
about the instrument from statements about the candidate before you have
results, not after.

**Do not measure improvement without applying the change.** The first
plasticity loop scored a proposal before committing it, so every improvement
was zero, nothing was ever admitted, and an entire diagnostic measured initial
state. Simulate, verify, then commit or roll back.

## What would actually move the question

1. **Scale.** Thirty-two histories of two hundred observations is far below
   where developmental effects would separate architectures. The instrument is
   ready for far more than was run through it.
2. **Native training at meaningful scale.** These materials develop only
   through verified rewrites during a run. Nothing here trains.
3. **Richer answers.** An eight-symbol alphabet puts chance at 0.125 and makes
   luck common. A larger structured answer space would sharpen every effect.
4. **Composition beyond pairs.** Hidden composition interleaves two families.
   Three or more, or a learned skill composed with a novel tool, is untested.
5. **A genuinely separate exact shell.** Identity and the claim boundary
   currently live in the same process as the approximate mechanisms.
6. **The nine pending mutations.** They are reported as pending, never as
   caught. Those failure modes are not excluded.

## Entry points

```bash
substrate genesis status
substrate genesis preflight
python -m substrate.genesis_tournament
python -c "from substrate import genesis_canaries as c; print(c.run_all()['all_pass'])"
```

See `docs/genesis/RUNBOOK.md` for the full sequence and
`docs/genesis/LIMITATIONS.md` for what this program does not support.

## The boundary

No outcome of this program assigns unqualified Nous, consciousness, sentience,
phenomenal experience, moral status, human equivalence, or unrestricted
autonomy. The inherited Final Revision result is preserved unchanged, including
its decisive effect of 0.0 with a 95% confidence interval of [0, 0]. A sandbox
is a place to run experiments, not a grant of autonomy.
