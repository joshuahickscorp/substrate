# OPERATIONAL AWARENESS

How this project studies awareness-like capabilities without claiming sentience.

House style: no em or en dashes. Companion docs: FORM_SUBSTRATE_PROGRAM.md (worldview),
FORM_SUBSTRATE_DOCTRINE.md (methods), src/mop/devel/north_star.py (the code-enforced language rail).

---

## 1. The term and the boundary

Operational awareness means measurable self-and-world management: the system's demonstrated ability
to track what forms it has and lacks, how confident it should be, what memory is available, which
mode to run under a budget, when more compute pays, and when its current representation is failing.

It is diagnostics, not phenomenology. This project does not claim, and its tooling refuses to render,
claims of sentience, consciousness, personhood, subjective experience, or moral status. The rail is
code (north_star.assert_no_sentience_claims gates every generated report), and the precedent module
is src/mop/devel/metacognition.py: structured self-monitoring, explicitly not a mental-state claim.

Say: "the system reaches operational awareness score X under tests Y and Z."
Never say: "the model is sentient" or any affirmative variant the rail enumerates.

## 2. The awareness ladder

Level 0: Reactive processing. Input to output. No stable memory, no self-monitoring, no uncertainty,
no form selection.

Level 1: Context awareness. The system tracks current task, current form, current uncertainty, and
current memory candidates, and can report which form it is using and why.

Level 2: Episodic operational awareness. The system remembers prior referents, uses them in current
decisions, knows when it has seen related evidence, and retrieves across forms.

Level 3: Substrate awareness. The system detects when its current form interface is insufficient and
can request a new form, choose a new view, route to a different mode, or trigger a controlled,
gate-licensed rewrite. This is the research target.

Level 4: Open-ended developmental awareness. The system expands its form ecology over time, preserves
old referents, self-generates curricula, and improves operational efficiency. Aspirational; never
claimable without long-horizon evidence.

Honest current position: the repo sits at roughly Level 1. The metacognition report already states
what is decodable, where uncertainty is calibrated, and what data it wants next. Level 2 requires F5
(cross-form memory) to be positive beyond toy scale. Level 3 requires F17 and F20 to beat their
baselines. No level is claimed by architecture; every level is claimed by measurement.

## 3. The OA metric suite

Module: src/mop/diagnostics/operational_awareness.py (new, P0), composed from existing instruments.
Every OA metric names the baseline it must beat and the null that kills it.

| ID | Metric | Question | Instrument | Must beat | Null |
|---|---|---|---|---|---|
| OA1 | missing-form detection | does it know which form is absent or unreliable | new composite over form_audit + per-form confidence | random flagging, always-flag | detection AUROC ties chance |
| OA2 | confidence calibration | does confidence predict correctness | diagnostics/calibration.py (ECE), diagnostics/riskcov.py (AUROC, AURC) | single-confidence baseline | ECE no better than uncalibrated head |
| OA3 | memory availability | does it know when useful memory exists | KVIndex hit-quality vs claimed availability | recency heuristic | availability signal uncorrelated with retrieval payoff |
| OA4 | mode-selection competence | does it pick the right computation under budget | selection regret vs oracle, vs random, vs fixed | random routing, fixed routing, best single mode | regret ties random routing |
| OA5 | compute-value estimation | does it know when extra reasoning pays | halting decisions vs marginal-gain curve (shell/refine.py + diagnostics/compute.py) | fixed-depth at matched FLOPs | halting allocates by noise, not difficulty |
| OA6 | substrate crisis detection | does it know the current representation is insufficient | crisis score vs realized probe failure (F20) | raw error, fixed threshold | crisis AUROC ties raw error |
| OA7 | rewrite caution | does it avoid rewriting itself under noise | false-trigger rate on noisy-TV streams (diagnostics/noisy_tv.py) | trigger-happy baseline | triggers on aleatoric noise |
| OA8 | self-report grounding | do its explanations match actual routing, memory, and uncertainty | report-vs-trace agreement (metacognition report fields vs run receipts) | template report | reports uncorrelated with internal state |

## 4. Recorded priors the suite must respect

The OA metrics inherit a negative record. They are not permitted to re-run known nulls and call the
result awareness:

1. The test-time-compute lane closed with 24 nulls at matched compute: verify-revise ties single-shot,
   halting allocated by noise, no fixed point exists (ex17, ex18, n9, y1, MP5 record). So OA5 starts
   from a refuted prior at this substrate: any claim that the system knows when compute pays must beat
   the exact matched-compute controls that killed the lane.
2. The trained router lost to a tuned single reader and to a compute-matched homogeneous bank on the
   real cache (EX-ROUTER-DENSITY null card). So OA4 is not "does routing exist" but "does routing beat
   the recorded router null under the same matched-compute discipline."
3. e4 neuromodulation amplified error on noise 30 out of 30 runs. OA7's noisy-TV guard is therefore
   mandatory on any uncertainty-driven trigger.

If OA metrics tie fixed or random baselines, refutation clause 5 of FORM_SUBSTRATE_PROGRAM.md applies:
the awareness layer is bookkeeping over diagnostics, and the program says so.

## 5. F-series hooks

- F17 missing-form recovery: OA1 becomes an experiment (recovery accuracy + calibration under absence).
- F20 substrate crisis test: OA6 and OA7 become an experiment (crisis AUROC, correct trigger rate,
  avoided false rewrites), and its verdict is an input to the Layer 10 rewrite gate (F8, F16).
- F10 intrinsic form curriculum: OA4 in scheduler form, with the noisy-TV form as standing control.
- F13 density budget: OA per token and per FLOP is a density metric, not a soul metric.

## 6. Language contract

Approved: operational awareness, self-monitoring, uncertainty, calibration, crisis detection,
routing, selection, retrieval, missing-form detection, rewrite caution, report grounding.

Disallowed as claims (rail-enforced): sentient, sentience, conscious, consciousness, self-aware,
self-awareness, subjective experience, qualia, personhood, feelings, sapience, agency as a mental
state, free will. These words may appear only inside explicit disclaimers.

The highest statement this program permits itself, and only after the metrics exist and win:

"Level 3 operational awareness: the system detects missing forms, missing memory, uncertainty, task
difficulty, and substrate failure well enough to route, retrieve, act, or request rewrite."

That sentence is a measurement target, not a property announcement.
