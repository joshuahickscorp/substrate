# Rollout Lane Result: What the V-JEPA 2 Predictor Actually Forecasts (Facet 12)

## 1. Headline (honest)

The entire MoP corpus used V-JEPA 2's ENCODER and discarded its PREDICTOR, the learned
latent-space simulator of video dynamics. This is the first time the program drives the
predictor as a world model. Facet 12's acceptance gate (DR13 rollout-error compounding, now on
REAL predictor rollouts instead of synthetic transitions) returns a preregistered NULL on
usability: on 24 synthetic bound-nuisance clips the frozen predictor beats all three controls
(persistence, matched random-init predictor, shuffled-target) by a non-overlapping seed CI at
EVERY horizon 1 to 8, but by only about 5 to 7 percent, never within reach of the preregistered
usability bar (real error below half the best control). A directional-but-sub-usable signal is a
null under the preregistered rule, and a tie is a null.

So the predictor carries a genuine, sign-stable, adversarially-verified world-model signal on our
content, and that signal is far too weak to support multi-step latent planning. The rollout lane
(counterfactual and interventional abstraction, the ex2 latent-planning precursor, DR7 latent
chain-of-thought) is NOT licensed at usable fidelity by this evidence. Facet 12 does not move off
0 on usability grounds.

A second wave (section 11) tested the sharper, task-relevant criterion (does the rolled-out latent
keep the moving object's POSITION decodable, the thing counterfactual abstraction and planning
actually need) and reached the same verdict: WALL, correctly labeled null-by-ill-posedness because
the synthetic clips move sub-patch at every horizon. It also produced a genuinely new,
adversarially-survived mechanistic finding: object position SURVIVES the compounded rollout but in a
representational sub-space the encoder-trained head cannot read (in-domain probe R2 0.73 vs
encoder-trained 0.09 at h=1), which names a concrete Studio fix (a readout adapter) and confirms the
representational gap of wave 1 is a systematic sub-space shift, not noise.

This verdict is PROVISIONAL on the clipset. The clips are synthetic and out of distribution for a
predictor trained on real video, and the whole-future-slot masking is out of distribution for the
predictor's training mask pattern, so the honest real-scale verdict requires the Studio re-run on
hosted real corpora (facet 14 feeds facet 12), which the same instrument runs turnkey via
`--clip-dir`. What DID convert is the lever itself: from an unmeasured capability to a measured one
with a preregistered, bit-exact, adversarially-verified instrument and a decisive synthetic-clipset
number.

## 2. What the lever is

V-JEPA 2 ships an encoder plus a PREDICTOR: a masked spatiotemporal-patch predictor over one clip's
32 temporal-slot by 16x16 spatial patch grid (256 patches per slot, 8192 total). Given encoder
hidden states at CONTEXT patch indices plus TARGET patch indices, it forecasts the target patch
representations. The teacher for that forecast is the encoder's OWN representation of the target
patches, which is exactly V-JEPA's training signal. Rolling the forecast forward (substituting the
predicted slot representations back into the context buffer before predicting the next slot) is an
open-loop latent rollout, and the question is whether that rollout stays faithful for enough steps
to be a usable world model, or whether error compounds so fast the lane is bounded to one-step
counterfactuals (the terminal ex2/DR13 wall the audit anticipated).

## 3. The instrument is real and correct (validated, not assumed)

- The model loads as VJEPA2Model from the local HF cache (`facebook/vjepa2-vitl-fpc64-256`,
  `local_files_only`), float32, CPU. MPS overflows 64-frame V-JEPA on the M3 Pro, so CPU.
- Grid confirmed: 32 temporal slots, 16x16 spatial, 256 patches per slot.
- BIT-EXACT teacher: the harness's direct-submodule predictor call and its teacher target were
  checked against the top-level VJEPA2Model path. pred(top-level) vs pred(submodule) max|diff| =
  0.0; teacher(top-level target_hidden_state) vs harness teacher (encoder full-clip state at the
  target slot) max|diff| = 0.0; nmse identical across paths. The harness measures exactly the
  model's own masked-prediction objective, not a proxy.

## 4. Preregistration (fixed in code before any number)

From `scripts/mop_dr13_predictor_fidelity.py`, committed before running:

- Horizons h in {1, 2, 3, 4, 6, 8} temporal-slot lookaheads. Rollout compounds by feeding predicted
  slots back into the context buffer (true open-loop rollout).
- Usable horizon = the largest CONTIGUOUS h (from h=1) at which real_nmse.hi < every control_nmse.lo
  (non-overlapping seed CI, lower is better) AND real_nmse.mean < 0.5 * best_control_nmse. A tie is
  a NULL.
- Three non-vacuous controls:
  (a) persistence: copy the last ground-truth context slot (no dynamics).
  (b) random_init: the SAME predictor architecture with freshly initialized weights, identical pipe.
  (c) shuffled_target: the real predictor output scored against a DIFFERENT clip's true target slot.
      This control shares the predictor-vs-encoder representational gap (see section 6), so beating
      it isolates genuine clip-specific dynamics net of that gap.
- Verdict rule: convert if usable horizon >= 2; wall-to-1-step if real is cleanly usable only at
  h=1; null otherwise.
- 24 clips bucketed into 3 seed groups for a seed CI, T_START = 4.

## 5. Results (24 synthetic clips, CPU, 2026-07-03)

Per-horizon nmse (lower is better). "real/best" is the ratio of real error to the best (smallest)
control error; usable requires it below 0.5.

| h | real nmse | persistence | random-init | shuffled | best/real ratio | beats all 3 (CI) | usable |
|---|-----------|-------------|-------------|----------|-----------------|------------------|--------|
| 1 | 0.767     | 0.964       | 1.035       | 0.824    | 0.93            | yes              | no     |
| 2 | 0.816     | 0.906       | 1.038       | 0.862    | 0.95            | yes              | no     |
| 3 | 0.833     | 0.906       | 1.040       | 0.873    | 0.95            | yes              | no     |
| 4 | 0.833     | 1.138       | 1.044       | 0.873    | 0.95            | yes              | no     |
| 6 | 0.815     | 1.022       | 1.037       | 0.858    | 0.95            | yes              | no     |
| 8 | 0.971     | 1.064       | 1.033       | 0.982    | 0.99            | yes              | no     |

- Real beats all three controls by a non-overlapping seed CI at every horizon. That is a real
  signal, not chance: the seed CIs are tight (real_nmse ci_half about 0.001 to 0.003).
- Real never comes close to the 0.5 usability bar. The margin over the strongest control
  (shuffled_target) is about 5 to 7 percent and collapses to about 1 percent by h=8.
- Encoder adjacent-slot nmse scale = 0.948. Adjacent V-JEPA temporal slots are nearly decorrelated
  in nmse, so one step of true motion is about 0.95; the predictor's 0.767 is below that but
  captures only about 20 percent of the transition.
- Cosine distance tells the same story more gently: real cosd 0.349 at h=1 (cosine similarity 0.65)
  rising to 0.82 by h=8.

## 6. The representational-gap finding (from the leakage probe)

The independent verifier's leakage probe (V3) asked the predictor to reconstruct a slot that is
ALREADY in its context. If masking leaked, that would be a trivial near-zero copy. It is not:
in-context nmse = 0.750, future (h=1) nmse = 0.775. Two consequences, both load-bearing for honesty:

1. The predictor is NOT trivially copying the target (future is genuinely harder than in-context),
   so the h=1 signal is real prediction, not leakage.
2. The predictor is LOSSY even on fully visible content (0.750), so most of the roughly 0.77
   one-step "rollout error" is a predictor-vs-encoder REPRESENTATIONAL GAP, not forecast failure.
   The MARGINAL one-step forecast cost is only about 0.025 nmse above that representational floor.

The shuffled_target control shares this representational gap (it is also predictor-output vs an
encoder slot), so "real beats shuffled at every horizon" isolates genuine clip-specific dynamics
NET of the gap. The gap does not rescue usability, though: the predictor's outputs do not live
cleanly in encoder space, which is itself a reason latent-planning-by-rollout in encoder space is
hard, and the raw error is what a downstream rollout consumer would actually see.

## 7. Independent adversarial verification (all pass)

`verify_facet12.py` does NOT import the harness. It re-derives the load-bearing claims by an
independent path with DISJOINT seeds (fresh clips seeded 50000+, harness clips seeded 1000+;
random-init seed 999 vs harness 12345) and tries to break them:

- V1a random-init weights differ from real: mean|dw| = 0.154 (control b non-vacuous). PASS
- V1b shuffled-target is a genuinely different clip: all clips true != other. PASS
- V2a real < persistence at h=1 on fresh clips: 0.766 vs 0.961. PASS
- V2b real < random-init at h=1 on fresh clips: 0.766 vs 1.023. PASS
- V3 leakage probe, future harder than in-context: 0.775 > 0.750. PASS
- V4 harness h=1 reproduces on fresh clips: verify 0.766 vs harness 0.767. PASS

A shrinkage artifact is ruled out by construction: nmse rewards matching the target magnitude, so a
predictor shrinking toward zero would push nmse toward 1, not down; real's low nmse plus the cosine
alignment (0.65 at h=1) confirm directional prediction, not magnitude collapse.

## 8. Verdict and what it licenses

NULL on usability. The rollout lane is NOT licensed for multi-step latent planning or counterfactual
rollout at usable fidelity ON THIS SYNTHETIC CLIPSET. There IS a genuine, adversarially-verified,
sign-stable world-model signal (real beats all three controls at every horizon), so the predictor
is a real if weak simulator of our content; it is just far below the fidelity a rollout consumer
needs. Per the goal loop, a proven wall with a mechanism is success: the mechanism here is that the
predictor's one-step forecast, while better than every baseline, sits deep inside a large
representational gap and against a latent whose adjacent slots are nearly decorrelated, so open-loop
rollouts diverge immediately.

The verdict is PROVISIONAL on content. These clips are synthetic and out of distribution for a
predictor trained on real video, and the whole-future-slot masking is out of distribution for the
training mask pattern. The licensed real-scale verdict requires the Studio re-run on hosted real
corpora. Two outcomes for the Studio:

- If on real video the margin widens past the usability bar: the rollout lane is licensed and facet
  12 moves off 0 toward its ceiling (counterfactual abstraction, ex2 latent planning, DR7).
- If it stays weak on real video: that is the honest ex2/DR13 wall at real scale, exactly as the
  audit anticipated ("bounded to one-step counterfactuals... itself the honest ex2/DR13 verdict").

## 9. Scoring impact

Facet 12 stays at 0 on usability. The lever converts from UNMEASURED to MEASURED: the program now
owns a preregistered, bit-exact, adversarially-verified DR13-on-real-predictor instrument, a
decisive synthetic-clipset number, and a turnkey real-corpora next step. No axis score is claimed
or moved on the strength of a synthetic OOD clipset. This is an M3 Pro early-lever result
(doctrine-sanctioned to run ahead of the spine); it does not complete Studio WAVE 0, which still
requires the M1 Ultra (MPS-vs-CPU microbench at 128 GB, 1000-clip real cache rebuild, full gates on
that box). This result is folded into STUDIO_RUN_REPORT.md when the Studio creates it at WAVE 0.

## 10. Reproduction

```
PYTHONPATH=src:scripts:. OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/mop_dr13_predictor_fidelity.py --n-clips 24
# real corpora (Studio): add --clip-dir DIR of .pt clip tensors [frames,3,H,W]
```

Instrument: `scripts/mop_dr13_predictor_fidelity.py` (preregistered, in code). Synthetic-transition
sibling (the compounding-with-horizon reference): `scripts/mop_dr13_horizon_limit.py`. Audit context:
`docs/mixture_of_perspectives/STUDIO_POTENTIAL_AUDIT.md` facet 12.

## 11. Wave 2: decodability-retention (does the rollout track motion?)

Raw nmse (waves above) is not what the rollout lane exists for. Counterfactual/interventional
abstraction and latent planning need the rolled-out latent to keep TASK CONTENT decodable,
specifically to track WHERE A MOVING OBJECT GOES. Wave 2 preregistered that criterion: fit a linear
ridge probe on TRUE encoder latents at slot T_START+h to decode the object centroid (cx, cy), apply
the same probe unchanged to the compounded open-loop rollout latent, and require the rollout to beat
PERSISTENCE (hold the last real slot, the "assume nothing moved" floor) plus the random-init and
shuffled controls by a non-overlapping seed CI, retaining at least half the true-latent
above-persistence decodability. Persistence is the sharp control: a world model must track motion
better than assuming nothing moved.

VERDICT: WALL, correctly labeled NULL-by-ill-posedness. Facet 12 does not move off 0.

- Under the realistic encoder-trained readout the rollout decodes position at the random/shuffled
  floor: position R2 0.09 / 0.06 / 0.06 / 0.07 / 0.05 / -0.005 at h = 1/2/3/4/6/8, versus persistence
  0.35 / 0.33 / 0.35 / 0.23 / 0.29 / 0.21 and the true ceiling 0.40 / 0.37 / 0.43 / 0.41 / 0.42 /
  0.32. Retention vs persistence is NEGATIVE at every horizon; beats-persistence by seed CI is FALSE
  at every horizon; usable_horizon = 0.

The independent adversarial re-derive (fresh seeds, fresh code, shuffled clip-disjoint split) held
the wall on all four vectors AND corrected one over-claim in the build's own write-up, which is the
adversarial discipline working as intended:

- MOTION PREMISE FALSIFIED. The build's pixel-centroid motion gate (which reported h=8 displacement
  18.8 px and set motion_testable = true) was EXTRACTION NOISE. Reconstructing the object's true
  trajectory analytically from the generator's own RNG draw order (r, rot, x0, y0, vx, vy; _hue
  consumes no RNG, verified against `scripts/compositional_under_nuisance.py`) shows true
  displacement is SUB-PATCH at every horizon (median 0.60 / 1.21 / 1.81 / 2.42 / 3.63 / 4.84 px; h=8
  is 0.30 patch). Pixel-vs-analytic per-clip displacement is NEGATIVELY correlated at every horizon
  (about -0.55 short, -0.30 at h=8): the clips the extractor thought moved most actually moved least.
  make_bound_nuisance_clip draws vx, vy in [-0.2, 0.2] normalized (about vx/32 per slot), so motion
  is intrinsically sub-patch and there is no velocity argument to amplify. Corrected label:
  motion_testable = FALSE, the clipset cannot pose the motion-tracking question. This is a null on an
  ill-posed clipset, not a demonstrated dynamics wall.
- PROBE HONESTY CLEAN: the probe is fit on true encoder latents and applied to predicted latents (not
  re-fit on predicted latents, so the representational gap is not laundered); the split is
  clip-disjoint and the wall reproduces under a shuffled split.
- COMPOUNDING GENUINELY OPEN-LOOP: open-loop vs teacher-forced rollout L2 divergence is 0.0 at h=1
  and monotone for h>=2 (0.54, 0.61, 1.26 at h=2, 4, 8), so predicted latents are truly fed back.

NEW MECHANISTIC FINDING (survives adversarial). The wall is NOT total content destruction. A probe
fit IN-DOMAIN on rollout latents recovers position well: R2 0.73 / 0.69 / 0.64 / 0.45 at h =
1/2/4/8, versus 0.05 to 0.09 for the encoder-trained probe. So object position SURVIVES the
compounded rollout, but the predictor writes it into a representational sub-space the encoder-trained
head cannot read zero-shot. This confirms wave 1's representational gap is a systematic sub-space
shift, not noise. Even the generous in-domain probe beats in-domain persistence by seed CI at ONLY
h=1 (fragile, sub-patch motion, wide CIs), so it is not a licensing signal; but it names a concrete
mechanistic fix for the Studio: a per-representation-space READOUT ADAPTER between predictor output
and any downstream planning/counterfactual head.

Licensed re-test (Studio): real moving video with genuinely supra-patch object motion (facet 14
corpora), decoded through a readout adapter fit on rollout latents. Method and provenance:
`scratchpad/facet12b/` (decodability_retention.py, adversarial_pass.py, motion_validation +
analytic-trajectory correction). No axis score is moved on synthetic ill-posed content.

ADAPTER SCAFFOLD (turnkey re-test, `scripts/mop_dr13_readout_adapter.py`, `--clip-dir` for real video):
a linear predictor-space to encoder-space adapter, fit self-supervised on VISIBLE slots, is built and
smoke-run. Finding (synthetic, provisional): the adapter genuinely HALVES the representational gap on
visible slots (in-context nmse 0.727 to 0.357) but that correction does NOT transfer to the open-loop
rollout (adapted rollout nmse 0.82 to 1.20, slightly worse than the raw 0.76 to 0.83), because the
compounded rollout latents drift out of the visible-slot distribution. So a NAIVE visible-slot adapter
is insufficient; the Studio should fit the adapter on actual rollout predictions (teacher-forced targets)
or per-horizon, on real moving video. The scaffold produces this verdict preregistered and de-risked.
