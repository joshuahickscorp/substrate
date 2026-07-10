# F8 and F16 scientific execution contract

F8 and F16 have two deliberately separate paths.

- `execution_mode: smoke` is the default. It runs one mechanics step, leaves every claim metric
  `null`, sets `scientific_result: false`, and can never be promoted.
- `execution_mode: scientific` runs the complete four-arm experiment only after every evidence file,
  artifact hash, split, referent, inherited feature, and compute plan passes validation. It fails closed
  and writes receipts if any check fails.

The registry entries remain `registry-only`. Exercising the engine is not evidence that a licensed
natural run happened.

Both lanes are `env-later` and campaign `scale_boundary: environment`, not `gpu-later` or Studio-scale.
Their fixture-scientific path executes on the current machine, while the first unresolved inputs are a
rights-clean natural source, executable real-weight receipt, and trusted provenance authority. Resource
projections still describe M3 and possible Studio envelopes, but without a measured M3 failure they do
not create a hardware tier.

## Scientific arms

F8 compares:

1. `plastic_rewrite`: the inherited encoder weights themselves receive referent-paired
   self-supervised updates, then a frozen linear probe is trained.
2. `frozen_inherited`: a linear head is trained on immutable inherited features.
3. `larger_frozen_shell`: a wider nonlinear shell is trained on the same immutable features.
4. `random_init_same_arch`: the exact encoder architecture starts randomly initialized and receives
   the same self-supervised curriculum as the plastic arm.

F8's registered `representation_rewrite_delta` is the mean held-out cosine shift, `1 - cosine`, between
the inherited and rewritten representations. Candidate-versus-inherited accuracy is reported under a
separate auxiliary name and is not substituted for representation change.

F16 compares:

1. `blank_slate`: the encoder starts randomly initialized, receives the self-supervised curriculum,
   and is then frozen for a linear probe.
2. `frozen_inherited`: the inherited feature baseline.
3. `larger_frozen_shell`: the stronger inherited-feature shell control.
4. `random_init_same_arch`: an exact deep copy of the blank arm's initial weights is kept frozen while
   its probe trains. The only candidate/control difference is plastic self-supervised encoder training.

All arms receive the same declared estimated FLOP budget per seed. The convention charges linear
multiply-adds, activation forward/backward, SSL and cross-entropy losses, parameter/input gradients,
Adam moment and parameter updates, inherited or random feature production, and held-out inference. The
engine reduces examples for expensive arms and refuses when its estimated end-to-end ledger exceeds
the preregistered tolerance. Equal updates are not claimed because that would give larger architectures
more compute. This is algorithmic FLOP matching, not a measured hardware instruction, time, or energy
match; every result says so explicitly.

## Dataset package

`data_rights_manifest` points to a content-hashed NPZ file with exactly these arrays:

| Array | Contract |
| --- | --- |
| `inputs` | finite float tensor, `[referent, input_dim]` |
| `view_a`, `view_b` | referent-aligned self-supervised views with the exact input shape |
| `factor_labels` | nonnegative factor target per referent |
| `transfer_labels` | nonnegative transfer target per referent |
| `domain_labels` | nonnegative domain ID; test domains must be disjoint from train domains |
| `split` | frozen `0=train`, `1=validation`, `2=test` assignment |
| `referent_ids` | nonempty, globally unique explicit referents |
| `view_a_referent_ids`, `view_b_referent_ids` | exact row-order copies proving each view's referent binding |

Both target label spaces must contain at least two train classes, and every test class must occur in
train. Split values must be exact integers, both view-ID arrays must equal the canonical referent order,
and exact duplicate input payloads cannot cross train/test. The disjoint domain check makes the score a
held-out-domain transfer result rather than a random-row split.

The rights manifest uses schema `mop-rewrite-data-rights/v1` and records the package path and SHA-256,
license, source, explicit referent scheme, frozen-split declaration, and one provenance class: `fixture`
or `natural`.

## Encoder and inherited-feature receipt

`real_encoder_manifest` uses schema `mop-rewrite-encoder-receipt/v1`. It points to:

- an immutable `mop-mlp-npz/v1` encoder checkpoint, with `weight_0`, `bias_0`, and subsequent numbered
  layers;
- an immutable NPY inherited-feature matrix;
- exact hashes, architecture dimensions, activation, model identity, and reproduction tolerance.

The preflight loads the safe NPZ checkpoint without pickle, recomputes every inherited feature from the
rights-manifested inputs, and refuses any cache whose maximum error exceeds the tolerance. Natural scope
also requires `weights_real: true`, `feature_cache_real: true`, and nonempty training provenance.
Unsupported checkpoint formats do not silently fall back to random weights.

Before materializing an NPZ, preflight reads its ZIP member sizes and enforces both the configured
on-disk package cap and uncompressed resident-tensor cap. Raising either cap is an explicit run-config
change; a large archive cannot silently consume the laptop's unified memory. Class labels must be
contiguous from zero, and the largest arm is also checked against an explicit trainable-parameter cap
before any optimizer state is allocated.

## Evidence scopes and promotion

Every evidence document in one attempt must carry the same `artifact_class`.
The matched-compute plan and the F8 shell-failure or F16 inherited-baseline receipts must repeat the
exact dataset and weight hashes. The prerequisite wrapper must also point to a real source receipt whose
bytes match `receipt_sha256`; that source is parsed and independently checked for experiment ID, result
flag, dataset hash, and weight hash. Both F8 and F16 require an immutable seed plan bound to the dataset,
ordered seeds, margin, compute tolerance, and experiment ID. An unrelated or post-selected receipt
therefore cannot be reused.

- A fixture must declare `fixture_only: true`, `natural_data: false`, `weights_real: false`, and
  `feature_cache_real: false`.
- Natural evidence must declare the inverse and supply real training provenance.

A fixture may run all scientific arms so CI, null logic, density accounting, and resource handling are
tested. Its output is always stamped `fixture_taint_irreversible: true`,
`natural_claim_eligible: false`, and `promotion_eligible: false`, regardless of its scores.

Coordinated local JSON can still falsely call fixture bytes natural. The engine has no legal-rights or
pretraining-provenance trust root, so even a coherent `artifact_class: natural` package is stamped
`natural_evidence_declared: true` but remains `natural_claim_eligible: false` and
`promotion_eligible: false`. Promotion is fail closed until a separate verifier with a configured
external authority attests rights and weight provenance. The registry is never modified automatically.

The scientific null itself uses candidate minus the strongest control independently within each seed,
then subtracts the preregistered margin before computing CI and sign consistency. A positive average
cannot hide a seed that fails the margin.

## Receipts and hardware claims

Every attempt writes:

- `preflight_receipt.json`: document and cross-artifact checks;
- `resource_projection.json`: package bytes, projected FLOPs, estimated peak bytes, measured current
  host facts, and comparisons with `m3pro-local-max`, `studio-1tb`, and `studio-m1ultra` envelopes;
- `attempt_receipt.json`: completed, refused, or failed status, evidence hashes, device, result
  fingerprint, and promotion state;
- `scientific_progress.json`: an atomic update after every completed arm, including consumed estimated
  FLOPs, updates, examples, current seed, and an attempt-window resource sample. A later timeout or
  caught failure cannot erase already consumed work.

Resource comparisons are explicitly projections. They set `measured_hardware_wall: false` and cannot be
used as evidence that the current laptop exhausted memory or time, or that a planned Studio exists. A
hardware boundary requires a separate measured attempt on the relevant host.

Attempt memory uses a 10 ms process-RSS sampler, a resettable CUDA peak when available, and sampled MPS
current allocation because MPS exposes no resettable per-attempt peak API. The receipt separates this
attempt window from process-lifetime `ru_maxrss` and records the sampling limitations; it never presents
an old process high-water mark as this attempt's peak.

The deterministic package generator and end-to-end adversarial tests live in
`tests/integration/test_f_missing_lanes.py`. They prove valid fixture evidence reaches the scientific
engine, content tampering and unrelated receipts refuse, coherent natural-claim laundering cannot
promote, referent/split/duplicate leakage refuses, margin logic uses per-seed strongest controls, partial
failures preserve consumed work, seed plans reproduce, and fixture evidence never promotes.
