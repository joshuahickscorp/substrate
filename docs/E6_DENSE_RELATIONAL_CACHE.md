# E6 cache-first dense relational path

## What changed

E6 now has a bounded token-aware execution path that never loads an encoder and never flattens a
full dense clip into a learned layer. The legacy runner remains a frozen-random mechanics scaffold;
the new path is `scripts/e6_dense_relational_cache.py` backed by
`mop.experiments.e6_dense_relational`.

The redesign also corrects a task flaw. Predicting a composite class that is wholly absent from
training cannot establish recombination generalization because that output class has no supervised
examples. The cache-first path predicts factor A and factor B separately after training on one set
of combinations and evaluating on disjoint validation/test combinations. Every factor level must
still occur in training.

## Bounded token interface

For one memmapped cache row `[token axes..., feature]` at a time, the readout:

1. preserves canonical token order;
2. partitions tokens into at most 16 fixed contiguous bins;
3. mean-pools within each bin;
4. applies a frozen feature projection of rank at most 64;
5. applies a frozen summary projection to at most 256 values;
6. fits two ridge heads, one per factor.

The flat control globally mean-pools all tokens and projects to the exact same summary dimension.
The token-shuffle control preserves every token value but independently permutes positions per
referent. The learned, flat, shuffled, random-structured, and random-flat heads therefore have the
same number of learned parameters.

At full ViT-B/64-frame shape, the legacy input would be `18,432 * 768 = 14,155,776` values per
clip. The default new learned head sees 32 values. The full cache is never materialized as a
flattened matrix.

## Paired cache gate

Mechanics require both caches to pass strict citable validation and to match exactly on:

- dense shape and row count;
- ordered explicit referents;
- factor columns and factor metadata;
- train, validation, and test rows;
- disjoint held-out factor combinations;
- the source block and per-referent input tensor hashes;
- the architecture signature.

The substantive arm must declare `random_init: false`; the control must declare
`random_init: true`. Latent `keys.npy` values are not compared because real and random encoders
should produce different retrieval keys. Referent sidecars, not feature values, establish
same-content identity.

Scientific promotion additionally requires:

- at least 200 rights-clean natural-video clips;
- learned objective `inherited-frozen` with a real checkpoint receipt;
- control objective `random-control` with exact seeded state-dict and official-source hashes;
- the pinned V-JEPA 2.1 ViT-B architecture;
- byte-identical 384px inputs over the two encoders;
- content, split, annotation, view-recipe, and source-authority SHA256s;
- unique inputs with untouched test membership;
- five fixed projection seeds;
- primary delta over the strongest per-seed control above 0.02;
- a conservative five-seed confidence lower bound above zero, no sign flip, position specificity,
  exact parameter matching, and an off-ceiling learned score.

The cache manifest validator now recognizes `vjepa_official_random_init` as a distinct honest
backend. Its receipt must pin the official repository commit and hash the V-JEPA 2.1 vision
transformer source. It cannot masquerade as the existing Hugging Face random-init backend.

## Tiny mechanics result

`proof/E6_DENSE_RELATIONAL_MECHANICS.json` was produced from two ephemeral 36-row programmatic
caches with shape `[36,16,8]`. Both caches passed citable mechanics, exact referent/factor/split/
source pairing passed, every head was parameter matched, and the readout stayed bounded at 32
values rather than flattening 128.

Scientific promotion is false. The primary mean delta was positive in this deliberately tiny
fixture, but its confidence interval crossed zero and seed signs varied. More importantly, the
fixture fails the source/promotion requirements: it is programmatic, too small, not official
ViT-B, and lacks real/random weight receipts. This is the intended fail-closed result.

## Remaining blockers

1. Finish the separate ViT-B checkpoint download, hash, strict-load, and 8-frame forward sequence
   after CM7 and only if the disk gate remains green.
2. Build an official-architecture seeded random-init ViT-B loader and receipt; do not reuse the
   V-JEPA 2 Hugging Face architecture as a 2.1 control.
3. Expand a development-only SANPO train-pool cohort to at least 200 clips selected before pixel
   inspection for two nondegenerate annotated factors and disjoint combination splits. The current
   ten-session smoke cohort is intake evidence, not an adequately powered E6 design.
4. Preprocess each selected clip once at 384px, hash the exact tensor, and feed those byte-identical
   tensors to learned and random encoders. Keep the official SANPO test split sealed from tuning.
5. Cache dense tokens serially, one heavy process at a time, with full encoder/init/source receipts.
6. Run the command below, inspect all source and statistical gates, then let the independent E6
   verifier decide promotion. ViT-L/g/G remain locked.

## Future command

```bash
PYTHONPATH=src .venv/bin/python scripts/e6_dense_relational_cache.py \
  --learned-cache data/cache/vjepa21_vitb_sanpo_e6_real \
  --random-cache data/cache/vjepa21_vitb_sanpo_e6_randominit \
  --bins 8 --feature-rank 16 --summary-dim 32 \
  --seeds 0-4 --min-margin 0.02 \
  --proof proof/E6_DENSE_RELATIONAL_NATURAL.json
```

Until both input caches exist, the only authorized command is the tiny mechanics fixture:

```bash
PYTHONPATH=src .venv/bin/python scripts/e6_dense_relational_cache.py --fixture
```
