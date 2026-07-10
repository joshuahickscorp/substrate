# SANPO-to-custom-substrate bridge

## Outcome

The verified SANPO-Real smoke cohort now has a fail-closed input seam for the project's own portable
video substrate. The seam does not import an inherited encoder and the completed preflight did not
load or run any model. It verified all 90 consumer files (10 descriptions and 80 frames), preserved
the source session/split/attribute/referent identities, and decoded only the six train plus two
validation clips.

The official test remains sealed. Its 16 frame files were SHA-256 verified as part of source
integrity, but no test PNG was decoded or inspected. A normal caller can iterate development clips or
load a named development session; supplying a test-session ID is refused.

The durable preflight is `proof/SANPO_CUSTOM_SUBSTRATE_BRIDGE_PREFLIGHT.json`. At the current frozen
boundary it records:

- source content set `48896f6545ce0b9b2e4c1a338f55447dceac628eb321479fbcdd9ccc79c66afa`;
- bridge-plan identity `7808a352ef0e72412a2738d2549a349091e133545facc86cfa453e1b7339d741`;
- 288,966,841 consumer bytes rehashed, 90/90 files verified, and 10/10 sessions cross-bound;
- eight development clips and 64 development frames decoded in about 4.06 seconds;
- one finite `float32` tensor per session with shape `[1,3,8,256,256]` and range `[0,1]`;
- zero official-test frames decoded; and
- no inherited encoder, portable artifact, model forward, gradient, update, or training process.

This is natural-video **input evidence**, not representation-quality or learning evidence.

## Frozen preprocessing contract

`configs/custom_substrate/sanpo_natural_bridge_v1.json` binds the final intake proof, independent
verification proof, consumer manifest, explicit splits, frame referents, and content-set hash by
SHA-256. It also freezes the following transform:

1. Decode the generation-pinned source as PNG with Pillow 12.2.0 and convert to RGB.
2. Preserve the exact ordered source indices `0,8,16,24,32,40,48,56`.
3. Resize the shorter spatial side to 256 with bilinear interpolation. Round each resized dimension
   by `floor(original * scale + 0.5)` and clamp it to at least 256.
4. Take the integer-centered `256 x 256` crop.
5. Convert `uint8` RGB to `float32`, divide by 255, and apply no channel standardization.
6. Stack exactly one session as `[batch,channel,time,height,width] = [1,3,8,256,256]`.

Every emitted clip carries its session ID, official split, local role, `is_park`, full high-level
attributes, ordered frame indices, frame SHA-256 values, and referent IDs. The preflight also hashes
the resulting tensor, so the exact decoded development surface is independently comparable later.
One-session-at-a-time iteration bounds residency and leaves batching, augmentation, and model input
transfer outside this frozen source transform.

Reproduce the no-model preflight with:

```bash
PYTHONPATH=src .venv/bin/python scripts/sanpo_custom_substrate_bridge.py preflight \
  --plan configs/custom_substrate/sanpo_natural_bridge_v1.json \
  --proof proof/SANPO_CUSTOM_SUBSTRATE_BRIDGE_PREFLIGHT.json
```

## Two-stage artifact evaluation

The future evaluation interface is implemented but deliberately not executed yet. It is usable only
after CM7 produces a portable artifact whose complete evidence chain passes the independent verifier.
The portable loader re-verifies that chain before any natural-video forward.

### Stage 1: development-only selection

This stage decodes only train and validation. It runs one independently verified artifact one session
at a time, requires finite pooled retrieval keys, freezes two cosine centroids from the six train
sessions, and reports the two-session validation result as an interface diagnostic. Artifact
compatibility—not the tiny validation score—is the selection gate. The receipt explicitly records
that no official-test pixels were used.

```bash
PYTHONPATH=src .venv/bin/python scripts/sanpo_custom_substrate_bridge.py evaluate-development \
  --plan configs/custom_substrate/sanpo_natural_bridge_v1.json \
  --artifact-dir artifacts/custom_substrate/tiny-video-substrate-<artifact-id> \
  --selection-receipt proof/SANPO_CUSTOM_SUBSTRATE_DEVELOPMENT_SELECTION.json \
  --device cpu
```

The same command may use `--device mps` after the active training process has exited and normal host
preflight is green. Changing the artifact, source bytes, plan, or frozen centroid bytes invalidates
the selection receipt.

### Stage 2: explicit one-shot official test

This is a separate command and it refuses to run without the literal unlock flag. Before decoding the
first test PNG it exclusively creates the fixed ledger
`proof/SANPO_CUSTOM_SUBSTRATE_OFFICIAL_TEST_ONE_SHOT.json`. A failed attempt still consumes the
one-shot. An existing ledger refuses a rerun. The command requires the exact artifact, bridge plan,
content set, selection receipt, and frozen train centroids from stage 1, then applies them unchanged
to both test sessions.

```bash
PYTHONPATH=src .venv/bin/python scripts/sanpo_custom_substrate_bridge.py evaluate-official-test-once \
  --plan configs/custom_substrate/sanpo_natural_bridge_v1.json \
  --artifact-dir artifacts/custom_substrate/tiny-video-substrate-<artifact-id> \
  --selection-receipt proof/SANPO_CUSTOM_SUBSTRATE_DEVELOPMENT_SELECTION.json \
  --device cpu \
  --unlock-official-test
```

The result is permanently nonpromotable because `n=2`. It may not authorize tuning, architecture
changes, threshold changes, a second test run, a generalization claim, or an F8/F16 claim. If the
diagnostic exposes a problem, the honest next experiment requires a new preregistered natural-video
cohort with a new untouched test set; the two SANPO sessions do not become development data.

## Refusal surface

The bridge stops before decode on any proof-byte drift, schema failure, destination mismatch,
consumer content-set mismatch, missing or changed description/frame, unsafe path, wrong session or
role count, train/test crossing, attribute mismatch, reordered frame, referent mismatch, unbound
content row, intake-download SHA mismatch, Pillow-version drift, transform-plan drift, or plan-self
hash drift.

The future artifact stages add refusal for a missing independent-verifier promotion, artifact
manifest drift, non-finite or malformed pooled output, altered train centroid, selection receipt that
used test data, absent explicit unlock, and an existing one-shot ledger.

Focused verification commands are:

```bash
.venv/bin/python -m pytest tests/unit/test_sanpo_custom_substrate_bridge.py -q
.venv/bin/ruff check src/mop/substrate/sanpo_bridge.py \
  scripts/sanpo_custom_substrate_bridge.py tests/unit/test_sanpo_custom_substrate_bridge.py
.venv/bin/mypy src/mop/substrate/sanpo_bridge.py
```
