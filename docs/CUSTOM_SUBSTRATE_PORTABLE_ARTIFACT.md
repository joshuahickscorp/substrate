# Portable custom-substrate artifact

## Outcome

CM7 now has an independent-platform exit seam. A completed run can be converted into a compact,
content-addressed inference artifact containing one independently selected online model, its exact
evidence chain, and a stable interface for dense spatiotemporal tokens plus a pooled retrieval key.
The runtime is implemented locally with PyTorch and the standard library. It does not import an
inherited encoder, Transformers, Hugging Face, or the CM7 workbench.

No canonical artifact has been exported yet. Export is intentionally impossible until the long CM7
run is complete, its current-evidence attestation passes, and a separate verifier emits the strict
receipt described below.

## Why this is a platform seam

The portable artifact carries the trained substrate rather than a dependency on the platform used to
develop its requirements. It freezes four contracts:

1. **Architecture identity.** The full tubelet/patch geometry, width, depth, heads, MLP ratio, input
   maxima, parameter count, and training-geometry token count are bound to the selected state.
2. **Representation interface.** `forward` returns named `dense_spatiotemporal_tokens` and
   `pooled_retrieval_key` tensors. Dense tokens preserve time-major, row-major, column-major patch
   order; the pooled key is their arithmetic mean.
3. **Scientific identity.** The immutable raw training receipt (R), current-evidence attestation (E),
   environment receipt (H), independent verifier (V), final composite receipt, selected arm,
   checkpoint, frozen configs, and source snapshots travel with the model.
4. **Claim boundary.** CM7 remains deterministic programmatic-video evidence. Export does not turn it
   into natural-video, general-capability, intelligence, or sentience evidence.

The objective predictor remains in the state so the exported state hash exactly equals the evaluated
online checkpoint. The public inference interface exposes encoder tokens and their pooled key, not
objective-specific predictor activations.

## Fail-closed selection

The raw training receipt cannot authorize its own export, and the final composite is never mistaken
for raw evidence. The exporter requires the declared immutable chain:

- immutable `raw_workbench_receipt.json` with hash R and preliminary, non-authoritative promotion;
- `current_evidence_attestation.json` with hash E, bound to R and the current-audit hash;
- `environment_receipt.json` with hash H, bound to R and the implementation manifest;
- `independent_verifier.json` with hash V and schema
  `mop-custom-substrate-cm7-independent-verifier/v1`, bound to R/E/H; and
- final `workbench_receipt.json`, which preserves every parsed raw field, binds R/E/H/V, and adds the
  only authoritative promotion block.

The verifier must recompute the three learned objectives from raw R, apply the declared family of 12
comparisons with Holm one-sided tests and simultaneous Bonferroni Student-t lower bounds, and return
`promote-local-objective-lever`. The exporter then selects the **lowest complete seed for that
familywise-corrected winning objective** and independently checks its arm, checkpoint, and online
state.

The fixed seed rule prevents choosing the most flattering seed after seeing results. Export always
uses the online `model` state because that is the component CM7 evaluates; it refuses the EMA target,
random-target control, frozen control, a non-winning objective, or a different seed-selection rule.

Print the exact machine-readable verifier contract with:

```bash
PYTHONPATH=src .venv/bin/python scripts/custom_substrate_artifact.py contract
```

Content hashes establish integrity and drift detection, not signer identity. Independence therefore
still depends on the separate verifier implementation and review authority; the R/E/H/V chain makes
that assumption auditable rather than silently treating raw training output as verified.

## Deterministic artifact format

The exporter writes a directory named from the SHA-256 of the complete manifest identity:

```text
tiny-video-substrate-<artifact-id>/
├── manifest.json
├── weights.mopbin
├── evidence/
│   ├── workbench_receipt.json
│   ├── raw_workbench_receipt.json
│   ├── current_evidence_attestation.json
│   ├── environment_receipt.json
│   ├── independent_verifier.json
│   └── selected_arm_receipt.json
├── provenance/
│   ├── resolved_config.json
│   ├── dataset_manifest.json
│   ├── requirements_audit.json
│   ├── requirements_current_audit.json
│   ├── implementation_manifest.json
│   ├── teacher_audit.json
│   ├── implementation/...
│   └── requirements/...
└── runtime/
    └── custom_artifact.py
```

`weights.mopbin` is a deterministic little-endian tensor table: a canonical JSON header followed by
sorted raw tensor bytes. Loading it does not call `torch.load`, pickle, a model hub, or the network.
The loader bounds and validates the header, requires unique sorted names and contiguous offsets,
checks every tensor size, recomputes the online state hash, constructs the portable architecture, and
loads it strictly. It then rechecks every bundled provenance file and the receipt/verifier/selection
cross-bindings before returning the frozen evaluation-mode model.

The manifest has no export timestamp or machine path. Re-exporting the same verified inputs and
runtime source produces the same weights, manifest, artifact id, and directory contents. Re-exporting
into an existing destination reuses it only after a full offline verification.

## Future execution

After the citable 180-minute run finishes and its receipt-chain finalizer has written R/E/H/V plus the
composite, the exact non-writing audit is:

```bash
PYTHONPATH=src .venv/bin/python scripts/custom_substrate_artifact.py preflight \
  --run-dir runs/custom_substrate/cm7_local180_citable_v3 \
  --verifier runs/custom_substrate/cm7_local180_citable_v3/independent_verifier.json
```

Only if `eligible` is `true`, export the canonical artifact:

```bash
PYTHONPATH=src .venv/bin/python scripts/custom_substrate_artifact.py export \
  --run-dir runs/custom_substrate/cm7_local180_citable_v3 \
  --verifier runs/custom_substrate/cm7_local180_citable_v3/independent_verifier.json \
  --output-root artifacts/custom_substrate
```

The command prints the exact content-addressed directory. Verify and optionally exercise its public
interface offline with:

```bash
PYTHONPATH=src .venv/bin/python scripts/custom_substrate_artifact.py verify \
  --artifact-dir artifacts/custom_substrate/tiny-video-substrate-<artifact-id> \
  --device cpu --smoke
```

Programmatic use is deliberately small:

```python
from pathlib import Path
import torch

from mop.substrate.custom_artifact import load_portable_artifact

loaded = load_portable_artifact(Path("artifacts/custom_substrate/tiny-video-substrate-<artifact-id>"))
clips = torch.zeros(1, 3, 8, 256, 256)
with torch.inference_mode():
    output = loaded.model(clips)
dense_tokens = output.dense_spatiotemporal_tokens
pooled_key = output.pooled_retrieval_key
```

## Refusal matrix

Export or load stops on the first violated identity. Covered refusals include:

| Failure | Result |
|---|---|
| missing, incomplete, resumable, wall-stopped, or disk-stopped workbench receipt | refuse |
| R/E/H/V file absent, changed, cross-bound incorrectly, or composite differs from raw | refuse |
| authoritative composite promotion is false or treats raw promotion as final | refuse |
| verifier is uncorrected, non-promoting, or selects an invalid objective family | refuse |
| selected arm is incomplete or differs from the embedded arm record | refuse |
| checkpoint path substitution, byte drift, schema drift, or identity drift | refuse |
| online state differs from the selected raw arm state hash | refuse |
| receipt, config, model spec, parameter count, or token geometry disagree | refuse |
| frozen implementation or requirements snapshot changes | refuse |
| programmatic evidence is labelled as natural or general evidence | refuse |
| exported manifest, weights, or bundled provenance bytes change | refuse |

These are artifact-integrity gates, not a new scientific result. Natural-video adaptation, richer
objectives, memory, actions, audio, or a later architecture can now target this stable local interface
without inheriting a remote model runtime, while retaining exactly which CM7 evidence justified the
starting state.
