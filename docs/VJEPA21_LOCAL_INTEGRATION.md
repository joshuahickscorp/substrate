# V-JEPA 2.1 local dense integration

## Outcome

V-JEPA 2.1 is not unpublished and is not a Studio-only model-availability problem. Meta released
four official 384px PyTorch checkpoints on 2026-03-16. The smallest is the approximately 80M
parameter ViT-B/16. Its exact source repository is pinned locally at commit
`204698b45b3712590f06245fbfba32d3be539812`, and its 1.664 GB checkpoint has a live, range-capable
official object. The remaining honest gate is local integration: full-byte acquisition, a locally
computed SHA256, strict `ema_encoder` load, and a real finite forward receipt.

This does not yet make E6 or DR14 scientifically complete. It retires only the stale upstream
availability premise.

Primary sources:

- [Meta's official V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2)
- [V-JEPA 2.1 paper, arXiv v3](https://arxiv.org/abs/2603.14482v3)

## Pinned authorities

| Authority | Pinned value |
|---|---|
| Official repository | `https://github.com/facebookresearch/vjepa2.git` |
| Repository commit | `204698b45b3712590f06245fbfba32d3be539812` |
| Commit time | `2026-03-23T10:13:05Z` |
| Paper | `arXiv:2603.14482v3`, updated 2026-06-11 |
| Paper PDF SHA256 | `6a7be4dbfd2131ef05640be457abef4cf57e1031dc97b894eab62c58782a3cac` (published as arXiv's ETag) |
| ViT-B checkpoint | `https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt` |
| Content-Length | `1,664,223,428` bytes |
| ETag | `"be0dc26f052ae6a7476714cd53176836-199"` |
| S3 version | `xJBU4AkoA4gv5boeC6gOA7eElzWcubxY` |
| Last-Modified | `Mon, 16 Mar 2026 09:12:19 GMT` |
| First 64 KiB SHA256 | `61efdf5f03e06d7d4dd6c4b35566dede6523453def98d51369e616093b23fbaf` |
| Last 64 KiB SHA256 | `cc219dadd7ab9e7dcb37121380e1a6ec24a7d8684fef2a8438e0e64e5c23ae59` |

The multipart ETag is not a file MD5. Meta does not publish a full checkpoint SHA256 in the pinned
README. The downloader therefore binds the live object to exact URL, length, ETag, S3 version,
last-modified time, and boundary-range hashes, then computes a full local SHA256 after the entire
transfer. A file without that receipt is rejected rather than adopted.

The majority of the official repository is MIT-licensed. Its README lists three Apache-2.0 source
exceptions, none imported by the encoder-only seam. The README publishes the model links but does
not state a separate checkpoint-weight license. The proof records that ambiguity; it does not
expand the source-code license into an unstated model-weight grant. The arXiv paper is marked
CC BY-NC-ND 4.0.

## Architecture and tensor contract

The official ViT-B builder fixes:

- ViT-B, 12 blocks, 12 attention heads, width 768;
- 3D convolutional tokenizer with tubelet 2 and spatial patch 16;
- 384px checkpoint resolution and a 64-frame configured temporal grid;
- 3D RoPE with interpolation and learned image/video modality embeddings;
- dense final output, not a pooled clip vector;
- strict checkpoint key `ema_encoder` for the distilled ViT-B release.

The direct model input is `B,C,T,H,W`. The inference output is
`B,(T/2)*(H/16)*(W/16),768`:

| Forward rung | Output tokens | Shape | FP32 dense payload |
|---|---:|---|---:|
| 8 frames, first runtime probe | 2,304 | `[1,2304,768]` | 7.08 MB |
| 16 frames | 4,608 | `[1,4608,768]` | 14.16 MB |
| 64 frames, configured grid | 18,432 | `[1,18432,768]` | 56.62 MB |

The paper motivates dense features through context-token supervision, deep self-supervision,
modality-specific tokenizers, and model/data scaling. Those training findings are requirements
evidence for the custom substrate, not proof that a particular MOP experiment improves merely by
switching encoders.

## Why the local loader bypasses pretrained torch.hub

At the pinned official commit, `src/hub/backbones.py` comments out the public base URL and activates
`VJEPA_BASE_URL = "http://localhost:8300"` for testing. Calling the documented entrypoint with
`pretrained=True` would therefore target a local test server. The repository source is still the
correct architecture authority, but its automatic download route is not usable at this commit.

The local seam consequently:

1. verifies the exact official repository commit and hashes every load-bearing source file;
2. constructs only the official `vit_base` encoder with the exact official kwargs;
3. does not instantiate the checkpoint's predictor or distillation teacher;
4. opens the separately verified archive with `weights_only=True` and `mmap=True`;
5. extracts and cleans only `ema_encoder`, following Meta's own key-cleaning rule;
6. calls `load_state_dict(..., strict=True)`, freezes every parameter, and switches to eval mode.

No fallback to a random architecture, Hugging Face placeholder, partial state load, or alternate
checkpoint is permitted on this path.

## macOS runtime boundary

Meta's README states that `decord` does not support macOS and that the project does not endorse a
specific replacement. The complete official training/evaluation data stack is therefore not
declared runnable here.

The encoder itself accepts an already-decoded tensor. MOP already has torchvision plus PyAV, and
the SANPO smoke corpus is a sequence of individual PNG frames. The selected local route is:

`rights-clean frames -> project decoder/preprocessor -> B,C,T,H,W tensor -> official encoder`

This bypasses decord without changing model code. The encoder-only imports need `torch`, `timm`,
and `einops`. MPS remains unverified: Meta strongly recommends CUDA, and neither the repository nor
paper guarantees Apple GPU support. CPU strict load comes first; an 8-frame CPU forward comes next;
MPS is a later measured rung, never an assumption.

## Host and disk feasibility

The repository checkout occupies about 11 MB. The checkpoint transfer writes one resumable `.part`
file and atomically renames it, so it does not require two checkpoint-sized copies. The preflight
requires:

`40.000 GB floor + 1.664 GB remaining checkpoint + 0.512 GB working headroom = 42.176 GB free`

The live preflight records the current value rather than freezing it in prose. Download must be
delayed or aborted if the projected post-transfer free space falls below 40 GB. ViT-L, ViT-g, and
ViT-G are discovery-only rows until ViT-B passes strict load and forward.

## E6 impact

E6's registry classification changes from “unpublished model” to “weights-needed local
integration.” Its current runner is not ready for the real dense model:

- it calls the general Hugging Face/frozen-random loader, not the pinned official loader;
- it creates 4-frame 16px synthetic tensors rather than a citable 384px natural-video cache;
- it flattens dense `[B,N,D]` features to `[B,N*D]`;
- at the configured 64-frame grid that is 14,155,776 features per clip, making the current first
  linear layer enormous and destroying token structure;
- it directly encodes all 192 examples rather than streaming a bounded cache;
- its current structured features are elementwise products/differences between two flattened clip
  vectors, not an object/relation readout over spatial-temporal tokens.

So the correct E6 migration is cache-first and token-aware: stream small batches, preserve the
token grid and referents, use a preregistered bounded readout (for example fixed-rank spatial bins or
slots), parameter/compute-match the flat control, and retain a matched random-architecture cache.
The official ViT-B forward is requirements evidence and an integration milestone, not an E6 result.

## DR14 impact

DR14's existing runner implements low-rank, quantization, and noise corruption on 2D pooled
features. Its prose mentions a dropped-channel dense arm, but that arm is not implemented. The
remaining blocker is therefore twofold: a citable dense cache and local corruption mechanics that
preserve token layout while dropping preregistered channel/token groups. It is not a measured
hardware boundary.

A safe first dense pilot can use the ViT-B 8- or 16-frame cache, but scientific promotion still
requires DR14's power floor, natural referents, shared corruption tensors, difficulty calibration,
and matched single-pass control.

## Exact command sequence after CM7

The source checkout and metadata preflight are light and can run during CM7. Keep weight transfer
and every model command serial after CM7 exits:

```bash
# 1. Refresh the local doctor immediately before the acquisition decision.
PYTHONPATH=src .venv/bin/python scripts/studio_doctor.py \
  --profile m3pro-local-max --out proof/STUDIO_READINESS_CURRENT_HOST.json

# 2. Re-run live source/object/disk checks. ready_to_download must remain false while CM7 runs.
PYTHONPATH=src .venv/bin/python scripts/vjepa21_official.py preflight

# 3. Install only encoder imports if the preflight lists either as missing. This does not install decord.
uv pip install --python .venv/bin/python -e '.[vjepa21]'

# 4. Refresh the doctor after environment changes, then require encoder_only_ready=true,
#    heavy_lane.clear_for_new_heavy_lane=true, and ready_to_download=true.
PYTHONPATH=src .venv/bin/python scripts/studio_doctor.py \
  --profile m3pro-local-max --out proof/STUDIO_READINESS_CURRENT_HOST.json
PYTHONPATH=src .venv/bin/python scripts/vjepa21_official.py preflight

# 5. Explicitly acquire only ViT-B. This computes the full local SHA256.
PYTHONPATH=src .venv/bin/python scripts/vjepa21_official.py download

# 6. Strict-load ema_encoder in a supervised CPU child, no forward yet.
PYTHONPATH=src .venv/bin/python scripts/vjepa21_official.py probe \
  --mode load --device cpu --timeout 1800

# 7. Smallest real 384px dense forward, expected finite [1,2304,768].
PYTHONPATH=src .venv/bin/python scripts/vjepa21_official.py probe \
  --mode forward --device cpu --frames 8 --timeout 3600
```

Only after both proof files pass should the queue consider a 16-frame cache, a 64-frame forward,
or ViT-L metadata/download. No ViT-L/g/G command is exposed by the downloader today.

## Proof semantics

`proof/VJEPA21_VITB_LOCAL_PREFLIGHT.json` is metadata/source/disk evidence only. `all_ok` means the
cheap official source/object/config/disk checks pass. It is not download permission. The separate
`gates.ready_to_download` field additionally requires a green local doctor no more than 15 minutes
old and no active heavy lane; it must stay false while CM7 runs. The proof explicitly records
`model_loaded=false`, `forward_executed=false`, and both E6/DR14 compatibility claims as false.

Future load and forward receipts will be `proof/VJEPA21_VITB_LOAD.json` and
`proof/VJEPA21_VITB_FORWARD.json`. A timeout alone is not an out-of-memory result. Larger variants
remain gated unless the finite-shape forward passes.
