# V-JEPA 2.1 local dense integration

## Outcome

V-JEPA 2.1 is not unpublished and is not a Studio-only model-availability problem. Meta released
the official 384px PyTorch ViT-B checkpoint on 2026-03-16. This approximately 80M parameter
instrument is the only live model in the integration seam. Its exact source repository is pinned at
`204698b45b3712590f06245fbfba32d3be539812`. Its 1.664 GB checkpoint is retained locally with full
SHA256 `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`.

Local integration is complete for the official ViT-B encoder seam: strict `ema_encoder` load
passed, all 86,833,152 parameters remained frozen, and finite CPU forwards passed at both 8 frames
(`[1,2304,768]`) and the configured 64 frames (`[1,18432,768]`). The task layer is also wired for
serial resumable learned/random dense caches. The first remaining E6/DR14 boundary is a citable
rights-clean natural task cohort, not checkpoint availability or Studio compute.

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
| Retained full local SHA256 | `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d` |
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
paper guarantees Apple GPU support. CPU is already sufficient for strict load and measured 8-frame
and 64-frame forwards. MPS remains an optional measured optimization, never a prerequisite.

## Host and disk feasibility

The repository checkout occupies about 11 MB. The checkpoint was acquired through one resumable
`.part` file and atomically renamed, so it did not require two checkpoint-sized copies. The
acquisition preflight required:

`40.000 GB floor + 1.664 GB remaining checkpoint + 0.512 GB working headroom = 42.176 GB free`

That gate is historical for the retained ViT-B bytes. Dense cache construction independently keeps
a 40 GB floor and projects each arm before allocation. No larger model is part of the live E6 or
DR14 roadmap.

## E6 impact

E6 is now classified `environment-needed`, not `weights-needed` or Studio-scale. The registered
runner defaults to cache-first and refuses a legacy fallback. Its bounded readout preserves fixed
token-position bins, uses frozen low-rank projections, and compares against parameter-matched
learned-flat, token-shuffle, and exact-architecture seeded random-init controls.

One immutable preprocessed tensor manifest supplies both encoders in exact referent order. The
manifest hashes source authority, view recipe, split membership, annotations, every tensor file,
and every decoded tensor. Cache progress hashes every output row and resumes only when the full
identity matches. This retires the former generic-loader, flattening, and direct-encode flaws.

It does not turn the forward receipts into an E6 result. Scientific promotion still needs at least
200 rights-clean natural clips, nondegenerate annotated factors, disjoint combination splits,
untouched test membership, serial learned/random cache materialization, and independent statistical
verification.

## DR14 impact

DR14 now implements low-rank, quantization, noise, and dense dropped-channel arms. The dense arm
accepts only a citable cache by default, verifies the exact ordered output-row hashes, constructs
deterministic nested channel groups, and gives both reasoning arms the same materialized corrupted
view. It trains the tied iterative and FLOP-matched untied single-pass controls on the clean view.

The runner remains a pilot surface and explicitly refuses scientific promotion. A natural task
cache, power floor, difficulty calibration, and independent statistics remain required. This is an
environment/evidence boundary, not an unimplemented mechanics or measured hardware boundary.

## Completed integration sequence and current task preflight

The following serial sequence produced the retained checkpoint and runtime receipts. It is
historical evidence, not work that remains to be repeated:

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

# 8. Configured-grid dense forward, expected finite [1,18432,768].
PYTHONPATH=src .venv/bin/python scripts/vjepa21_official.py probe \
  --mode forward --device cpu --frames 64 --timeout 3600
```

The current safe task command is metadata-only and can run while another heavy lane owns compute:

```bash
PYTHONPATH=src .venv/bin/python scripts/vjepa21_dense_tasks.py preflight \
  --proof proof/E6_VITB_DENSE_PREFLIGHT.json
```

It does not construct a model, read checkpoint tensor bytes, or execute a forward. No additional
model acquisition command is exposed by this integration.

## Proof semantics

`proof/VJEPA21_VITB_LOCAL_PREFLIGHT.json` remains metadata/source/disk evidence only. The runtime
claims are instead supported by `proof/VJEPA21_VITB_LOAD.json`,
`proof/VJEPA21_VITB_FORWARD.json`, and `proof/VJEPA21_VITB_FORWARD_64F.json`. They bind one retained
checkpoint hash to strict load, frozen parameters, and the two measured finite shapes.

`proof/E6_VITB_DENSE_PREFLIGHT.json` is task-integration evidence only. Its `all_ok` means runtime
authority, registration, cache/control interfaces, and no-heavy behavior are coherent. It does not
mean the natural input manifest or either dense cache exists, and it always records
`scientific_promotion=false` until those independent evidence gates are satisfied.
