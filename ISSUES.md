# ISSUES

Degraded / deferred / expected-failure items, each with reason + unblock step.
Empty of hard failures means a clean run.

## Deferred (environment, not a defect)
- Real V-JEPA latent caching: encoder weights not fetched in this session (no model
  download). Synthetic-latent path is operational and all downstream is built/tested on
  it. Unblock: `uv pip install -e ".[encoder]"` + network, then
  `python scripts/cache_latents.py encoder=vjepa2_vitl_fpc64_256 +source=<videos>`.
- Tier R campaign legs (env rollouts + capstone): queued with enabled=false; need a
  rented CUDA box and a real environment. Unblock: set enabled=true on the Studio and
  provide the env adapter.
- V-JEPA 2.1 dense weights: NOT published on HF under any verified id (probed 2026-06; the
  `vjepa21_*` configs carry placeholder ids + `available: false`). The 2.1-only experiments
  (E6 dense vs pooled) stay deferred. Unblock: when 2.1 ships, set the real hf_id +
  `available: true` + `prefer_real: true`. Real V-JEPA 2 ViT-L/H/g ids ARE verified present.
- Real natural-video decode: `substrate/video.py` is backend-agnostic and lazy-imports
  torchvision/decord; neither is a hard dep. Unblock: `uv pip install -e ".[video]"`. The
  preprocessing core is tested today; full decode is exercised on the Studio with real clips.

## Deferred (acquisition lane, gated by design, not a defect)
- Heavy dataset downloads: every remote source in `registry/datasets.yaml` is DRY-RUN by
  default in `scripts/studio_pipeline.py acquire`. Real bytes need `--execute --budget-gb N`
  and, for any source with terms, `--accept-license`. On the current device the downloader has
  no credentials/fetcher for remote methods, so remote sources record a clean `blocked` status
  (never a partial). Unblock on the Studio: provide credentials/tools, then run acquire --execute.
- Manual-license sources (ssv2, ego4d_subset): status `manual`; the planner will not select
  them unless the profile allows manual auth AND `--accept-license` is passed. Unblock: complete
  the signed terms (see each source's `auth_steps` / the license ledger), then re-plan.
- Blocked / deferred sources (laion_tiny_meta blocked; ego4d_full, ego_exo4d_subset deferred):
  full Ego4D and Ego-Exo4D are beyond a 1 TB local disk and are NEVER planned by default; LAION
  is withdrawn pending a safety-reviewed re-release. These are intentional, surfaced in the
  license ledger, and require dedicated storage / a re-release before they could be considered.
- Real remote fetch + archive extraction: the downloader detects unsafe archive members
  (path traversal / absolute paths) but the actual streaming fetcher is a Studio-side callback
  (not implemented on this device, by design). Unblock: supply a `fetch_fn` with credentials.

## Degraded
- faiss 1.14 + torch on Apple Silicon: `faiss.search()` HARD-SEGFAULTS (rc 139, dual OpenMP
  runtime) when run after torch is imported, and a segfault cannot be caught in-process.
  MITIGATED, not a live risk: `KVIndex` probes faiss.search safety once in a subprocess
  (`shell.buffer.faiss_search_safe`) and falls back to EXACT brute-force retrieval if unsafe;
  the buffer default index is now `brute`. Correctness is never affected (brute is exact); only
  large-scale retrieval speed, which is a Studio concern. Unblock: re-probe on the Studio (more
  GPU cores / a faiss build without the OpenMP clash may make faiss.search safe there).

## Expected failures (xfail, kept in the suite)
(none currently; any flaky-on-Metal exact assertion is converted to a tolerance from the
 determinism utility rather than xfailed.)
