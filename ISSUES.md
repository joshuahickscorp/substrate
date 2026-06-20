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

## Degraded
(none)

## Expected failures (xfail, kept in the suite)
(none currently; any flaky-on-Metal exact assertion is converted to a tolerance from the
 determinism utility rather than xfailed.)
