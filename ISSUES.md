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

## Degraded
(none)

## Expected failures (xfail, kept in the suite)
(none currently; any flaky-on-Metal exact assertion is converted to a tolerance from the
 determinism utility rather than xfailed.)
