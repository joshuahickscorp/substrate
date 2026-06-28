# APPLE_SILICON

This project is Apple-Silicon-native. The frozen V-JEPA substrate and the whole trainable shell
run on Metal (MPS) as the primary target: an M3 Pro laptop today, a bigger M-series Mac Studio
later. The Studio is NOT a CUDA box. cuda stays supported only for rented GPUs used by Tier R
environment rollouts (E5/E10), which Metal cannot cover cheaply.

## Why MPS-first
- Unified memory. There is no host/device copy and no pinning: the encoder, the cached latents,
  and the shell share one pool. `pin_memory: false` on Apple Silicon. A Mac Studio M-series Ultra
  has far more unified memory than the 18 GB laptop, so the scale-up is "same MPS code, bigger
  chip", not a port.
- fp16 on Metal is fast and halves memory. `devices.autocast(info)` wraps inference (the frozen
  encoder, probes) in fp16 on mps/cuda and is a no-op on cpu. `configs/device/mps.yaml: amp=true`.
- The device layer detects the machine: `devices.apple_silicon_info()` returns chip, performance
  and efficiency core counts, and unified memory. On this machine: Apple M3 Pro, 6P + 6E, ~19 GB.

## The device boundary
Everything device-touching goes through `devices.resolve(cfg.device.kind)`. `auto` picks mps on
Apple Silicon (then cuda, then cpu). Unsupported Metal ops fall back to cpu via
`PYTORCH_ENABLE_MPS_FALLBACK=1` (set automatically) and `devices.safe_to` / `autofallback`, so a
patchy op degrades, never crashes. `device=cpu` forces the deterministic, bit-identical path
(see the determinism study: CPU is bit-identical run-to-run, Metal is ~50% at temp 0).

## CPU parallelism (the campaign harness)
`harness/cpu_pool.py` saturates the CPU as a process pool sized to the physical cores (6P + 6E =
12 here), with per-worker BLAS thread caps so workers x threads stays near the core count (the
oversubscription trap). Heavy legs use fewer workers x more threads; small-head legs use many
workers x 1 thread. Sustained load thermal-throttles a laptop, so measured CPU timings are
conservative versus the better-cooled Studio (the cost projection labels them laptop-throttled).

## Known Metal limitation (measured)
A 64-frame ViT-L forward (8192 tokens) hangs the MPS graph compiler on this M3 Pro. Real-encoder
latent caching therefore runs on cpu (measured 24 to 32 s/clip; see STATUS.md and the
retrospective ledger). The
`mps_safe_token_cap` knob in `configs/device/mps.yaml` documents the threshold above which an
encoder forward should be routed to cpu. Standard heads, predictors, ensembles, and the shell run
fine on mps. On a Mac Studio with more GPU cores this limit is expected to lift; verify before
trusting MPS for large-token encoder forwards.

## MLX (optional, future)
Apple's MLX can accelerate encoder inference throughput on Apple Silicon and is installable via
the `apple` extra (`uv pip install -e ".[apple]"`). It is NOT a dependency and not on the hot
path: PyTorch-MPS is the safe default (the V-JEPA HF integration is PyTorch). Treat MLX as a
profiled-bottleneck throughput experiment for encoder caching, not a rewrite. Do not let it become
a yak-shave.

## What to flip for the Studio (Apple Silicon)
- Keep `device=mps`. Raise batch sizes and dataset/cache sizes to use the larger unified memory.
- Use the bigger encoder (`encoder=vjepa2_vitg`) and the 2.1 dense variants; with more GPU cores
  the 64-frame forward should run on Metal directly.
- Rented CUDA (`device=cuda`) is only for the Tier R env-rollout legs. Everything else stays MPS.
See SCALING.md for the full flip-list.
