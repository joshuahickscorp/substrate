# PERFORMANCE DENSITY DOCTRINE

Performance is not engineering-later. Performance is part of the science. A substrate that wins only
by more compute is not dense, and density is the one axis a small lab can actually win.

House style: no em or en dashes. Companion docs: FORM_SUBSTRATE_PROGRAM.md (worldview),
FORM_SUBSTRATE_DOCTRINE.md (methods), BLACKHOLE.md (the same philosophy applied to code volume).
This doctrine extends BLACKHOLE.md from code density to capability density: maximum capability,
minimum cost, provable by a number or it did not happen.

---

## 1. The three-number contract

Every experiment result reports three numbers or it is not a result:

```text
capability score   (what it can do)
cost score         (what that ability costs)
density score      (capability / cost)
```

A result that improves accuracy by 1 percent at 10x compute is not automatically progress. A result
that keeps accuracy equal and improves capability per FLOP by 1.25x may be a major result. A result
that routes expensive computation only when needed can matter more than a raw benchmark gain.

## 2. The metric families

Cost metrics, each with its measuring instrument in this repo:

| Cost axis | Instrument | Status |
|---|---|---|
| parameters, FLOPs, active FLOPs | src/mop/diagnostics/compute.py (D5 accounting; matched-compute check) | exists |
| bytes (caches, buffers, stores) | src/mop/substrate/storage.py (estimate_cache_bytes, list_caches_with_size) | exists |
| repo and artifact mass | src/mop/studio/density_receipt.py (mop-studio-scorecard receipts) | exists |
| wall-clock, throughput | harness timers + studio encode microbenchmarks | exists |
| peak memory (RSS, MPS) | src/mop/studio/memory_envelope.py | exists |
| retrieval latency | shell/buffer.py KVIndex (report faiss vs brute backend) | exists |
| accuracy-vs-cost frontier | src/mop/diagnostics/riskcov.py (Pareto frontier area) | exists |
| per-experiment density block | src/mop/diagnostics/performance_density.py | new (P0) |

Density ratios in use (pick at least one per experiment, preregister it in the registry row):

```text
accuracy / FLOP            transfer / parameter        retention / byte
adaptation / update        retrieval / millisecond     alignment / cache GB
planning gain / rollout FLOP     capability / wall-clock second
old-form retention / new-form cost     score / estimated energy proxy
```

## 3. Acceptance rules

A new architecture, mode, or form interface is accepted only if it improves at least one of:

1. capability at matched cost
2. cost at matched capability
3. stability at matched compute
4. transfer at matched data
5. retention at matched memory
6. operational-awareness score at matched token or FLOP budget

Matched means matched by construction (shell/capmatch.py, diagnostics/compute.py), not by assertion.
The strongest recorded lesson stands: the trained-router density mechanism lost to a compute-matched
homogeneous 40-copy bank on the real cache. Density claims die at matched compute unless they are real.

## 4. Registry wiring

Every active F-series registry row carries:

```text
primary capability metric   (metrics[0])
primary cost or density metric   (one of metrics[])
matched-cost baseline   (one of controls[])
```

F13 (form_energy_budget) is the dedicated bank experiment: sweep form width, token count, replay
bytes, shell size; report the Pareto frontier, never the peak. The null is that all form interfaces
lie on the same density frontier as raw features. If that null holds, form structure buys no
efficiency and the program says so.

## 5. Measured reality (the numbers the doctrine starts from)

- Encode and cache construction are measured costs, not assumed walls. The official dense ViT-B
  instrument strictly loads and completes a finite 64-frame CPU forward in about 25.2 seconds with
  about 1.33 GB maximum observed process-tree RSS. Encode once when scientific identity permits,
  stream caches, and retain full manifests.
- Dense tokens cost 8192 tokens per clip (storage.DENSE_TOKENS_PER_CLIP): dense caches dominate
  byte budgets; any dense claim reports alignment or transfer per cache GB.
- Hardware escalation target: none until a survivor repeatedly fails a named non-factorizable
  memory, intrinsic-latency, or inseparable synchronized-state requirement after local reductions.
  The current operational profile is a 300-minute adaptive leg with a 40 GB free-disk floor.
- The shell is tiny by doctrine; experiment cost is dominated by encode and cache I/O, so density
  wins come from interface and routing choices, not shell shrinkage.

## 6. Technology policy

Harness what exists, measure where it fails, invent only where the bottleneck is real. No GPU
defeatism, no GPU worship, no framework worship. Any new technology (MLX, custom Metal kernels,
sparse execution, quantized inference, compressed form stores) enters only with answers to:

1. What capability does it unlock?
2. What cost does it reduce?
3. What null result would make it not worth it?
4. What baseline must it beat?
5. Is it performance, science, or both?

Sparse execution claims follow the e7 lesson: sparse must beat parameter-matched dense, and report
active FLOPs, or the reduction was just capacity accounting.

## 7. Standing targets

Falsifiable efficiency targets for the form-substrate phase, each a preregistered claim, not a vibe:

- 1.25x capability per FLOP over the best single-form baseline on at least one F-series bed
- 1.5x retention per byte over stored-exemplar replay at matched retention (F11 lane)
- retrieval latency at or under the same-form kNN baseline while beating it on cross-form recall (F5)
- routing that spends expensive modes on at most the fraction of inputs where they pay (OA5 lane)

## 8. What refutes this doctrine's premise

If every density gain disappears under matched compute, matched capacity, and seed stability, then
the density program is bookkeeping, not science. Refutation clause 6 of FORM_SUBSTRATE_PROGRAM.md:
the honest conclusion would be that the current implementation is controlled multimodal
infrastructure with good accounting. That would still be useful. It would not be a discovery.
