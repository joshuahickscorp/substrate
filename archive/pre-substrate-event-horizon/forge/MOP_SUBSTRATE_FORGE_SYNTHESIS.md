# MOP Substrate Forge: Boundary Synthesis

Program `mop-substrate-forge-v1`, branch `agent/mop-substrate-forge` off `3b86774`. Prior evidence immutable;
collapse branch untouched. Activation false. A tie is a null. This is a BOUNDARY CHECKPOINT, not a global
Substrate Event Horizon (the section-28 work floor is incomplete).

## Execution-context note
Executed in a bounded interactive session, not a persistent 24-hour daemon. The substantive new Forge science
was done across turns with detached compute and checkpointed. Continuous 24-hour daemon operation requires the
user to keep driving the session; that is the only genuine blocker to literal continuous execution.

## The decisive new science (both terminal, independently verified)
1. Memory-compression frontier (HAR raw, LSTM, gap none 0.369 -> full 0.734 = 0.364): **compression_null.**
   Bounded compression (prototype k-means class-means, coreset k-center, dual recent+prototype = Architecture E)
   does NOT robustly beat GDumb. A 5-seed frontier showed an apparent advantage (prototype +0.027, coreset
   +0.058) that VANISHED at 8 seeds (prototype cap375 mean +0.037 but lower-95-CB -0.000; coreset null or
   negative). This re-confirms the C1 stable-seed lesson and closes a genuinely new lever: memory
   representation/compression, distinct from the already-closed R1/P1R learned-retrieval question. The memory
   bottleneck is genuinely capacity, and GDumb is already the best bounded policy.
2. PAMAP2 second domain: **invalid_no_temporal_headroom.** The bag-of-timesteps (0.776) beats the GRU (0.702)
   and shuffling timesteps does not hurt, so windowed PAMAP2 activity recognition is order-insensitive. A
   required domain was attempted and honestly sealed invalid.

## Why no global event horizon
Section 28 requires all three domains terminal (audio not attempted, blocked by absent torchaudio/librosa),
A-T/E/F terminal (F not evaluated), all improvement rounds, cross-domain persistence (blocked: PAMAP2 invalid,
so no second valid temporal domain), and three-role audit agreement. Not satisfied. The HAR-scoped prior result
cannot satisfy a global claim, and this checkpoint does not claim one.

## Resumable work floor
Architecture F (Predictive Consolidation); A-T improvement rounds A-T1/A-T2/A-T3; a lawful numpy-loadable audio
feature source (current libs block audio); the full memory-frontier budget grid; cross-domain persistence
pending a second valid temporal domain; expanded property/mutation tests.

## Evidence ceiling (unchanged, reinforced)
No substrate architecture, timescale, memory policy, or compression strategy beats strong matched conventional
alternatives. The memory-capacity gap is real but is not exploitable by bounded representation any better than
GDumb. Activation false.

## Resume commands
- resume: read MOP_SUBSTRATE_FORGE_STATE.json; branch agent/mop-substrate-forge, worktree /Users/scammermike/Downloads/mop-substrate-forge
- verify: python (independent recomputation in MOP_SUBSTRATE_INDEPENDENT_VERIFICATION.json)
- status: cat runs/substrate/mop-substrate-forge-v1/status.json
