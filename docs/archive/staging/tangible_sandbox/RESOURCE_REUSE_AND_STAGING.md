# Tangible Sandbox resource reuse and staging

**Status:** operational inventory recorded while R2 is live. This document
does not amend R2, authorize a new source, or alter its sealed storage guard.

## Retention

Keep the R2 acquisition roots intact. The next blind shadow reuses the
validated archives, caches, processed derivatives, source checkouts, and
STSC-1 artifacts. Its run root is new, but it does not duplicate the retained
source material. On the same filesystem, the retained project size is already
represented in free-space measurement and must not be subtracted a second
time.

| Retained resource | Present locally | Next-use role |
| --- | ---: | --- |
| FSD50K + LibriSpeech archives/cache | 80.13 GiB downloaded, 18 validated archives | audio/speech stimulus bank, subject to the existing license/filter ledger |
| Pinned source repositories | 11 complete checkouts | adapters, task schemas, and environment preparation |
| LongMemEval-V2 public material | about 1.14 GiB under the existing public root | longer-history/task metadata candidate |
| STSC-1 builder-visible artifact base | present | zero-download new tangible task composition |

## Guard policy

**Active R2:** do not change its guard. It is bound to the dynamic storage
seal and currently protects approximately 186.27 GiB free space. The current
headroom is about 247.7 GiB, but it is on the same APFS volume as unrelated
work, so it is not a license to consume all of that space during a live
continuity measurement.

**Next shadow:** measure again immediately before preflight. Its guard is:

```text
max(20% of volume, 50 GiB)
+ selected new-source bytes still to acquire
+ measured own-run growth
+ measured peak transient writes
+ post-run clean-clone reserve
+ explicit user reserve
```

The next control plane computes this from the selected adoption cards. Existing
same-volume project bytes appear once through `df` free space; only newly
selected source bytes are added to the reservation.

## What can safely be prepared now

1. Use STSC-1 recomposition as the default shadow stimulus bank. It needs no
   download and can create genuinely new task/history compositions after R2
   verifies.
2. Preserve the current archive and Git pin receipts as the source basis for
   the next data-adoption cards. Do not rehash tens of gigabytes while R2 is
   running; the existing validation receipts already carry the digests.
3. Keep LongMemEval-V2, tau2, the audio/speech material, and existing tangible
   artifacts as candidate banks. Their next-run selection still needs parity,
   evaluator-only split, and task-level hash commitments.
4. Run only cheap control-plane checks while R2 is live: lifecycle tests,
   adapter-contract checks, metadata inventory, and draft review. The real
   1/2/4-worker calibration stays deferred because it intentionally consumes
   CPU and I/O.

## Deliberately deferred until R2 is complete

| Candidate expansion | Why it is not bulk-fetched during R2 |
| --- | --- |
| WebArena / SWE-bench environments | environment images and Docker state can create large Colima/APFS drift; this was implicated in invalid earlier continuity attempts |
| AI2-THOR / TEACh / ProcTHOR | needs runtime assets and/or simulator setup, not just the already-pinned code |
| AndroidWorld | requires an Android Studio/AVD or experimental Docker path; the documented baseline is about 8 GiB disk before task work |
| Kubric | needs Docker/Blender and separately licensed or remote assets |
| OSWorld, WorkArena, MLE-bench, Common Voice | gated access, instance, competition, or dataset terms must be accepted and recorded before acquisition |

These are candidates for a later breadth/interactivity arm, not prerequisites
for the first blinded 24-hour shadow. This preserves causal clarity: the
shadow changes task composition and blinding, while a future interactive or
training arm can precommit its own data and environment intervention.
