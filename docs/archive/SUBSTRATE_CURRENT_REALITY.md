# SUBSTRATE_CURRENT_REALITY

Verified state of `/Users/scammermike/Downloads/substrate` at HEAD `4f662307`,
branch `codex/odyssey-preflight-repair`, on 2026-08-24.

Every claim in §4 of the Ascension directive is answered below as
CLAIM / EVIDENCE / CURRENT STATE / ACTION / RECEIPT. Nothing here is carried from
the directive's own text; each row was measured.

Method: sixteen independent read-only Grok diagnostic lanes, plus direct
verification by the mutation authority of every load-bearing claim.

---

## A. Baseline facts the directive asserted

| # | Claim | Current state | Evidence |
|--:|---|---|---|
| 1 | working repo at `~/Downloads/substrate` | **TRUE** | `git rev-parse --show-toplevel` |
| 2 | HEAD `4f662307` | **TRUE** — `4f66230735dd067d076b8e87388ee006994b31d7` | `git rev-parse HEAD` |
| 3 | ~72,254 tracked files | **TRUE — exactly 72,254** | `git ls-tree -r --name-only HEAD \| wc -l` |
| 4 | `data` symlinks to corpdrive | **TRUE, and load-bearing through the loader** | G002 |
| 5 | ~96 GB corpus | **TRUE — 102,792,352,949 B apparent (95.733 GiB), 214,156 files** | G013 |
| 6 | ~4.4 TiB free on the drive | **TRUE — 4,540.4 GiB free; SSD 405.1 GiB free** | G014 |
| 7 | Odyssey NOT launched | **TRUE** — `ODYSSEY_7D.authority.json` absent, no worker run tree, `readiness` returns false | G010, G018 |
| 8 | G01–G15 sealed/pass | **TRUE — all fifteen `pass`, including G06-DC** | G018 |
| 9 | `launch_allowed: false` | **TRUE** — `ODYSSEY_PREFLIGHT.json:205` | G018 |
| 10 | preflight `admitted_waiting_for_authority_seal` | **TRUE** — same file, line 210 | G018 |

## B. Suspected problems the directive flagged

| # | Claim | Verdict | Current state | Action | Receipt |
|--:|---|---|---|---|---|
| 11 | old R2 86 GB acquisition mostly absent | **CONFIRMED, but harmless** | 17 of 18 archives gone (85,698,751,194 B, 99.61%); only `dev-clean.tar.gz` survives and hash-matches. **All 21 sealed source-selection assets are present and hash-match** — the loss was entirely outside the sealed set | none; re-acquisition is a WAVE-2 adoption question | `G012_R2_ARCHIVE_DISPOSITION.md` |
| 12 | 14/15 gate receipts + preflight untracked | **CONFIRMED, worse than stated** | **0 of 15** gate evidence files are tracked. G10 isolation observations are *gitignored*. HEAD additionally presents a **false G06-DC** bound to frozen `b282b4c6…` at commit `3841fd16` instead of the admitted `59f1fec2…` at `4f662307` | commit or bulk-manifest the baseline evidence (G004) | `ODYSSEY_READY_BASELINE_V1.json` |
| 13 | ~24,446 tracked deletions | **CONFIRMED — exactly 24,446** | A superseded density-era Lean 4.33.0-rc1 sandbox (`canary-B-*`, from `6dc1e066`) plus derived tool-cache IO. Candidate and control copies are identical blobs; 12,175 of 12,177 Lean blobs already exist under still-tracked `g06dc-B-*` | **keep them deleted** — the working tree is right and HEAD is wrong | G003 |
| 14 | 16 Ed25519 private keys in history | **CONFIRMED, plus a second generation** | 16 tracked at HEAD, introduced by `6dc1e066`; a further 16 uncommitted in the working tree. **Never pushed**: `git branch -r --contains 6dc1e066` is empty, `main` carries zero key paths | rotate locally; **do not push the four affected branches**; no history rewrite | `SECURITY_HISTORY_REWRITE_PLAN.md` |
| 15 | ~9.42 GiB Lean toolchain tracked | **CONFIRMED — 10,117,893,872 B across 59,280 files**, four copies, 2.63 GiB unique | `.git` is 2.36 GB | externalize to content-addressed bulk (G011) | G011 |
| 16 | venv 3.13.13 vs pin 3.12.13 | **CONFIRMED, now REPAIRED** | Canonical pin is CPython **3.12** (Makefile:7, portability.py:454, CI). Venv rebuilt to **3.12.13** | done | G007 |
| 17 | pytest absent from the venv | **CONFIRMED, now REPAIRED** | pytest 9.1.1 installed; `make test` runs | done | G007 |
| 18 | host Lean/docx incomplete | **CONFIRMED** | `lean`, `elan`, `lake` absent (`~/.elan` does not exist); venv lacked docx/whisper/sympy/torch. Present: z3 4.16.0, ffmpeg 8.1.2, Blender 4.2.1, docker, ollama, aria2c, git-lfs | restore Lean; declare the rest | G009 |
| 19 | `"activation": True` at `odyssey_detachment.py:541` | **CONFIRMED as a literal, REFUTED as a path** | A row of `ordered_external_steps` describing what a *human operator* does at step 3, in a document whose own `activation` is `False` and whose `forbidden_by_this_command` lists `launchctl_bootstrap`/`launchctl_kickstart`/`supervisor_start`/`worker_start`. **No reachable runtime path sets activation true.** The structural-audit test does fail, so `substrate audit` exits 1 | rename the descriptor key to `activates`; leave the audit check intact | G010 |
| 20 | tar backup existence unknown | **RESOLVED — it is gone** | `substrate-2026-08-24.tar.zst` absent; the drive root holds only an unrelated `legal-scans` tar. Corpus is single-copy | **user accepted single-copy risk permanently, on the record** | G015, G033 |

## C. Things the directive did not anticipate

| finding | current state | receipt |
|---|---|---|
| **The sealed model panel was gone from the host** | `gpt-oss:20b`, `qwen3:30b`, `deepseek-r1:32b` all absent. **Restored 2026-08-24** — 15/15 sealed layer blobs present, principal organ re-hashed to `e7b273f9…d09efb` matching the seal | `G082_MODEL_PANEL.md` |
| **The experiment cannot discriminate** | n=8 gives power 0.052 at SESOI 0.05. All eight generators sit at ceiling, floor, or answer-in-prompt. **Measured**: frontier B memoryless accuracy **1.000 (24/24)** | `G086_G087_DISCRIMINATION_DESIGN.md` |
| **Frontier E is the one working shape** | memoryless 0.583 vs a learnable 1.000; class is a pure function of `prior_commitment`, verified over 200 tasks. Learnable from history, not from one instance | same |
| **Density solves the power problem** | schedule uses 3.15% of measured organ capacity; more paired histories per frontier raise n without adding days | `G083_POWER_VIA_DENSITY.md` |
| **A hash-chained developmental log already exists** | `v5state.CognitiveEvent`/`PermanentEntity`, with parent digest and exact restore-by-reduction. Odyssey does not use it | G039 |
| **EWM and Self Model are records, not organs** | `WorldModel` is never imported by the runtime; `Substrate.step` never writes the `world` region; `SelfModel` records no facts in the live cycle | G035, G036 |
| **The canonical successor name** | `substrate-odyssey-ascension-v1`, not `ODYSSEY_WITNESSED_V1` — the latter would fail closed against `program.get("id") == PROGRAM` | G017 |
| **Host state is outside custody** | the R2 archives, `~/.elan`, and the ollama blobs all vanished together. The migration preserved repo and corpus but not host state that sealed receipts depend on | G019 |

---

## Incidents recorded during verification

**STALE_GIT_INDEX_LOCK.** `.git/index.lock`, **zero bytes, dated Aug 4**, with no live
git process. Read-only git commands worked throughout the session, so nothing surfaced
it; `git checkout --` was the first index-writing operation attempted and it failed.
Removed after confirming no git process held it. It follows that **no index-writing git
operation has succeeded in this repository since Aug 4**. Whether that contributed to the
24,446-path divergence between HEAD and the working tree is **not claimed** — it is
consistent with the timeline and remains an open question. Remediation queued as a
repository-doctor check that inspects lock age and live processes and **refuses automatic
deletion when ownership is ambiguous** (G092).

**FROZEN_IMPLEMENTATION_DRIFT.** `ODYSSEY_FROZEN_BUILD.json` pins seventeen modules via
`implementation_sha256`; `_validate_frozen_build` (`odyssey_authority.py:1010-1017`)
refuses machine-gate sealing on any digest change. Discovered by causing it: a one-key
rename in `odyssey_detachment.py` failed 12 tests. Reverted. See the frozen-implementation
boundary section of the ledger.

## Standing constraints

1. **Do not push** `codex/odyssey-preflight-repair` or the three `grok/odyssey-*` tips until key rotation and an index-hygiene test exist.
2. **Do not restore** the 24,446 deleted paths.
3. **Do not lower SESOI**, and do not promote events or microcycles to the unit of analysis.
4. **Do not treat a passing receipt as proof that its bytes exist.** Three separate losses this session followed that assumption.
5. **Do not decide blast radius from descriptor binding alone.** Module digest, frozen-build, environment, generated-artifact and transitive gate bindings all count. This rule exists because I violated it.
6. **Never merge the two proof objects.** The ancestor has fifteen green gates; Ascension has none until it earns them.

## Closeout state, 2026-08-24

Everything above was diagnosis. This is what changed by the end of the session.

**Repaired and verified**

| item | state |
|---|---|
| `.venv` | rebuilt to the canonical **3.12.13**; pytest 9.1.1; docx/openpyxl/pypdf/sympy at manifest versions |
| Lean | **4.33.0-rc1, commit 62eed1db** — restored from the repo-tracked toolchain for a 5.6 MB download instead of 2.6 GB |
| sealed model panel | restored; all 15 layer blobs at their sealed digests; principal organ content-hashed |
| brittle paths | all three repaired and relocation-tested; `grep -rn '/Users/scammermike' src/substrate/*.py` returns **zero** |
| git index writes | **PASS** — full add→reset round trip; the Aug-4 stale lock removed |
| SSD/HDD | verified by measurement; read **and** write through `data/` reach the drive |
| doctor | `tools/substrate_doctor.py`, 33 checks, **33 ok / 0 warn / 0 FAIL** |
| ancestor | **zero frozen drift** across all 17 pinned modules |
| activation | physically proven false through six checks — `ACTIVATION_FALSE_PROOF.md` |
| custody | 473 stranded objects digested and bundled (188 KB, round-trip verified) |
| security | generation-2 keys destroyed; ignore rules and a content-based regression test added |

**Deliberately not done** — each is an operator action or belongs to the next campaign:
the index/commit pass (which clears the 16 generation-1 keys and the 24,446 deletions in
one command), any history rewrite, the `odyssey_detachment` audit repair (frozen-pinned),
and the entire Ascension architecture.

**The finding that outranks all of the above:** the repository is healthy and the
experiment it hosts still cannot discriminate. n=8 gives power 0.052 at SESOI 0.05, and
all eight task generators sit at ceiling, floor, or answer-in-prompt — frontier B
measured at 24/24 for an organ with no memory at all. Fixing the machine did not fix the
science, and was never going to.
