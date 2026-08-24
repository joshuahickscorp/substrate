# SUBSTRATE_ASCENSION_HANDOFF_V1

Written at the close of the pre-Ascension session, 2026-08-24. Read this first.

**The one sentence:** the repository works and the old Odyssey ancestor is intact and
provably undrifted, but the experiment that ancestor was built for cannot answer its own
question — so the next campaign is building a new subject, not launching this one.

---

## 1. Repository location

```
/Users/scammermike/Downloads/substrate
```

Do **not** move it under `~/Desktop` or `~/Documents` — both are iCloud-synced, and an
iCloud eviction is what began this whole recovery. The doctor asserts this.

## 2. Storage layout

| | path | holds |
|---|---|---|
| **SSD** | `~/Downloads/substrate` | source, `.git`, tests, `.venv`, receipts, run metadata |
| **Bulk** | `/Volumes/corpdrive/substrate/data` | 96 GB corpus, large immutable assets |

`data/` is a **symlink** to the bulk volume. Measured: SSD pack-sequential 4,797 MiB/s
and 4 KiB random p95 0.13 ms, versus 136 MiB/s and 12.4 ms on the USB volume. **Keep any
fsyncing trace on the SSD**; moving it to the drive is a two-orders-of-magnitude latency
mistake. Free: ~342 GiB SSD, ~4,540 GiB bulk.

The drive also carries a **second full checkout** of the repo at the same HEAD. It is a
stale backup, not the architecture. Do not work in it.

## 3. Branch / HEAD — CONSOLIDATED ONTO MAIN

```
branch  main   (tracking origin/main, 0 ahead / 0 behind)
HEAD    17cbc9ad  "Land the pre-Ascension closeout"
```

The work was landed on `origin/main` as **one commit on top of the real
`c7343c75`** — a plain fast-forward, no force-push, and the other 29 remote
branches are untouched.

**Why not the branch's own history:** `codex/odyssey-preflight-repair` contained
commit `6dc1e066`, which added sixteen unencrypted Ed25519 private keys, plus a
Lean toolchain with two blobs above GitHub's 100 MB hard limit (`libLean.a` at
202 MB, `libleanshared.dylib` at 165 MB). That history was never pushed and never
will be. `git filter-repo` stripped both path families locally, taking `.git`
from **2.2 GB to 213 MB**; `origin/main` has **zero** key blobs reachable and
zero oversized blobs.

Backup of the pre-rewrite repository:
`/Volumes/corpdrive/substrate-git-backup-20260824-190835.tar` (2.18 GiB,
verified). Ref mapping: `receipts/PRE_REWRITE_REFS.txt` and
`.git/filter-repo/commit-map` (867 mappings).

**Note:** `ODYSSEY_READY_BASELINE_V1.json` binds `git_head: 4f662307`, which no
longer exists after the rewrite. The record's *content* digests all still
re-resolve; only the commit reference is stale. Regenerate it early in the next
session.

## 4. Current git state

- Index writes **work**. Proven by a full add → in-index → reset → removed round trip
  with no lock left behind.
- A **stale `.git/index.lock` dated Aug 4, zero bytes, no owning process** was blocking
  every index-writing operation. Removed this session. Read-only git worked throughout,
  which is why nothing surfaced it. `pyproject.toml` (Aug 4 06:33) and `uv.lock`
  (Aug 4 07:07) also carry uncommitted edits from that same day — consistent with an
  operation interrupted mid-flight. **Not claimed as proven causation** for the
  HEAD/working-tree divergence, but it is the same date.
- Working tree: ~24,832 tracked-dirty, ~31,285 untracked. Every family is classified —
  see §12 and `receipts/G003_DELETION_DISPOSITION.md`.

## 5. Development environment

Works. Verified end to end this session.

```
.venv          Python 3.12.13        (canonical pin: Makefile:7, portability.py:454, CI)
pytest         9.1.1
imports OK     substrate numpy cryptography pytest docx openpyxl pypdf sympy
lean           4.33.0-rc1, commit 62eed1db  (restored from the repo-tracked toolchain)
also present   z3 4.16.0 · ffmpeg/ffprobe 8.1.2 · blender 4.2.1 (app bundle) · ollama
               · aria2c · git-lfs · colmap · docker
```

Rebuild command if `.venv` is ever destroyed — this should be boring:

```
rm -rf .venv && uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
uv pip install --python .venv/bin/python python-docx openpyxl pypdf sympy
```

Known residual: `uv pip install` does **not** consume `uv.lock`, so numpy resolves to
2.5.2 where the lock says 2.5.1. Switching to `uv sync` is unfinished work.

**One command to establish reality:**

```
.venv/bin/python tools/substrate_doctor.py
```

33 checks across git, environment, tools, storage and ancestor. Reports facts, never
mutates, never deletes. Currently 33 ok / 0 warn / 0 FAIL.

## 6. Test status

`make test` runs unaided. See §10 for the live numbers and the failure taxonomy. The
important rule: **a test red because the frozen ancestor refuses a proposed mutation is
not a defect; a test red because a dependency is missing is.**

## 7. Data / corpus status

96 GB (102,792,352,949 bytes apparent, 214,156 files) under
`data/substrate/tangible_sandbox`:

| | size |
|---|--:|
| `prefetch/odyssey-public-v1` | 90 GB |
| `sources/` (11 pinned git clones) | 3.7 GB |
| `public/` | 1.1 GB |
| `processed/` | 740 MB |
| `archives/` | 322 MB |
| `cache/` | **0 B** |

**Integrity is hash-complete.** CLEVR v1.0 zip (19,021,600,724 B) sha256-verified against
the sealed selection; all **534** MOVi-A objects md5-verified (63,909,466,900 B, zero
mismatches); 16/16 `MANIFEST.sha256` files match.

**What is gone, and why it does not matter:** the R2 acquisition receipt records
86,036,677,480 bytes downloaded, and **17 of 18 archives are absent** (85,698,751,194 B,
99.61%). But **all 21 assets bound by `ODYSSEY_SOURCE_SELECTION.sealed.v2.json` are
present and hash-match**. The loss was entirely outside the sealed set. The recovery list
is empty. Re-acquisition is a WAVE-2 adoption decision, not a repair.

**The corpus is a single copy on one USB drive.** The user accepted that risk
permanently and on the record. Do not re-raise it.

## 8. The ancestor

```
ODYSSEY_READY_BASELINE_V1
  frozen build   59f1fec259da3bc3…
  git head       4f662307
  gates          15/15 pass, including G06-DC
  launch_allowed false
  status         admitted_waiting_for_authority_seal
  activation     false
  drift          ZERO
```

Record: `receipts/ODYSSEY_READY_BASELINE_V1.json` (root `08746c17…`). All 15 gate
evidence digests and all 15 subject digests re-resolve.

**Two proof objects, never merged.** The ancestor has fifteen green gates. Ascension has
**none** until it earns them. Never write "Ascension has 15 green gates".

## 9. Frozen implementation map

`receipts/SUBSTRATE_FROZEN_IMPLEMENTATION_MAP_V1.json` (sha256 `0b8010d8…`).

**17 pinned implementation modules + 11 pinned inputs.** `_validate_frozen_build`
(`odyssey_authority.py:1010-1017`) refuses to seal **any** machine gate when a pinned
module's digest changes. The pinned set includes everything Ascension must change —
`task_bank_generator`, `odyssey_arms`, `odyssey_worker`, `odyssey_authority`.

So the sequence is forced: **edit → refreeze → regenerate all gates.** Not "edit and keep
gates green".

`telegram_notifier` resolves to `tools/odyssey7d_telegram_notifier.py` — the boundary
extends **outside `src/substrate/`**. A scan confined there misses part of it.

Only the FROZEN-PINNED tier is mechanically exact. The other classification tiers are
name-heuristics and **must not** be used as a mutation-safety signal.

## 10. Baseline-health command

```
.venv/bin/python -m pytest tests/substrate/test_baseline_health.py -q
```

Recomputes every pinned digest and **names the drifted module directly**, instead of
surfacing as an unrelated `Refused` buried in a rehearsal test. Carries a self-check
proving the detector detects drift. Currently green.

## 11. Security status

Full detail: `receipts/SUBSTRATE_SECRET_HISTORY_REPORT.md`.

- **16 unencrypted Ed25519 private keys are still in the index at HEAD.**
- **Never pushed.** `git branch -r --contains 6dc1e066` is empty; `main` carries zero key
  paths. The exposure has not left this machine.
- Generation-2 live keys **destroyed** this session (fingerprints recorded). Ignore rules
  and a content-based regression test added.
- **No history rewrite performed** — deliberately. It removes nothing an attacker can
  reach, and would risk the ancestor's provenance. Plan for later:
  `receipts/SECURITY_HISTORY_REWRITE_PLAN.md`.

> **DO NOT PUSH** `codex/odyssey-preflight-repair` or the three `grok/odyssey-*` tips
> until the index is cleared. A push puts 16 signing keys on the public origin; a PR to
> `main` puts them on CI at `fetch-depth: 0`.

## 12. Unresolved defects

| # | item | severity | why it remains | blocks next Ultragoal? | owner |
|--:|---|---|---|---|---|
| 1 | 16 private keys in the index | **high** | index/commit is an operator action | **No** — but blocks any push | operator |
| 2 | 24,446 tracked deletions unreconciled | med | same commit pass as #1 | No | operator |
| 3 | 473 of 579 launch-critical objects untracked | med | committing is an operator action; **mitigated** by a verified 188 KB bundle | No | operator |
| 4 | `activation: True` literal at `odyssey_detachment.py:541` | low | frozen-pinned; editing it breaks all gate sealing | No | Ascension refreeze |
| 5 | Ollama binary digest drift vs sealed manifest | low | user steered toward a runtime change instead | No | Ascension (G089) |
| 6 | `uv pip install` ignores `uv.lock` | low | needs `uv sync` migration | No | Ascension |
| 7 | Odyssey cannot discriminate | **critical (science)** | requires the new subject | **It is the mission** | next Ultragoal |

Items 1–3 are **one operator command**, and it is the same command:

```
git rm -r --cached --quiet artifacts/substrate/odyssey7d/tool-cache
git rm -r --cached --quiet artifacts/substrate/odyssey7d/tool-work
```

## 13. Completed steers

`~/.claude/ultragoal/odyssey-ascension/STEERS.md` — verbatim, append-only.

- **S001** frozen build boundary / reseal policy
- **S002** maximal pre-Odyssey expansion / new developmental subject policy
- **S003** this closeout

## 14. Durable Ultragoal state

```
~/.claude/ultragoal/odyssey-ascension/
  GOAL.md                  120 obligations, 17 verified
  STEERS.md                S001–S003 verbatim
  ultragoal-directive.md   the original directive, verbatim
  receipts/                every receipt, the ancestor bundle, the generators
```

The governor for this session is disarmed at closeout. **The next chat writes a new
`/ultragoal`** — G093–G120 and waves 2–16 are handed off, not cancelled.

## 15. Files created this session

**In the repo:** `tools/substrate_doctor.py`,
`tests/substrate/test_baseline_health.py`, `tests/substrate/test_no_private_keys_tracked.py`,
this handoff. **Modified:** `.gitignore`, `src/substrate/odyssey_tools.py`,
`src/substrate/sandbox_campaign.py`, `src/substrate/final_revision_field_campaign.py`
— all three unpinned, all path-portability only.

**In the ledger:** 17 receipts including `SUBSTRATE_CURRENT_REALITY.md`,
`ODYSSEY_READY_BASELINE_V1.json`, `SUBSTRATE_FROZEN_IMPLEMENTATION_MAP_V1.json`,
`SUBSTRATE_CUSTODY_MANIFEST_V1.json`, `ANCESTOR_EVIDENCE_BUNDLE.tar.gz` (188 KB, 473
objects, round-trip verified), `G086_G087_DISCRIMINATION_DESIGN.md`,
`G083_POWER_VIA_DENSITY.md`.

## 16. Things the next agent MUST NOT assume

1. **That a passing receipt means its bytes exist.** Three separate losses this session
   followed that assumption — the R2 archives, `~/.elan`, and the sealed model blobs. The
   migration preserved repo and corpus but not host state.
2. **That the 15 green gates say anything about Ascension.** They are the ancestor's.
3. **That blast radius equals descriptor binding.** It also includes module digest,
   frozen-build, environment, generated-artifact and transitive gate bindings. I made
   exactly this mistake and broke 12 tests with a one-key rename.
4. **That `make test` leaves the tree clean.** `test_odyssey_rehearsal.py` writes
   into `artifacts/substrate/odyssey7d/v1/rehearsal/`, so a full run dirties ~38
   tracked files. `git checkout -- artifacts/substrate/odyssey7d/v1/rehearsal`
   restores them.
5. **That `data/` is on the SSD.** It is a symlink to a removable USB volume that must
   stay mounted.
6. **That a green test suite means the science works.** It does not. The experiment
   cannot currently discriminate — see below.
7. **That untracked means unimportant.** 473 launch-critical objects are untracked.
8. **That `substrate v5 verify` works from a clean clone.** It needs a ~1.1 GiB raw run
   tree that is not committed.

## 16b. The traps that will cost you hours

From an adversarial handoff review, each independently re-verified by me.

**1. `cd data/..` is a *different git repository*.** `data` resolves to
`/Volumes/corpdrive/substrate/data`, so its parent is `/Volumes/corpdrive/substrate` — a
**separate clone** at the same commit with its own `.git`. Editing, `git clean`-ing or
`rm -rf`-ing there writes to the wrong object database or destroys the corpus this tree
depends on. Verified: `git rev-parse --show-toplevel` gives
`/Users/scammermike/Downloads/substrate` here and `/Volumes/corpdrive/substrate` there.

Related: **`du -sh data` reports `0B`** because `du` does not follow the symlink. The
96 GB is invisible from the repo root.

**2. Odyssey is not on the `substrate` CLI.** `cli.py` exposes `v2`–`v5`,
`nous-closure`, `genesis`, `sandbox`, `product`, `verify`, `doctor`, `run` — and **no
`odyssey`**. Odyssey runs as `python -m substrate.odyssey7d` / `odyssey_authority` /
`odyssey_transition`. So `make accept`, `substrate verify`, `substrate doctor` and
`substrate run` all exercise **earlier campaigns**, not Odyssey. `substrate doctor` is v1
synthesis health; the closeout tool is `tools/substrate_doctor.py` and is deliberately
**not** wired into the CLI.

**3. CI does not cover this branch or Odyssey.** `.github/workflows/substrate.yml`
triggers only on `main` and `agent/substrate-event-horizon`, and its job covers v5, Nous
Closure, Final Revision, Genesis I/II and the sandbox — never Odyssey. A green CI badge
says nothing about the work here.

**4. `run/current-transition.json` is stale.** It is an **Aug-3** receipt reading
`state: waiting_for_verified_r2`, `preflight_authorized: false`, with a *different*
`frozen_build_sha256` (`74ab2d2b…`) than the live preflight (`59f1fec2…`). `start-here.md`
points at it as the fastest current-state check. It is not.

**5. Three git checkouts exist**, not one: this tree, the corpdrive clone, and a detached
worktree at `~/Downloads/forge/projects/substrate` (`911a680b`, clean).

## 17. First recommended action

```
cd ~/Downloads/substrate && .venv/bin/python tools/substrate_doctor.py
```

Then read `receipts/SUBSTRATE_CURRENT_REALITY.md`, then
`receipts/G086_G087_DISCRIMINATION_DESIGN.md`.

**Why the second one matters most.** The measured finding that should shape the entire
next campaign:

- n=8 paired blocks gives **power 0.052** at the program's own SESOI of 0.05 — that is the
  Type I rate. Detectable Cohen's *d_z* is 1.156.
- **All eight task generators fail to discriminate.** Verified at source: frontier A's
  answer is a digest of a visible field; B's request literally prints the answer (a
  memoryless organ scored **24/24**); G's target is an independent random vector; H's is
  `rng.choice`.
- One frontier does work — **E** — and shows the shape to copy: memoryless 0.583 against a
  learnable 1.000, because its rule is a deterministic function of a visible field that
  can only be learned across tasks. It saturates in a day, so the real requirement is
  learnable structure deep enough to still pay out on Day 6.
- The power problem has a free fix: the schedule uses **3.15%** of measured organ
  capacity, so more paired histories per frontier raise n without adding days.
  `receipts/G083_POWER_VIA_DENSITY.md` has the arithmetic.

Odyssey has **not** launched, `activation` is **false**, and no Ascension frozen build
exists.
