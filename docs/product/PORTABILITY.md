# Substrate / Odyssey portability

This document is the operator procedure for moving a Substrate/Odyssey working tree
onto external storage and bringing it back later. The goal is verification and
re-run without guesswork, including after a path change or a rebuilt machine.

## What to copy

Copy the **entire working tree**, not only the git-tracked files.

Include explicitly:

| Path | Why |
| --- | --- |
| All git-tracked sources (`src/`, `docs/plans/`, `evidence/`, `ops/configs/`, …) | Program code and sealed plans |
| `data/` (gitignored, about 176 GiB) | Odyssey public corpus under `data/substrate/tangible_sandbox/…` |
| `uv.lock` | Locked Python resolution |
| `docs/plans/substrate/tangible_next_launch/SUBSTRATE_PORTABILITY_MANIFEST.json` | Host portability record |

You may omit:

| Path | Why |
| --- | --- |
| `.venv/` | Absolute shebangs break after a path change; recreate with restore |
| `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/` | Disposable |
| Large unrelated scratch under `/tmp` | Not part of the tree |

A pure `git clone` is **not** enough: `data/` never travels with git.

## What does not travel with the folder

These live outside the repository and must be re-established on the destination host.

### External tools

Recorded in the portability manifest (and originally in
`ODYSSEY_TOOL_PANEL_INVENTORY.json`):

- Lean 4 toolchain under `~/.elan` (elan)
- Z3 (`brew` formula / Cellar path)
- ffmpeg (`brew` formula / Cellar path)
- Blender app bundle under `/Applications/Blender.app`
- Ollama app binary (app bundle or PATH symlink)

### Ollama model weights

Pinned models live under `~/.ollama` (tens of gigabytes) and never ride along with
the folder:

- `gpt-oss:20b` (pinned Odyssey model)
- `qwen3:30b` and `deepseek-r1:32b` (frozen canary `candidate_aliases`)
- `nomic-embed-text` (embedding tool)

### Absolute virtualenv paths

`.venv` scripts embed absolute interpreter shebangs. After the folder path changes,
those scripts point at the old location. Do not rely on a copied `.venv`.

## On return: verify first

From the restored folder root:

```bash
# Optional but recommended after a path change
python3 -m substrate.portability restore

# Full host check against the manifest (non-zero exit if anything blocks a run)
python3 -m substrate.portability verify
```

`verify` never repairs silently. Each item is reported as one of:

- `present-and-matching`
- `present-but-drifted`
- `missing`

Every unsatisfied item includes the exact remediation command.

Fast corpus presence check (manifest file only, not every corpus byte):

```bash
python3 -m substrate.portability verify --quick-corpus
```

Use full `verify` (no `--quick-corpus`) before a scientific run so every
`MANIFEST.sha256` entry is checked against on-disk bytes. Manifests are **read**,
never regenerated.

## What restore does

```bash
python3 -m substrate.portability restore
```

Idempotent. Safe without a password:

1. Recreate `.venv` at the **current** path from the locked dependency set (offline
   when the local `uv` cache already holds wheels).
2. `ollama pull` any missing **pinned** model named in the manifest.
3. Verify corpus integrity against existing per-dataset `MANIFEST.sha256` files.

Restore **prints but does not execute**:

- `brew install …`
- elan bootstrap / toolchain install
- Blender or Ollama app installation
- anything that needs sudo

## What a human must reinstall on a rebuilt machine

Run `python -m substrate.portability verify` and execute every printed remediation
that restore refused. Typical set:

1. Install **uv** and a Python 3.12 toolchain if missing.
2. Install **elan** and Lean `v4.33.0-rc1` (see manifest reinstall command).
3. `brew install z3 ffmpeg` (or equivalent on the host package manager).
4. Install **Blender 4.2.1 LTS** to `/Applications/Blender.app` (macOS).
5. Install **Ollama**, then allow restore (or manual `ollama pull`) for the four
   pinned models.
6. Ensure the full `data/` tree was copied; if not, re-copy from the sealed host
   and re-run verify. Do not invent new `MANIFEST.sha256` files.

## Regenerating the portability manifest

Only on a known-good host after tool or corpus changes:

```bash
python3 -m substrate.portability generate
```

Every digest in `SUBSTRATE_PORTABILITY_MANIFEST.json` is measured from real bytes.
Do not hand-edit digests.

## Related records

| Record | Role |
| --- | --- |
| `docs/plans/.../SUBSTRATE_PORTABILITY_MANIFEST.json` | Portable host/tool/corpus/model inventory |
| `docs/plans/.../ODYSSEY_TOOL_PANEL_INVENTORY.json` | Original tool panel measurement |
| `docs/plans/.../ODYSSEY_FROZEN_BUILD.json` | Frozen build digest |
| `docs/plans/.../ODYSSEY_SOURCE_SELECTION.sealed.v2.json` | Sealed source selection digest |
| `data/.../<dataset>/MANIFEST.sha256` | Per-dataset corpus integrity |

Activation remains off. This procedure does not launch Odyssey or alter sealed
gates.
