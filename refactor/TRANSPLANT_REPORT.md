# Substrate 4.0.0a1 transplant report

Date: 2026-09-18

The frozen package is now the canonical active Substrate implementation in this repository. The source package remains explicitly `source-frozen-unexecuted`: this transplant does not claim behavioral qualification, AGI, sentience, or consciousness.

## Active implementation

- Version: `4.0.0a1` (`FREEZE.json`)
- Canonical source: `src/substrate/` — 18 Python modules, 6,005 Python lines
- Active tests: `tests/` — freeze test set, 7 Python files including `conftest.py`
- Active configuration: `pyproject.toml`, `configs/`, `Makefile`, `.github/workflows/static.yml`
- CLI entrypoint: `substrate=substrate.cli:main`
- One Odyssey program: `src/substrate/odyssey.py` and `configs/odyssey.json`
- Hawking boundary: `src/substrate/hawking.py`, `src/substrate/handoff.py`, and `src/substrate/HAWKING_LICENSE.txt`
- Portable/realization separation: NR validation and NX machine binding remain explicit; learned numeric state is not bundled

The active source, tests, configs, compact evidence pack, and freeze metadata were copied from the supplied package. No `llama.cpp`, Ollama, MLX model, server, weights, or other model runtime was imported beyond the freeze's optional local-fitting interface.

Target-specific integration is limited to preserving the repository's historical-store ignore boundaries and teaching the static checker to exclude those stores plus the transplant ledgers from the active-source line ceiling. The canonical `src/substrate` and freeze test/config trees are byte-identical to the supplied package.

## Transplant inventory

The complete pre-transplant path-by-path map is retained in [TRANSPLANT_INVENTORY.json](TRANSPLANT_INVENTORY.json). It contains 474 reviewed old tracked paths:

- 12 `REPLACED` by the freeze's canonical files
- 366 `RETIRED` because the freeze supersedes the old active implementation
- 96 `HISTORICAL_TEST_OBLIGATION` entries retained as migration obligations
- 0 `MERGED` paths and 0 compatibility shims remaining; `handoff.py`'s predecessor migration is quarantine/lineage handling, not a parallel runtime

Before the transplant, the repository contained 192 old `src/substrate` modules (136,733 source lines) and 97 test files. The old active source, native tree, operations tree, local configs/tools, old workflow, and old product/planning surfaces were moved out of the active implementation. A recoverable migration backup is at `/Users/scammermike/Downloads/substrate-pre-freeze-legacy` (about 69.9 MB); it is not on the active Python import path.

Historical product/planning material that remains useful is under `docs/archive/pre-freeze-active-docs/`. The large pre-existing archive, artifacts, runs, and evidence stores were not deleted or rewritten.

There are no active parallel `substrate_v2`, `substrate_v4`, `legacy substrate`, compatibility runtime, or old native runtime directories in the canonical source tree. Any legacy material left in the repository is ignored historical storage or documentation, not an active implementation owner.

## Historical evidence reconciliation

Evidence was preserved and indexed rather than treated as runtime source or qualification:

- Freeze-selected evidence: 114 expected, 114 hash-matched, 0 missing, 0 mismatched.
- Freeze distribution manifest: 119 expected entries, 119 matched, 0 missing, 0 mismatched.
- Full compact-pack source index: 10,081 source entries, 10,081 matched, 0 missing, 0 mismatched, 0 unexpected target files.
- The compact representation uses canonical sorted compact JSON blobs, a source-path index with original size and SHA-256, and `raw-files.tar.zst` for bytes that cannot be canonicalized. It records the source hashes even when JSON whitespace/key order changes.
- Compact-pack manifest: 10,081 files; 7,617 JSON files; 2,464 non-JSON files; 7,613 canonical JSON blobs; 245 duplicate canonical JSON representations; 1,977,498,553 source bytes represented.
- The compact pack itself is 137,442,414 bytes on disk. The original loose `evidence/` store remains at 1,977,498,553 bytes for preservation.

The reconciliation is machine-readable in [TRANSPLANT_EVIDENCE_RECONCILIATION.json](TRANSPLANT_EVIDENCE_RECONCILIATION.json). Relative to the 114 freeze-selected evidence files, the remaining historical corpus is 9,967 additional source entries represented by the compact pack. No expected freeze evidence is missing or unverifiable. The external `data` symlink points to `/Volumes/corpdrive/substrate/data` and was not hash-verified because that volume is outside this checkout. The larger `artifacts/` and `runs/` stores were retained untouched, but no freeze manifest claims them as qualified evidence.

Historical storage left in place includes `evidence/`, `evidence-compact/`, `artifacts/`, `runs/`, `archive/`, `.venv/`, and `logs/`. These are not active Substrate modules.

## Hawking integration and provenance

The freeze prepares a clean adapter boundary without vendoring or inventing a Hawking runtime. The donor is `joshuahickscorp/hawking` at commit `6a86a5cd49fbe9211ec11e133e9d70a4e2044238`; the license is preserved in `src/substrate/HAWKING_LICENSE.txt`.

| Capability | Donor source | Treatment | Current status / future dependency |
|---|---|---|---|
| Portable NR rules and seals | `tools/nr_container.py` — blob `10007c2d679ec560a17d8b1a14321db5fc8d560e` | Adapted pure validation and identity checks | Prepared; future Hawking representation closure still needs an executed provider qualification |
| Machine-specific NX genome binding | `tools/nx_genome.py` — blob `00228167d694b44c0f76e71d36aab451b8e29cd8` | Adapted checks that bind NX to exact NR bytes and machine genome | Prepared; real lowering/kernel/device integration remains future work |
| Gravity/Nova lineage | `tools/future/gravity_nova_lineage.py` — blob `8dffa027ec7db9c005fbf6c0d55952fd5f523c23` | Adapted lineage/rollback admission checks | Prepared; future descendant training and independent qualification remain external |
| Hawking bridge contract | `crates/hawking-adapters/src/bridge_surface.rs` — blob `09c3e0dbc788ca72eefacf15dac35cef907d7647` | Wrapped by the Substrate-side capability-port contract | Prepared; no claim that the current donor implements the proposed full ABI |
| Live HTTP surface declaration | `crates/hawking-serve/src/http.rs` — blob `1a95da64dfb90ce51f44de7f1888f672a03cb031` | Inspected surface/status constants and conservative loopback client | Prepared; unsupported/partial endpoints remain explicit and are never faked as successful |

No Hawking service, Rust closure, Metal/CUDA/FPGA kernel, trained weight, credential, or machine-specific production state was copied. Whole-system future inheritance remains allowed by the contract; Hawking is not required for basic Substrate operation.

## Validation performed

- Static source inspection: `source-inspection-clean`; 26 Python files parsed/compiled without execution; 5,454 AST calls reviewed; 2,243 resolved calls inspected; 0 static errors.
- Imports: all 18 `substrate` modules imported successfully with bytecode disabled.
- CLI surface: `python -m substrate.cli --help` completed successfully.
- Tests run: 0. Test collection: 0.
- Odyssey runs: 0. Model calls: 0. Training, service, container, and expensive runtime execution: 0.

The current machine-readable static result is [STATIC_INSPECTION.json](STATIC_INSPECTION.json). It records static inspection only and does not imply behavioral equivalence or test passage.

## Remaining real gaps

- Actual pinned Hawking ABI/provider mappings and full runtime/compiler/device qualification.
- Real local organ providers, optional neural fitting qualification, and machine-specific NX lowering.
- External multimodal apprenticeship, broad semantic abstraction, deeper recurrence, and longitudinal developmental evidence.
- Executed Odyssey research and any consciousness-relevant evidence; none was run or fabricated here.
- Independent qualification of imported historical artifacts that lack an expected freeze hash.
