"""Versioned scientific receipt contracts and append only legacy adapters.

The contract keeps a correction's scientific meaning explicit.  A replacement
or delta is not relabelled as a raw convergence shard merely to satisfy a
validator.  Consumers ask this module for the canonical arm projection instead.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from mop.temporal import io

VERSION = "mop-temporal-receipt-contract/v1"
PROJECTION_VERSION = "mop-temporal-correction-projection/v1"
MIGRATION_VERSION = "mop-temporal-receipt-migration/v1"

BASE_CONVERGENCE = "base_convergence"
EXTENDED_CONVERGENCE = "extended_convergence"
CORRECTION = "correction"
PRINCIPAL = "principal"
VERIFICATION = "verification"
KINDS = {BASE_CONVERGENCE, EXTENDED_CONVERGENCE, CORRECTION, PRINCIPAL, VERIFICATION}


class ContractError(ValueError):
    """A stable validation failure suitable for a deterministic hold."""


def declaration(kind: str, projection: str) -> dict:
    if kind not in KINDS:
        raise ContractError(f"unknown receipt kind: {kind}")
    return {"version": VERSION, "kind": kind, "projection": projection}


def kind_for_stage(stage: str, identity: str) -> str | None:
    if stage == "e2_converge" and identity.startswith("cshard_"):
        return BASE_CONVERGENCE
    if stage == "e2_converge_extended" and identity.startswith("xshard_"):
        return EXTENDED_CONVERGENCE
    if stage in {
        "e2_converge_corrections",
        "e2_optimization_corrections",
        "e2_principal_corrections",
    }:
        return CORRECTION
    if stage == "e2_principal":
        return PRINCIPAL
    return None


def _canonical_result_valid(document: dict, proof: bool = False) -> bool:
    version_key, digest_key = (
        ("sha256_version", "sha256") if proof else ("result_hash_version", "result_sha256")
    )
    return (
        document.get("program") == io.PROGRAM
        and isinstance(document.get("source_commit"), str)
        and len(document["source_commit"]) == 40
        and isinstance(document.get("source_tree_oid"), str)
        and len(document["source_tree_oid"]) == 40
        and document.get(version_key) == "canonical_json_v2"
        and document.get(digest_key)
        == io.sha_obj({k: v for k, v in document.items() if k != digest_key})
    )


def _relative(path: Path) -> str:
    try:
        return path.relative_to(io.ROOT).as_posix()
    except ValueError as exc:
        raise ContractError("receipt path escapes repository root") from exc


def _migration_prefix(path: Path) -> str:
    try:
        stage = path.parent.relative_to(io.RUNS).as_posix().replace("/", "__")
    except ValueError as exc:
        raise ContractError("migration source is outside the receipt root") from exc
    return f"{stage}__{path.stem}__{io.sha_file(path)[:12]}"


def _migration(path: Path, kind: str) -> dict | None:
    root = io.RUNS / "receipt_migrations"
    if not root.is_dir() or not path.is_file():
        return None
    expected_hash = io.sha_file(path)
    candidates = sorted(root.glob(f"{_migration_prefix(path)}*.json"))
    for candidate in candidates:
        try:
            document = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            document.get("schema") == MIGRATION_VERSION
            and document.get("source_path") == _relative(path)
            and document.get("source_sha256") == expected_hash
            and document.get("kind") == kind
            and document.get("normalized_contract", {}).get("version") == VERSION
            and document["normalized_contract"].get("kind") == kind
            and _canonical_result_valid(document)
        ):
            return document
    return None


def effective_declaration(document: dict, kind: str, path: Path | None = None) -> dict:
    contract = document.get("receipt_contract")
    if isinstance(contract, dict):
        if contract.get("version") != VERSION or contract.get("kind") != kind:
            raise ContractError("receipt contract version or kind mismatch")
        return contract
    migration = _migration(path, kind) if path is not None else None
    if migration is None:
        raise ContractError("receipt contract missing and no exact migration sidecar exists")
    return migration["normalized_contract"]


def _mapping_at(mapping: dict, key: int):
    return mapping.get(str(key), mapping.get(key))


def _parent_receipts(projection: dict, cell: str) -> None:
    parents = projection.get("parent_receipts")
    if not isinstance(parents, list) or not parents:
        raise ContractError("correction parent receipts missing")
    matching = [parent for parent in parents if isinstance(parent, dict) and parent.get("cell") == cell]
    if not matching:
        raise ContractError("correction parent factorial cell mismatch")
    for parent in matching:
        if not isinstance(parent, dict):
            raise ContractError("correction parent binding malformed")
        relative = parent.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ContractError("correction parent path is not repository relative")
        path = io.ROOT / relative
        if not path.is_file() or parent.get("sha256") != io.sha_file(path):
            raise ContractError("stale parent receipt")
        try:
            parent_document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError("correction parent receipt is unreadable") from exc
        parent_cells = {parent_document.get("cell")}
        parent_cells.update(
            row.get("cell")
            for row in parent_document.get("runs", [])
            if isinstance(row, dict)
        )
        if cell not in parent_cells:
            raise ContractError("correction parent factorial cell mismatch")


def _arm_grid(document: dict, projection: dict, records: dict) -> None:
    grid = projection.get("grid")
    seeds = projection.get("seeds")
    cell = projection.get("cell")
    bed = document.get("bed")
    if cell != document.get("cell"):
        raise ContractError("mismatched factorial cell")
    if (
        not isinstance(grid, list)
        or not grid
        or len(grid) != len(set(grid))
        or not all(isinstance(value, int) and value > 0 for value in grid)
    ):
        raise ContractError("correction grid malformed")
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(seeds) != len(set(seeds))
        or not all(isinstance(seed, int) for seed in seeds)
    ):
        raise ContractError("correction seed inventory malformed")
    if not isinstance(records, dict) or {int(key) for key in records} != set(grid):
        raise ContractError("correction arm grid missing")
    scores = document.get("seed_scores")
    curve = document.get("curve")
    if not isinstance(scores, dict) or not isinstance(curve, dict):
        raise ContractError("correction score projections missing")
    for budget in grid:
        rows = _mapping_at(records, budget)
        shown = _mapping_at(scores, budget)
        if not isinstance(rows, list) or not isinstance(shown, list):
            raise ContractError("correction arm records malformed")
        row_seeds = [row.get("seed") for row in rows if isinstance(row, dict)]
        if len(rows) != len(seeds) or row_seeds != seeds or len(row_seeds) != len(set(row_seeds)):
            raise ContractError("wrong or duplicate correction arm identity")
        for row, score in zip(rows, shown, strict=True):
            if (
                row.get("bed") != bed
                or row.get("cell") != cell
                or row.get("updates") != budget
                or row.get("score") != score
                or not isinstance(row.get("checkpoint_sha"), str)
                or len(row["checkpoint_sha"]) != 64
            ):
                raise ContractError("correction arm identity or score mismatch")
        if not math.isclose(
            float(_mapping_at(curve, budget)),
            sum(float(value) for value in shown) / len(shown),
            rel_tol=0,
            abs_tol=1e-5,
        ):
            raise ContractError("correction curve does not equal arm projection")


def _correction_projection(document: dict, declaration_: dict) -> None:
    projection_name = declaration_.get("projection")
    if projection_name in {"measured_arm_grid", "legacy_measured_arm_grid"}:
        records = document.get("arm_records")
        if not isinstance(records, dict):
            raise ContractError("legacy correction arm records missing")
        return
    projection = document.get("correction_projection")
    if not isinstance(projection, dict) or projection.get("schema") != PROJECTION_VERSION:
        raise ContractError("canonical correction projection missing")
    mode = projection.get("mode")
    if mode not in {"replacement", "delta", "run_replacement"}:
        raise ContractError("correction projection mode invalid")
    if mode == "run_replacement":
        cells = projection.get("cells")
        if (
            declaration_.get("projection") != "replacement_runs"
            or not isinstance(cells, list)
            or not cells
            or len(cells) != len(set(cells))
            or projection.get("runs") != document.get("runs")
        ):
            raise ContractError("principal correction run projection mismatch")
        for cell in cells:
            _parent_receipts(projection, cell)
        return
    if declaration_.get("projection") != f"{mode}_arm_grid":
        raise ContractError("correction declaration does not match its projection")
    if projection.get("cell") != document.get("cell"):
        raise ContractError("mismatched factorial cell")
    _parent_receipts(projection, projection.get("cell"))
    records = projection.get("arm_records")
    _arm_grid(document, projection, records)
    if mode == "delta":
        for budget in projection["grid"]:
            for row in _mapping_at(records, budget):
                if not math.isclose(
                    float(row.get("corrected_score")),
                    float(row.get("parent_score")) + float(row.get("delta")),
                    rel_tol=0,
                    abs_tol=1e-9,
                ):
                    raise ContractError("incorrect correction delta")


def validate(
    kind: str,
    document: dict,
    *,
    path: Path | None = None,
    verify_seal: bool = True,
) -> dict:
    if kind not in KINDS:
        raise ContractError(f"unknown receipt kind: {kind}")
    declaration_ = effective_declaration(document, kind, path)
    if verify_seal and not _canonical_result_valid(document, proof=kind == VERIFICATION):
        raise ContractError("receipt seal drift")
    projection = declaration_.get("projection")
    if kind in {BASE_CONVERGENCE, EXTENDED_CONVERGENCE}:
        if projection != "raw_arm_grid" or not isinstance(document.get("arm_records"), dict):
            raise ContractError("raw convergence arm grid missing")
    elif kind == CORRECTION:
        _correction_projection(document, declaration_)
    elif kind == PRINCIPAL:
        if projection != "factorial_runs" or not isinstance(document.get("runs"), list):
            raise ContractError("principal factorial runs missing")
    elif kind == VERIFICATION:
        if (
            projection != "independent_role_checks"
            or not isinstance(document.get("role_b"), dict)
            or not isinstance(document.get("role_c"), dict)
        ):
            raise ContractError("verification role checks missing")
    return declaration_


def arm_records(document: dict) -> dict | None:
    projection = document.get("correction_projection")
    if isinstance(projection, dict) and projection.get("mode") in {"replacement", "delta"}:
        return projection.get("arm_records")
    return document.get("arm_records")


def adapt_for_aggregation(document: dict) -> dict:
    """Return a normalized view without mutating the sealed source receipt."""
    records = arm_records(document)
    if not isinstance(records, dict) or document.get("arm_records") is records:
        return document
    normalized = copy.deepcopy(document)
    normalized["arm_records"] = copy.deepcopy(records)
    normalized["normalized_from_correction_projection"] = {
        "contract": VERSION,
        "source_result_sha256": document.get("result_sha256"),
    }
    return normalized


def migrate_path(path: Path, kind: str, projection: str) -> Path:
    """Bind an unchanged old receipt to v1 using an append only sidecar."""
    document = json.loads(path.read_text())
    normalized = declaration(kind, projection)
    if kind in {BASE_CONVERGENCE, EXTENDED_CONVERGENCE} and not isinstance(
        document.get("arm_records"), dict
    ):
        raise ContractError("old convergence receipt cannot migrate without arm records")
    if kind == CORRECTION and projection == "legacy_measured_arm_grid" and not isinstance(
        document.get("arm_records"), dict
    ):
        raise ContractError("old correction receipt cannot migrate without arm records")
    name = f"{_migration_prefix(path)}.json"
    target = io.RUNS / "receipt_migrations" / name
    if target.is_file():
        existing = _migration(path, kind)
        if existing is None:
            raise ContractError("existing receipt migration does not bind exact source bytes")
        return target
    return io.run_json(
        name,
        {
            "schema": MIGRATION_VERSION,
            "source_path": _relative(path),
            "source_sha256": io.sha_file(path),
            "kind": kind,
            "normalized_contract": normalized,
            "normalization": "append only metadata sidecar; source receipt bytes unchanged",
            "activation": False,
        },
        "receipt_migrations",
    )


def migrate_existing() -> list[Path]:
    specs = [
        ("e2_converge", "cshard_*.json", BASE_CONVERGENCE, "raw_arm_grid"),
        ("e2_converge_extended", "xshard_*.json", EXTENDED_CONVERGENCE, "raw_arm_grid"),
        (
            "e2_optimization_corrections",
            "optimization_*.json",
            CORRECTION,
            "legacy_measured_arm_grid",
        ),
    ]
    migrated = []
    for subdir, pattern, kind, projection in specs:
        for path in sorted((io.RUNS / subdir).glob(pattern)):
            if path.is_file() and not isinstance(json.loads(path.read_text()).get("receipt_contract"), dict):
                migrated.append(migrate_path(path, kind, projection))
    return migrated


def main(argv=None) -> None:
    argv = argv or []
    if argv and argv[0] != "migrate-existing":
        raise ValueError(argv)
    migrated = migrate_existing()
    print(f"receipt migrations: {len(migrated)} append only sidecars", flush=True)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
