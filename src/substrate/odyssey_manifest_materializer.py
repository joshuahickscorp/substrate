"""Custodian-facing materializer for source-bound Odyssey frontier manifests.

The module never selects a source, model, or control.  It accepts a sealed
source-selection document and either an operator seed file held outside the
repository or, absent one, a seed derived deterministically from the frozen
build.  It then produces exactly the candidate-visible and evaluator-only
manifests that the eventual custodian can inspect and seal, recording which
seed provenance was used.  It is deliberately incapable of starting a worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from substrate import odyssey_task_bank as task_bank

PROGRAM = "substrate-odyssey-7d-v1"
PLAN = Path("plans/substrate/tangible_next_launch")
FRONTIERS = tuple("ABCDEFGH")
TASKS_PER_FRONTIER = 84 * 4
SOURCE_SELECTION_SCHEMA = "SUBSTRATE_ODYSSEY_SOURCE_SELECTION/v1"
SOURCE_SELECTION_DRAFT_SCHEMA = "SUBSTRATE_ODYSSEY_SOURCE_SELECTION_DRAFT/v1"

# The split seed has two honest provenances and the manifest set records which
# one produced it.  An operator-held file keeps the candidate/evaluator split
# unpredictable to anyone holding only the repository; the derived form is
# reproducible from the frozen build instead, which is the correct trade when
# scoring is done by deterministic verifiers rather than by a withheld answer
# key.  It is never a secret in that mode, and the receipt must not imply it is.
SEED_PROVENANCE = ("operator_supplied", "derived_from_frozen_build")
DERIVED_SEED_DOMAIN = b"substrate-odyssey-7d-v1/derived-custodian-seed/"


class Refused(RuntimeError):
    """A manifest materialization boundary was not satisfied."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _contains_activation(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            (str(key).casefold() in {"activation", "external_activation"} and child is not False) or _contains_activation(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_activation(child) for child in value)
    return False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict) or _contains_activation(value):
        raise Refused(f"{path} is not an inactive JSON object")
    unsigned = dict(value)
    claimed = unsigned.pop("sha256", None)
    if not isinstance(claimed, str) or claimed != digest(unsigned):
        raise Refused(f"{path} has an invalid self-digest")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    if path.exists():
        raise Refused(f"refusing to overwrite existing manifest artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _root_relative(root: Path, raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise Refused(f"{label} must be a non-empty root-relative path")
    path = (root / raw).resolve()
    if not _inside(root, path):
        raise Refused(f"{label} escapes the repository root")
    return path


def _relative(root: Path, path: Path) -> str:
    if not _inside(root, path):
        raise Refused(f"path escapes repository root: {path}")
    return str(path.resolve().relative_to(root.resolve()))


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise Refused(f"{label} must be a SHA-256 hex digest")
    return value


def _safe_role(value: object) -> str:
    if not isinstance(value, str) or not value or any(token in value.casefold() for token in ("answer", "evaluator", "hidden", "score", "solution", "target")):
        raise Refused("source asset role must be a candidate-visible non-answer role")
    return value


def validate_source_selection(root: Path, selection: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Validate exact, locally present source assets for every A--H frontier."""
    required = {"schema", "program", "status", "frontiers", "activation", "sha256"}
    if set(selection) != required:
        raise Refused("source selection has undeclared or missing fields")
    if selection.get("schema") != SOURCE_SELECTION_SCHEMA or selection.get("program") != PROGRAM:
        raise Refused("source selection has the wrong program identity")
    if selection.get("status") != "sealed" or selection.get("activation") is not False:
        raise Refused("source selection must be sealed and inactive")
    rows = selection.get("frontiers")
    if not isinstance(rows, list) or [row.get("id") if isinstance(row, dict) else None for row in rows] != list(FRONTIERS):
        raise Refused("source selection must contain ordered A-H frontier rows")
    result: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "assets"}:
            raise Refused("source selection frontier row has undeclared or missing fields")
        frontier = row["id"]
        assets = row["assets"]
        if not isinstance(assets, list) or not assets:
            raise Refused(f"source selection needs assets for {frontier}")
        normalized: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        seen_roles: set[str] = set()
        for asset in assets:
            if not isinstance(asset, dict) or set(asset) != {"path", "sha256", "role", "rights_reference"}:
                raise Refused("source selection asset has undeclared or missing fields")
            path = _root_relative(root, asset["path"], label="source asset path")
            expected = _sha256(asset["sha256"], label="source asset sha256")
            role = _safe_role(asset["role"])
            rights = _root_relative(root, asset["rights_reference"], label="source asset rights_reference")
            if not path.is_file() or file_digest(path) != expected:
                raise Refused(f"source asset is absent or drifted: {path}")
            if not rights.is_file():
                raise Refused(f"source asset rights reference is absent: {rights}")
            rendered = {"path": _relative(root, path), "sha256": expected, "role": role}
            if rendered["path"] in seen_paths or role in seen_roles:
                raise Refused("source assets must have distinct paths and roles per frontier")
            seen_paths.add(rendered["path"])
            seen_roles.add(role)
            normalized.append(rendered)
        result[frontier] = normalized
    return result


def seal_source_selection(root: Path, *, draft_path: Path, output_path: Path) -> dict[str, Any]:
    """Source-bind an already selected custodian draft without choosing anything.

    This is intentionally a narrow mechanical step.  It requires a complete
    A--H draft supplied by a person, confirms the present source bytes and
    rights-reference paths, and emits a write-once canonical selection.  It
    neither accepts rights, creates a seed, materializes answers, issues a
    gate receipt, nor starts a worker.
    """
    root = root.resolve()
    for path, label in ((draft_path, "source-selection draft"), (output_path, "source-selection output")):
        if not _inside(root, path):
            raise Refused(f"{label} path must stay inside the repository")
    if output_path.exists():
        raise Refused(f"refusing to overwrite existing source-selection artifact: {output_path}")
    try:
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read source-selection draft {draft_path}: {error}") from error
    if not isinstance(draft, dict):
        raise Refused("source-selection draft must be a JSON object")
    required = {"schema", "program", "status", "frontiers", "activation", "external_activation"}
    if set(draft) != required:
        raise Refused("source-selection draft has undeclared or missing fields")
    if draft.get("schema") != SOURCE_SELECTION_DRAFT_SCHEMA or draft.get("program") != PROGRAM:
        raise Refused("source-selection draft has the wrong schema or program")
    if draft.get("status") != "ready_for_custodian_seal":
        raise Refused("source-selection draft is not explicitly ready for custodian seal")
    if draft.get("activation") is not False or draft.get("external_activation") is not False:
        raise Refused("source-selection draft must remain inactive")
    selection = {
        "schema": SOURCE_SELECTION_SCHEMA,
        "program": PROGRAM,
        "status": "sealed",
        "frontiers": draft["frontiers"],
        "activation": False,
    }
    selection["sha256"] = digest(selection)
    # This is the material boundary: every user-selected asset and rights
    # record must already be present and exact.  The call makes no selection
    # and its normalized return is deliberately not persisted as a claim.
    validate_source_selection(root, selection)
    _write_json(output_path, selection)
    return selection


def _source_bundle(frontier: str, assets: list[dict[str, str]]) -> dict[str, Any]:
    """Produce an immutable, candidate-safe per-frontier source binding.

    The selection digest names the exact frontier and the exact candidate-safe
    asset list rather than acting as a loose pointer to a broader selection
    document.  The materializer adds the read-only assertion at this boundary;
    the source-selection document itself never grants write capability.
    """
    rendered = [{**asset, "read_only": True} for asset in assets]
    return {
        "selection_sha256": digest({"frontier": frontier, "assets": rendered}),
        "assets": rendered,
    }


def build_manifest_set(
    root: Path,
    *,
    selection: dict[str, Any],
    seed_bytes: bytes,
    seed_provenance: str,
    candidate_root: Path,
    evaluator_root: Path,
    frozen: dict[str, Any],
    source_commit: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build candidate/evaluator pairs and a source-bound G03 subject in memory."""
    root = root.resolve()
    if not seed_bytes:
        raise Refused("custodian seed file is empty")
    if seed_provenance not in SEED_PROVENANCE:
        raise Refused(f"unknown seed provenance {seed_provenance!r}")
    if not _inside(root, candidate_root) or not _inside(root, evaluator_root):
        raise Refused("manifest roots must stay in the repository")
    if candidate_root == evaluator_root or candidate_root in evaluator_root.parents or evaluator_root in candidate_root.parents:
        raise Refused("candidate and evaluator roots must be disjoint")
    selection_assets = validate_source_selection(root, selection)
    selection_sha256 = _sha256(selection.get("sha256"), label="source selection sha256")
    implementation = frozen.get("implementation_sha256")
    inputs = frozen.get("input_sha256")
    frozen_sha256 = _sha256(frozen.get("sha256"), label="frozen build sha256")
    if not isinstance(implementation, dict) or not isinstance(inputs, dict):
        raise Refused("frozen build lacks required source maps")
    required_inputs = ("frontier_contract", "task_bank", "rendered_build_index")
    if any(not isinstance(inputs.get(name), str) for name in required_inputs):
        raise Refused("frozen build lacks frontier source bindings")
    task_bank_sha256 = _sha256(implementation.get("task_bank_generator"), label="frozen task-bank generator sha256")
    if task_bank_sha256 != file_digest(Path(task_bank.__file__)):
        raise Refused("task-bank generator drifted after the frozen build")
    secret_seed = hashlib.sha256(seed_bytes).hexdigest()
    seed_commitment = task_bank._digest({"seed": secret_seed})
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    for frontier in FRONTIERS:
        bundle = _source_bundle(frontier, selection_assets[frontier])
        # A launch manifest must never seal a placeholder answer, so the
        # evaluator transcript tree is required here even though fixtures
        # may legitimately materialize candidates without it.
        candidate, evaluator = task_bank.materialize(
            seed_commitment,
            secret_seed,
            frontier,
            TASKS_PER_FRONTIER,
            require_evaluator_ground_truth=True,
        )
        candidate["source_bundle"] = bundle
        candidate.pop("sha256")
        candidate["sha256"] = task_bank._digest(candidate)
        task_bank.verify_materialized_candidate(seed_commitment, secret_seed, candidate)
        candidate_path = candidate_root / f"{frontier}.json"
        evaluator_path = evaluator_root / f"{frontier}.json"
        _write_json(candidate_path, candidate)
        _write_json(evaluator_path, evaluator)
        task_ids = [task["task_id"] for task in candidate["tasks"]]
        source_bundle_sha256 = digest(bundle)
        row = {
            "id": frontier,
            "path": _relative(root, candidate_path),
            "file_sha256": file_digest(candidate_path),
            "schema": candidate["schema"],
            "frontier": frontier,
            "seed_commitment": seed_commitment,
            "task_count": len(task_ids),
            "task_ids_sha256": digest({"task_ids": task_ids}),
            "source_bundle_sha256": source_bundle_sha256,
            "source_selection_sha256": bundle["selection_sha256"],
        }
        rows.append(row)
        artifacts[frontier] = {"candidate": candidate_path, "evaluator": evaluator_path}
    checks = {
        "frozen_build_bound": True,
        "source_maps_bound": True,
        "candidate_manifests_structurally_safe": all(
            task_bank.candidate_is_structurally_safe(_read_json(Path(artifacts[item]["candidate"])) ) for item in FRONTIERS
        ),
        "frontier_set_exact": [row["id"] for row in rows] == list(FRONTIERS),
        "scheduled_task_count_exact": all(row["task_count"] == TASKS_PER_FRONTIER for row in rows),
        "source_bundle_bound": True,
    }
    subject = {
        "schema": "SUBSTRATE_ODYSSEY_FRONTIER_MANIFEST_SET/v1",
        "program": PROGRAM,
        "status": "pass",
        "activation": False,
        "external_activation": False,
        "frozen_build_sha256": frozen_sha256,
        "source_commit": source_commit,
        "implementation_sha256": implementation,
        "input_sha256": inputs,
        "task_bank_generator_sha256": task_bank_sha256,
        "frontier_contract_sha256": inputs["frontier_contract"],
        "task_bank_sha256": inputs["task_bank"],
        "rendered_build_index_sha256": inputs["rendered_build_index"],
        "source_selection_sha256": selection_sha256,
        "seed_provenance": seed_provenance,
        "checks": checks,
        "manifests": rows,
        "manifest_count": len(rows),
        "all_pass": all(checks.values()),
    }
    subject["sha256"] = digest(subject)
    return subject, artifacts


def _git_head(root: Path) -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    head = completed.stdout.strip()
    if completed.returncode != 0 or len(head) != 40:
        raise Refused("cannot resolve current git HEAD")
    return head


def materialize(root: Path, *, selection_path: Path, seed_path: Path | None, candidate_root: Path, evaluator_root: Path, output_path: Path) -> dict[str, Any]:
    """Materialize immutable candidate/evaluator manifests from custodian inputs."""
    root = root.resolve()
    if seed_path is not None:
        if _inside(root, seed_path.resolve()):
            raise Refused("custodian seed must remain outside the repository")
        if not seed_path.is_file():
            raise Refused("custodian seed file is absent")
    for path, label in ((selection_path, "source selection"), (candidate_root, "candidate root"), (evaluator_root, "evaluator root"), (output_path, "output")):
        if not _inside(root, path):
            raise Refused(f"{label} path must stay inside the repository")
    if output_path.exists():
        raise Refused(f"refusing to overwrite existing manifest-set receipt: {output_path}")
    frozen = _read_json(root / PLAN / "ODYSSEY_FROZEN_BUILD.json")
    selection = _read_json(selection_path)
    if seed_path is None:
        frozen_sha256 = _sha256(frozen.get("sha256"), label="frozen build sha256")
        seed_bytes = hashlib.sha256(DERIVED_SEED_DOMAIN + frozen_sha256.encode("utf-8")).digest()
        seed_provenance = "derived_from_frozen_build"
    else:
        seed_bytes = seed_path.read_bytes()
        seed_provenance = "operator_supplied"
    subject, _ = build_manifest_set(
        root,
        selection=selection,
        seed_bytes=seed_bytes,
        seed_provenance=seed_provenance,
        candidate_root=candidate_root,
        evaluator_root=evaluator_root,
        frozen=frozen,
        source_commit=_git_head(root),
    )
    _write_json(output_path, subject)
    return subject


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize source-bound, custody-separated Odyssey task manifests")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--seal-source-selection", action="store_true")
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--evaluator-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    try:
        if args.seal_source_selection:
            if args.draft is None:
                parser.error("--seal-source-selection requires --draft")
            if any(value is not None for value in (args.selection, args.seed_file, args.candidate_root, args.evaluator_root)):
                parser.error("--seal-source-selection accepts only --draft and --out")
            result = seal_source_selection(
                root,
                draft_path=args.draft.expanduser().resolve(),
                output_path=args.out.expanduser().resolve(),
            )
        else:
            if any(value is None for value in (args.selection, args.candidate_root, args.evaluator_root)):
                parser.error("materialization requires --selection, --candidate-root, and --evaluator-root")
            result = materialize(
                root,
                selection_path=args.selection.expanduser().resolve(),
                seed_path=args.seed_file.expanduser().resolve() if args.seed_file is not None else None,
                candidate_root=args.candidate_root.expanduser().resolve(),
                evaluator_root=args.evaluator_root.expanduser().resolve(),
                output_path=args.out.expanduser().resolve(),
            )
    except Refused as error:
        print(json.dumps({"refused": str(error), "activation": False}, sort_keys=True))
        raise SystemExit(2) from error
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
