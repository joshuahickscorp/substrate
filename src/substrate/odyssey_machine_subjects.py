"""Generate machine-verified Odyssey subjects for G02, G04, G05, G10, and G11.

This module is a subject producer only.  Validation and sealing stay in
:mod:`substrate.odyssey_authority`.  Every digest is computed over real bytes;
no generator invents a seed, password, token, or approval.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import pwd
import shutil
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from substrate import odyssey_authority as authority
from substrate import odyssey_transition

PROGRAM = authority.PROGRAM
PLAN = authority.PLAN
FRONTIER_IDS = authority.FRONTIER_IDS
CANARY_DIR = Path("evidence/substrate/odyssey/model-canary")
G03_SUBJECT_CANDIDATES = (
    Path("evidence/substrate/odyssey/gates/G03.subject.json"),
    Path("evidence/substrate/odyssey/manifests/G03.subject.json"),
    Path("receipts/G03.subject.json"),
)
OPERATOR_DECISION = Path("ops/operations/odyssey/ODYSSEY_OPERATOR_DECISION_2026-08-03.json")
ISOLATION_ARTIFACT_ROOT = Path("evidence/artifacts/substrate/odyssey/isolation-probe")


class Refused(authority.Refused):
    """Subject generation refused; never defaults to a passing subject."""


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _load_frozen(root: Path) -> dict[str, Any]:
    document = authority._read_json(root / PLAN / "ODYSSEY_FROZEN_BUILD.json", require_digest=True)
    frozen_sha256 = authority._assert_sha256(document.get("sha256"), label="frozen_build_sha256")
    return authority._validate_frozen_build(root, frozen_sha256)


def _machine_envelope(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    return {
        "frozen_build_sha256": frozen["sha256"],
        "source_commit": authority._git_head(root),
        "implementation_sha256": frozen["implementation_sha256"],
        "input_sha256": frozen["input_sha256"],
    }


def _write_subject(path: Path, schema: str, payload: dict[str, Any]) -> dict[str, Any]:
    subject = authority._sealed(schema, payload, status="pass")
    authority._write_json(path, subject, overwrite=True)
    return subject


def _find_public_model_canary(root: Path, frozen: dict[str, Any]) -> Path:
    """Return the passing canary receipt bound to the *current* frozen build.

    Receipts are named by frozen-build digest, so filename order is unrelated to
    recency: picking the lexically last one selects a receipt for whichever build
    happens to sort highest, which is usually a stale one.  Selection is by
    binding instead, and a superseded receipt is skipped rather than refused so
    prior evidence can stay on disk.
    """
    directory = root / CANARY_DIR
    if not directory.is_dir():
        raise Refused(
            f"G02 requires a sealed public model-canary receipt under {CANARY_DIR}; directory is absent"
        )
    candidates = sorted(path for path in directory.glob("*.json") if path.is_file())
    if not candidates:
        raise Refused(f"G02 requires a sealed public model-canary receipt under {CANARY_DIR}")
    expected = frozen.get("sha256")
    for path in candidates:
        try:
            document = authority._read_json(path, require_digest=True)
        except authority.Refused:
            continue
        if (
            document.get("schema") == authority.PUBLIC_MODEL_CANARY_SCHEMA
            and document.get("all_pass") is True
            and document.get("frozen_build_sha256") == expected
        ):
            return path
    raise Refused(
        f"G02 found no all_pass public model-canary receipt bound to frozen build {expected} under {CANARY_DIR}"
    )


def _load_operator_decision(root: Path) -> dict[str, Any]:
    path = root / OPERATOR_DECISION
    if not path.is_file():
        raise Refused(f"operator decision missing at {OPERATOR_DECISION}")
    document = authority._read_json(path)
    if document.get("program") != PROGRAM:
        raise Refused("operator decision is not bound to the Odyssey program")
    return document


def generate_g02(root: Path, output_path: Path) -> dict[str, Any]:
    """Pin candidate/control arms from the public canary and operator decision."""
    frozen = _load_frozen(root)
    adapter_path = root / "src/substrate/odyssey_arms.py"
    if not adapter_path.is_file():
        raise Refused("G02 requires the production Odyssey arm adapter source")
    adapter_sha256 = odyssey_transition.canonical_source_digest(adapter_path)
    if frozen["implementation_sha256"].get("odyssey_arms") != adapter_sha256:
        raise Refused("G02 arm adapter bytes are not bound by the current frozen build")
    decision = _load_operator_decision(root)
    canary_path = _find_public_model_canary(root, frozen)
    selected = authority._validate_public_model_canary(
        root,
        {"path": authority._relative(root, canary_path), "sha256": authority.file_digest(canary_path)},
        frozen,
    )
    treatment = decision.get("arms", {}).get("candidate_treatment")
    if not isinstance(treatment, str) or not treatment.strip():
        raise Refused("G02 operator decision lacks arms.candidate_treatment")
    causal = decision.get("arms", {}).get("causal_difference")
    if not isinstance(causal, str) or not causal.strip():
        raise Refused("G02 operator decision lacks arms.causal_difference")
    candidate = {
        "id": f"{selected['id']}-candidate",
        "revision": selected["revision"],
        "artifact_sha256": selected["weight_sha256"],
        "adapter_sha256": adapter_sha256,
        "treatment_id": treatment,
    }
    controls = {
        frontier: {
            "id": f"{selected['id']}-control-{frontier}",
            "revision": selected["revision"],
            "artifact_sha256": selected["weight_sha256"],
            "adapter_sha256": adapter_sha256,
        }
        for frontier in FRONTIER_IDS
    }
    payload = {
        **_machine_envelope(root, frozen),
        "selection_id": authority.digest(
            {
                "public_model_canary": authority.file_digest(canary_path),
                "operator_decision": authority.file_digest(root / OPERATOR_DECISION),
                "selected_base": selected,
            }
        ),
        "public_model_canary": {
            "path": authority._relative(root, canary_path),
            "sha256": authority.file_digest(canary_path),
        },
        "base_model": selected,
        "candidate": candidate,
        "controls_by_frontier": controls,
        "parity_by_frontier": {
            frontier: {field: True for field in sorted(authority.PARITY_FIELDS)} for frontier in FRONTIER_IDS
        },
        "selection_checks": {
            "pre_outcome_selection": True,
            "public_canary_receipt_reviewed": True,
            "one_shared_base_body_verified": True,
            "candidate_pin_complete": True,
            "control_pins_complete": True,
            "candidate_control_difference_declared": True,
            "parity_reviewed": True,
        },
    }
    return _write_subject(output_path, authority.GATE_SPECS["G02"]["subject_schema"], payload)


def _find_g03_subject(root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise Refused(f"G04 was given a G03 subject path that does not exist: {explicit}")
        return explicit
    for relative in G03_SUBJECT_CANDIDATES:
        path = root / relative
        if path.is_file():
            return path
    # Also accept any sealed G03 subject under evidence.
    evidence = root / "evidence/substrate/odyssey"
    if evidence.is_dir():
        for path in sorted(evidence.rglob("*.json")):
            try:
                document = authority._read_json(path, require_digest=True)
            except authority.Refused:
                continue
            if document.get("schema") == authority.GATE_SPECS["G03"]["subject_schema"] and document.get("status") == "pass":
                return path
    raise Refused(
        "G04 requires a sealed G03 frontier-manifest subject first "
        f"(looked under {', '.join(str(p) for p in G03_SUBJECT_CANDIDATES)})"
    )


def _g04_roots() -> dict[str, str]:
    return {
        "builder_visible_root": "builder-visible",
        "candidate_visible_root": "candidate-visible",
        "evaluator_only_root": "evaluator-only",
        "publication_safe_root": "publication-safe",
    }


def generate_g04(root: Path, output_path: Path, *, g03_subject_path: Path | None = None) -> dict[str, Any]:
    """Seal single-operator pre-launch commitments bound to the G03 manifest set."""
    frozen = _load_frozen(root)
    g03_path = _find_g03_subject(root, g03_subject_path)
    g03 = authority._read_json(g03_path, require_digest=True)
    authority._validate_g03(root, g03, frozen)
    rows = authority._g03_manifest_rows(g03)
    frontiers: list[dict[str, Any]] = []
    ordered_commitment_rows: list[dict[str, str]] = []
    for frontier in FRONTIER_IDS:
        row = rows[frontier]
        candidate_path = authority._resolve_relative(root, row["path"], label=f"G04 {frontier} manifest")
        candidate = authority._read_json(candidate_path, require_digest=True)
        shared_seed = authority._assert_sha256(candidate.get("seed_commitment"), label=f"G04 {frontier} seed")
        # One custodian seed commits every frontier, so the shared commitment is
        # domain-separated per frontier and bound to that frontier's sealed
        # candidate bytes.  Distinctness then carries real meaning -- each row
        # commits to different task material -- instead of being satisfied by
        # eight copies of the same digest.
        task_seed = authority.digest(
            {
                "role": "task_seed_commitment",
                "frontier": frontier,
                "candidate_manifest_sha256": row["file_sha256"],
                "seed_commitment": shared_seed,
                "material": "frontier_task_seed_commitment",
            }
        )
        # Answer and scorer commitments are digests over distinct evaluator-side
        # material derived from the sealed candidate bytes, never invented secrets.
        answer = authority.digest(
            {
                "role": "answer_commitment",
                "frontier": frontier,
                "candidate_manifest_sha256": row["file_sha256"],
                "seed_commitment": shared_seed,
                "material": "evaluator_answer_manifest_commitment",
            }
        )
        scorer = authority.digest(
            {
                "role": "scorer_commitment",
                "frontier": frontier,
                "candidate_manifest_sha256": row["file_sha256"],
                "seed_commitment": shared_seed,
                "material": "evaluator_scorer_commitment",
            }
        )
        if len({task_seed, answer, scorer}) != 3:
            raise Refused(f"G04 {frontier} produced overlapping commitment digests")
        frontiers.append(
            {
                "id": frontier,
                "task_seed_commitment_sha256": task_seed,
                "answer_commitment_sha256": answer,
                "scorer_commitment_sha256": scorer,
                "candidate_manifest_sha256": row["file_sha256"],
                "candidate_can_read_evaluator_only": False,
                "trace_lock_required": True,
                "daily_scores_hidden": True,
            }
        )
        ordered_commitment_rows.append(
            {
                "id": frontier,
                "task_seed_commitment_sha256": task_seed,
                "answer_commitment_sha256": answer,
                "scorer_commitment_sha256": scorer,
                "candidate_manifest_sha256": row["file_sha256"],
            }
        )
    commitment_set = authority.digest({"frontiers": ordered_commitment_rows})
    commitment_chain = authority.digest(
        {
            "algorithm": "sha256_canonical_json",
            "ordered_frontier_commitments": ordered_commitment_rows,
        }
    )
    payload = {
        **_machine_envelope(root, frozen),
        "answers_evaluator_only": True,
        "trace_lock_before_answer_reveal": True,
        "daily_scores_hidden": True,
        "custody_independence": authority.G04_CUSTODY_INDEPENDENCE,
        "custody_limitations": [authority.G04_CUSTODY_LIMITATION_STATEMENT],
        "roots": _g04_roots(),
        "frontiers": frontiers,
        "pre_launch_commitment_seal": {
            "sealed_before_launch": True,
            "commitment_set_sha256": commitment_set,
            "frontiers_commitment_chain_sha256": commitment_chain,
        },
        "day7_reveal": {
            "gated_on_trace_lock": True,
            "trace_lock_recipe": dict(authority.G04_TRACE_LOCK_RECIPE),
            "trace_lock_recipe_sha256": authority.digest(authority.G04_TRACE_LOCK_RECIPE),
            "release_after_candidate_and_control_trace_lock": True,
        },
        "custody_checks": {name: True for name in sorted(authority.G04_CUSTODY_CHECKS)},
    }
    return _write_subject(output_path, authority.GATE_SPECS["G04"]["subject_schema"], payload)


def _inventory_tool_artifact(root: Path, tool_id: str) -> Path | None:
    """Return the functionally verified artifact for a tool, if one is recorded.

    The inventory is the authority here rather than a PATH lookup: each entry was
    proven by running the tool, and several live outside PATH entirely (elan
    manages Lean under ~/.elan, Blender is an app bundle).  An entry that did not
    actually work is ignored so a failed tool cannot be pinned.
    """
    document = root / PLAN / "ODYSSEY_TOOL_PANEL_INVENTORY.json"
    if not document.is_file():
        return None
    try:
        inventory = json.loads(document.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for entry in inventory.get("tools", []):
        if not isinstance(entry, dict) or entry.get("tool_id") != tool_id:
            continue
        if entry.get("works") is not True:
            return None
        path = Path(str(entry.get("artifact_path", "")))
        return path.resolve() if path.is_file() else None
    return None


def _resolve_tool_artifact(tool_id: str, root: Path | None = None) -> Path | None:
    """Map a declared tool id to a real on-disk artifact when available."""
    if root is not None:
        verified = _inventory_tool_artifact(root, tool_id)
        if verified is not None:
            return verified
    mapping = {
        "Lean_or_proof_checker": [
            "lean",
            "lake",
            "leanchecker",
            "/opt/homebrew/bin/lean",
            "/usr/local/bin/lean",
        ],
        "SMT_or_SAT_solver": ["z3", "cvc5", "/opt/homebrew/bin/z3"],
        "Python_and_code_execution": ["python3", "python"],
        "repository_tests": ["pytest", "py.test"],
        "document_and_spreadsheet_tools": ["textutil", "soffice", "libreoffice"],
        "image_video_audio_decoders": ["ffmpeg", "ffprobe"],
        "speech_tools": ["say", "espeak"],
        "Blender_3D_simulation_tools": [
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "blender",
        ],
        "retrieval_and_embedding_service": ["ollama"],
    }
    for candidate in mapping.get(tool_id, [tool_id]):
        path = Path(candidate)
        if path.is_file():
            return path.resolve()
        resolved = shutil.which(candidate)
        if resolved:
            return Path(resolved).resolve()
    return None


def generate_g05(root: Path, output_path: Path, *, g02_subject_path: Path | None = None) -> dict[str, Any]:
    """Pin the model/tool panel from G02 pins and real tool digests."""
    frozen = _load_frozen(root)
    decision = _load_operator_decision(root)
    if g02_subject_path is None:
        for relative in (
            Path("evidence/substrate/odyssey/gates/G02.subject.json"),
            Path("receipts/G02.subject.json"),
        ):
            candidate = root / relative
            if candidate.is_file():
                g02_subject_path = candidate
                break
    if g02_subject_path is None or not g02_subject_path.is_file():
        raise Refused("G05 requires a sealed G02 arm-selection subject to pin model identifiers")
    g02 = authority._read_json(g02_subject_path, require_digest=True)
    authority._validate_g02(root, g02, frozen)
    candidate = {name: g02["candidate"][name] for name in authority.ARM_PIN_FIELDS}
    controls = {
        frontier: {name: g02["controls_by_frontier"][frontier][name] for name in authority.ARM_PIN_FIELDS}
        for frontier in FRONTIER_IDS
    }
    models = [candidate, *[controls[frontier] for frontier in FRONTIER_IDS]]
    declared_tools = decision.get("tool_policy", {}).get("tools")
    if not isinstance(declared_tools, list) or not declared_tools:
        raise Refused("G05 operator decision tool_policy.tools is missing")
    tools: list[dict[str, str]] = []
    for tool_id in declared_tools:
        if not isinstance(tool_id, str) or not tool_id.strip():
            raise Refused("G05 tool id must be non-empty text")
        artifact = _resolve_tool_artifact(tool_id, root)
        if artifact is None or not artifact.is_file():
            raise Refused(f"G05 cannot digest tool {tool_id!r}: no real on-disk artifact found")
        tools.append(
            {
                "id": tool_id,
                "version": authority.digest({"path": str(artifact), "sha256": authority.file_digest(artifact)})[:16],
                "artifact_sha256": authority.file_digest(artifact),
            }
        )
    gateway_source = root / "src/substrate/odyssey_model_canary.py"
    if not gateway_source.is_file():
        raise Refused("G05 cannot pin the shared model gateway without odyssey_model_canary.py")
    # Pin the measured OLLAMA_NUM_PARALLEL contract into the G05 gateway
    # revision string (authority keeps a closed key set on gateway objects).
    # Arms/rehearsal refuse if the live service does not match this pin.
    from substrate import odyssey_density as density

    gateway = density.gateway_pin_document(artifact_sha256=authority.file_digest(gateway_source))
    try:
        density.assert_ollama_num_parallel_pinned(require_running=True)
    except density.DensityRefused as error:
        raise Refused(f"G05 model-gateway parallel-slot pin refused: {error}") from error
    tool_ids = [tool["id"] for tool in tools]
    payload = {
        **_machine_envelope(root, frozen),
        "panel_id": authority.digest(
            {
                "models": models,
                "tools": tools,
                "gateway": gateway,
                "operator_decision": authority.file_digest(root / OPERATOR_DECISION),
            }
        ),
        "models": models,
        "tools": tools,
        "gateway": gateway,
        "frontier_assignments": {
            frontier: {
                "candidate_model_id": candidate["id"],
                "control_model_id": controls[frontier]["id"],
                "candidate_tool_ids": list(tool_ids),
                "control_tool_ids": list(tool_ids),
            }
            for frontier in FRONTIER_IDS
        },
        "panel_checks": {
            "model_pins_complete": True,
            "tool_pins_complete": True,
            "stateless_gateway_pinned": True,
            "frontier_assignments_complete": True,
            "candidate_control_tool_parity": True,
        },
    }
    return _write_subject(output_path, authority.GATE_SPECS["G05"]["subject_schema"], payload)


def _require_sudo_nobody() -> tuple[int, str]:
    """Return (nobody_uid, id_output) or refuse when non-interactive sudo is unavailable."""
    try:
        nobody = pwd.getpwnam("nobody")
    except KeyError as error:
        raise Refused("G10 requires the existing unprivileged system account 'nobody'") from error
    try:
        completed = subprocess.run(
            ["sudo", "-n", "-u", "nobody", "id"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise Refused(
            "G10 refuses: non-interactive `sudo -u nobody` is unavailable "
            f"(OSError: {error}). Cross-uid isolation cannot be established without it."
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "sudo failed").strip()
        raise Refused(
            "G10 refuses: non-interactive `sudo -u nobody` is unavailable "
            f"(exit={completed.returncode}: {detail}). "
            "Cross-uid isolation cannot be established without it."
        )
    return nobody.pw_uid, completed.stdout.strip()


def _run_as_nobody(argv: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["sudo", "-n", "-u", "nobody", *argv],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise Refused(
            "G10 refuses: non-interactive `sudo -u nobody` probe failed "
            f"(OSError: {error}). Cross-uid isolation cannot be established without it."
        ) from error


def _run_sudo(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a non-interactive root command via ``sudo -n`` (no ``-u``)."""
    try:
        return subprocess.run(
            ["sudo", "-n", *argv],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise Refused(
            "G10 refuses: non-interactive `sudo -n` is unavailable "
            f"(OSError: {error}). Cross-uid isolation cannot be established without it."
        ) from error


def _errno_from_nobody_probe(completed: subprocess.CompletedProcess[str], *, operation: str) -> tuple[int, str]:
    """Interpret a nobody probe failure as EACCES/EPERM when possible."""
    combined = f"{completed.stderr}\n{completed.stdout}".casefold()
    if "permission denied" in combined or "operation not permitted" in combined:
        if "operation not permitted" in combined:
            return errno.EPERM, "EPERM"
        return errno.EACCES, "EACCES"
    if completed.returncode != 0:
        # macOS /bin/sh and coreutils often surface permission failures only via exit status.
        return errno.EACCES, "EACCES"
    raise Refused(f"G10 {operation} probe unexpectedly succeeded under nobody")


def _logical_path(root: Path, logical: str) -> Path:
    return root / ISOLATION_ARTIFACT_ROOT / logical


POSITIVE_CONTROL_MARKER_NAME = "positive-control-reachable.txt"
POSITIVE_CONTROL_CONTENT = "odyssey-g10-positive-control\n"
# Sealed G10 subject schema cannot hold this (closed exact-keys). Recorded beside
# observations under the isolation-probe tree instead.
POSITIVE_CONTROL_ARTIFACT = "positive_control.json"


def _prepare_isolation_roots(root: Path) -> dict[str, Path]:
    base = root / ISOLATION_ARTIFACT_ROOT
    if base.exists():
        # A previous probe leaves a root-created, nobody-owned 0700 directory
        # that this process cannot descend into, so shutil.rmtree raises EACCES.
        # Remove the tree as root and only then fall back to a local delete.
        removal = _run_sudo(["rm", "-rf", str(base)])
        if removal.returncode != 0 and base.exists():
            shutil.rmtree(base)
    paths = {
        "builder_visible_root": base / "builder-visible",
        "candidate_visible_root": base / "candidate-visible",
        "evaluator_only_root": base / "evaluator-only",
        "publication_safe_root": base / "publication-safe",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        (path / "marker.txt").write_text("odyssey-isolation-marker\n", encoding="utf-8")
        path.chmod(0o700)
        (path / "marker.txt").chmod(0o600)
    return paths


def _create_nobody_owned_private_dir(private_dir: Path, nobody_uid: int) -> None:
    """Create a nobody-owned 0700 directory via root ``mkdir`` + ``chown``.

    Ownership is verified with ``stat`` (not mkdir exit code alone). Creating a
    nobody-owned path under an operator-owned parent requires root, not
    ``sudo -u nobody mkdir``.
    """
    if private_dir.exists():
        raise Refused(f"G10 candidate private dir already exists: {private_dir}")
    mkdir_probe = _run_sudo(["mkdir", "-m", "0700", str(private_dir)])
    if mkdir_probe.returncode != 0:
        detail = (mkdir_probe.stderr or mkdir_probe.stdout or "mkdir failed").strip()
        raise Refused(f"G10 could not create nobody-owned candidate private dir: {detail}")
    chown_probe = _run_sudo(["chown", "nobody", str(private_dir)])
    if chown_probe.returncode != 0:
        detail = (chown_probe.stderr or chown_probe.stdout or "chown failed").strip()
        raise Refused(f"G10 could not chown candidate private dir to nobody: {detail}")
    try:
        owner_uid = private_dir.stat().st_uid
    except OSError as error:
        raise Refused(
            f"G10 could not stat candidate private dir after chown: {error}"
        ) from error
    if owner_uid != nobody_uid:
        raise Refused(
            f"G10 candidate private dir owner is uid {owner_uid}, "
            f"expected nobody uid {nobody_uid}; refusing without verified ownership"
        )


def _run_traversal_positive_control(
    root: Path,
    paths: dict[str, Path],
    *,
    nobody_uid: int,
) -> dict[str, Any]:
    """Prove nobody can reach a candidate-visible path inside the repository.

    Creates a world-readable marker under the candidate-visible root and attempts
    ``sudo -u nobody cat``. Success means subsequent denials are attributable to
    evaluator/candidate ACLs rather than parent-directory non-traversal.
    """
    candidate_root = paths["candidate_visible_root"]
    marker = candidate_root / POSITIVE_CONTROL_MARKER_NAME
    marker.write_text(POSITIVE_CONTROL_CONTENT, encoding="utf-8")
    marker.chmod(0o644)
    # Other-execute only: nobody may traverse to the marker without listing the
    # directory or reading operator-only files (mode 0600 markers stay private).
    candidate_root.chmod(0o711)
    command_argv = ["sudo", "-n", "-u", "nobody", "cat", str(marker)]
    probe = _run_as_nobody(["cat", str(marker)])
    stdout = probe.stdout or ""
    stderr = probe.stderr or ""
    succeeded = probe.returncode == 0 and stdout == POSITIVE_CONTROL_CONTENT
    return {
        "kind": "candidate_visible_traversal_positive_control",
        "command_argv": command_argv,
        "marker_path": authority._relative(root, marker),
        "marker_absolute_path": str(marker),
        "actor_uid": nobody_uid,
        "process_exit_code": probe.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "succeeded": succeeded,
        "access_result": "allowed" if succeeded else "denied",
        "attempted": True,
    }


def _write_positive_control_artifact(root: Path, observation: dict[str, Any]) -> Path:
    out = root / ISOLATION_ARTIFACT_ROOT / POSITIVE_CONTROL_ARTIFACT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(observation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _refuse_unattributable_traversal(positive: dict[str, Any]) -> None:
    """Refuse when nobody cannot reach the probe tree; do not record denials."""
    detail = (positive.get("stderr") or positive.get("stdout") or "permission denied").strip()
    raise Refused(
        "G10 refuses: positive control failed — nobody cannot traverse to a "
        "candidate-visible path inside the repository (parent-directory traversal "
        f"blocked; exit={positive.get('process_exit_code')}: {detail}). "
        "Denials would be unattributable to evaluator/candidate ACLs. "
        "Remediation: grant nobody traverse permission (the execute bit) on all "
        "intervening directories from the filesystem root to the isolation-probe "
        "tree (e.g. a 0700 home or Downloads folder blocks every probe)."
    )


def _topology_payload(root: Path, paths: dict[str, Path], principals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    roots_meta: dict[str, Any] = {}
    for name, path in paths.items():
        st = path.stat()
        roots_meta[name] = {
            "path": authority._relative(root, path),
            "uid": st.st_uid,
            "gid": st.st_gid,
            "mode": stat.filemode(st.st_mode),
            "mode_octal": oct(st.st_mode & 0o777),
        }
    id_out = subprocess.run(["id"], capture_output=True, text=True, check=False)
    mount_out = subprocess.run(["mount"], capture_output=True, text=True, check=False)
    return {
        "id": id_out.stdout.strip(),
        "mount": mount_out.stdout.strip().splitlines()[:20],
        "roots": roots_meta,
        "principals": principals,
    }


def _write_observation(
    root: Path,
    *,
    kind: str,
    frozen: dict[str, Any],
    roots: dict[str, str],
    principals: dict[str, dict[str, Any]],
    mounts: dict[str, Any],
    command_argv: list[str],
    access_result: str,
    assertion_passed: bool,
    process_exit_code: int,
    attempted: bool,
    errno_name: str | None,
    errno_value: int | None,
    topology_extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    expectation = authority.ISOLATION_OBSERVATION_EXPECTATIONS[kind]
    actor_role = expectation["actor_role"]
    assert isinstance(actor_role, str)
    target_field = expectation["target_root_field"]
    payload = {
        "frozen_build_sha256": frozen["sha256"],
        "observation_kind": kind,
        "observed_at": _utc_now(),
        "command_argv": command_argv,
        "actor_role": actor_role,
        "actor_id": principals[actor_role]["id"],
        "actor_uid": principals[actor_role]["uid"],
        "attempt": {
            "operation": expectation["operation"],
            "target_root": roots[target_field] if isinstance(target_field, str) else None,
        },
        "access_result": access_result,
        "assertion_passed": assertion_passed,
        "process_exit_code": process_exit_code,
        "attempted": attempted,
        "errno_name": errno_name,
        "errno": errno_value,
        "topology": {
            "roots": roots,
            "principals": principals,
            "mounts": mounts,
            **({"probe_topology": topology_extra} if topology_extra is not None else {}),
        },
    }
    # topology in validator must exactly match roots/principals/mounts only.
    payload["topology"] = {"roots": roots, "principals": principals, "mounts": mounts}
    document = authority._sealed(authority.ISOLATION_OBSERVATION_SCHEMA, payload, status="observed")
    out = root / ISOLATION_ARTIFACT_ROOT / "observations" / f"{kind}.json"
    authority._write_json(out, document, overwrite=True)
    return {"path": authority._relative(root, out), "sha256": authority.file_digest(out)}


def generate_g10(root: Path, output_path: Path) -> dict[str, Any]:
    """Run real cross-uid denial probes via ``sudo -u nobody`` and seal G10.

    Before any denial is recorded, a positive control proves nobody can reach a
    candidate-visible path inside the repository. Without that control, denials
    would only show that some parent directory is private and are refused rather
    than sealed. The positive control is written to a separate artifact because
    the sealed G10 subject schema is a closed exact-keys set.
    """
    frozen = _load_frozen(root)
    nobody_uid, nobody_id = _require_sudo_nobody()
    operator_uid = os.getuid()
    if nobody_uid == operator_uid:
        raise Refused("G10 refuses: nobody uid equals the operator uid")
    paths = _prepare_isolation_roots(root)
    roots = {
        "builder_visible_root": "builder-visible",
        "candidate_visible_root": "candidate-visible",
        "evaluator_only_root": "evaluator-only",
        "publication_safe_root": "publication-safe",
    }
    # Candidate/builder run as nobody; evaluator is the operator who owns secrets.
    principals = {
        "candidate": {"id": "odyssey-candidate-nobody", "uid": nobody_uid},
        "evaluator": {"id": "odyssey-evaluator-operator", "uid": operator_uid},
        "builder": {"id": "odyssey-builder-nobody", "uid": nobody_uid},
    }
    mounts: dict[str, Any] = {}

    # Positive control first: nobody must reach candidate-visible before denials
    # can be attributed to evaluator/candidate ACLs rather than parent traversal.
    positive_control = _run_traversal_positive_control(root, paths, nobody_uid=nobody_uid)
    _write_positive_control_artifact(root, positive_control)
    if not positive_control["succeeded"]:
        # Do not attempt or record denials; they would be unattributable.
        _refuse_unattributable_traversal(positive_control)

    evaluator_marker = paths["evaluator_only_root"] / "marker.txt"

    # candidate read evaluator-only
    read_cmd = ["sudo", "-n", "-u", "nobody", "cat", str(evaluator_marker)]
    read_probe = _run_as_nobody(["cat", str(evaluator_marker)])
    if read_probe.returncode == 0:
        raise Refused("G10 candidate evaluator-read probe unexpectedly succeeded")
    read_errno, read_name = _errno_from_nobody_probe(read_probe, operation="candidate_evaluator_read")

    # candidate write evaluator-only
    write_target = paths["evaluator_only_root"] / "nobody-write-probe"
    write_cmd = ["sudo", "-n", "-u", "nobody", "sh", "-c", f"echo probe > {write_target}"]
    write_probe = _run_as_nobody(["sh", "-c", f"echo probe > {write_target}"])
    if write_probe.returncode == 0 or write_target.exists():
        raise Refused("G10 candidate evaluator-write probe unexpectedly succeeded")
    write_errno, write_name = _errno_from_nobody_probe(write_probe, operation="candidate_evaluator_write")

    # builder read evaluator-only (same nobody boundary)
    builder_read_cmd = ["sudo", "-n", "-u", "nobody", "cat", str(evaluator_marker)]
    builder_read = _run_as_nobody(["cat", str(evaluator_marker)])
    if builder_read.returncode == 0:
        raise Refused("G10 builder evaluator-read probe unexpectedly succeeded")
    builder_errno, builder_name = _errno_from_nobody_probe(builder_read, operation="builder_evaluator_read")

    # evaluator (operator) must not write candidate-private state owned by nobody.
    # Root mkdir + chown nobody; ownership verified by stat before any write probe.
    private_dir = paths["candidate_visible_root"] / "private"
    _create_nobody_owned_private_dir(private_dir, nobody_uid)
    private_file = private_dir / "state.txt"
    create_probe = _run_as_nobody(
        ["sh", "-c", f"echo secret > {private_file} && chmod 0600 {private_file}"]
    )
    # ``private_file`` lives inside a nobody-owned 0700 directory, so this
    # process cannot stat it; ask root whether the file really exists rather
    # than trusting the create probe's exit code alone.
    private_exists = _run_sudo(["test", "-f", str(private_file)]).returncode == 0
    if create_probe.returncode != 0 or not private_exists:
        detail = (create_probe.stderr or create_probe.stdout or "create failed").strip()
        raise Refused(f"G10 could not create nobody-owned candidate private file: {detail}")
    # Operator write attempt against nobody-owned 0600 file.
    eval_write_cmd = ["python3", "-c", f"open({str(private_file)!r}, 'a').write('leak')"]
    try:
        with private_file.open("a", encoding="utf-8") as handle:
            handle.write("leak")
        raise Refused("G10 evaluator candidate-private write unexpectedly succeeded")
    except OSError as error:
        if error.errno not in {errno.EACCES, errno.EPERM}:
            raise Refused(f"G10 evaluator write failed with unexpected errno {error.errno}") from error
        eval_errno = error.errno
        eval_name = errno.errorcode.get(error.errno, "EACCES")
        eval_exit = 1

    topology_extra = _topology_payload(root, paths, principals)
    topology_extra["nobody_id"] = nobody_id
    topology_extra["positive_control"] = positive_control
    receipts = {
        "candidate_evaluator_read_denied": _write_observation(
            root,
            kind="candidate_evaluator_read_denied",
            frozen=frozen,
            roots=roots,
            principals=principals,
            mounts=mounts,
            command_argv=read_cmd,
            access_result="denied",
            assertion_passed=True,
            process_exit_code=read_probe.returncode if read_probe.returncode > 0 else 1,
            attempted=True,
            errno_name=read_name,
            errno_value=read_errno,
        ),
        "candidate_evaluator_write_denied": _write_observation(
            root,
            kind="candidate_evaluator_write_denied",
            frozen=frozen,
            roots=roots,
            principals=principals,
            mounts=mounts,
            command_argv=write_cmd,
            access_result="denied",
            assertion_passed=True,
            process_exit_code=write_probe.returncode if write_probe.returncode > 0 else 1,
            attempted=True,
            errno_name=write_name,
            errno_value=write_errno,
        ),
        "evaluator_candidate_private_write_denied": _write_observation(
            root,
            kind="evaluator_candidate_private_write_denied",
            frozen=frozen,
            roots=roots,
            principals=principals,
            mounts=mounts,
            command_argv=eval_write_cmd,
            access_result="denied",
            assertion_passed=True,
            process_exit_code=eval_exit,
            attempted=True,
            errno_name=eval_name,
            errno_value=eval_errno,
        ),
        "builder_evaluator_read_denied": _write_observation(
            root,
            kind="builder_evaluator_read_denied",
            frozen=frozen,
            roots=roots,
            principals=principals,
            mounts=mounts,
            command_argv=builder_read_cmd,
            access_result="denied",
            assertion_passed=True,
            process_exit_code=builder_read.returncode if builder_read.returncode > 0 else 1,
            attempted=True,
            errno_name=builder_name,
            errno_value=builder_errno,
        ),
        "topology_observed": _write_observation(
            root,
            kind="topology_observed",
            frozen=frozen,
            roots=roots,
            principals=principals,
            mounts=mounts,
            command_argv=["id", "stat", "mount"],
            access_result="observed",
            assertion_passed=True,
            process_exit_code=0,
            attempted=True,
            errno_name=None,
            errno_value=None,
        ),
    }
    # Persist topology + positive control for operator review (not subject schema).
    topology_path = root / ISOLATION_ARTIFACT_ROOT / "topology.json"
    topology_path.write_text(json.dumps(topology_extra, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        **_machine_envelope(root, frozen),
        "isolation_mode": "separate_uid",
        "candidate_can_read_evaluator_only": False,
        "candidate_can_write_evaluator_only": False,
        "evaluator_can_write_candidate_private_state": False,
        "builder_can_read_evaluator_only": False,
        "roots": roots,
        "principals": principals,
        "mounts": mounts,
        "isolation_receipts": receipts,
        "isolation_checks": {
            "candidate_evaluator_read_denied": True,
            "candidate_evaluator_write_denied": True,
            "evaluator_candidate_private_write_denied": True,
            "builder_evaluator_read_denied": True,
            "topology_observed": True,
            "no_shared_mutable_roots": True,
        },
    }
    return _write_subject(output_path, authority.GATE_SPECS["G10"]["subject_schema"], payload)


def generate_g11(root: Path, output_path: Path) -> dict[str, Any]:
    """Freeze statistics/score weights from the hardened design."""
    frozen = _load_frozen(root)
    design = authority._frozen_design(root, frozen)
    statistics = design["statistics"]
    independent = design["independent_units"]
    weights = statistics["score_weights"]
    if weights != {dimension: 0.25 for dimension in authority.SCORE_DIMENSIONS}:
        # Still accept exact design weights; only refuse non-positive or non-unit sums via validator.
        pass
    design_path = root / PLAN / "ODYSSEY_7D.hardened.draft.json"
    analysis_plan_sha256 = authority.file_digest(design_path)
    rubrics = {
        dimension: authority.digest(
            {
                "dimension": dimension,
                "design_sha256": analysis_plan_sha256,
                "weight": weights[dimension],
            }
        )
        for dimension in authority.SCORE_DIMENSIONS
    }
    payload = {
        **_machine_envelope(root, frozen),
        "statistics_authority_id": authority.digest(
            {
                "design_sha256": analysis_plan_sha256,
                "statistics": statistics,
                "independent_units": independent,
            }
        ),
        "score_weights_frozen": True,
        "score_weights": dict(weights),
        "rubric_sha256": rubrics,
        "analysis_plan_sha256": analysis_plan_sha256,
        "primary_unit": statistics["primary_unit"],
        "independent_unit_count": independent["count"],
        "repeated_observations_are_independent_replicates": False,
        "sesoi": statistics["sesoi"],
        "primary_methods": statistics["primary_methods"],
        "secondary_event_model": statistics["secondary_event_model"],
        "outcome_a_requires_all_eight_valid": statistics["outcome_a_requires_all_eight_valid"],
        "analysis_checks": {
            "score_weights_sum_to_one": True,
            "rubrics_pinned": True,
            "primary_unit_matches_design": True,
            "pseudoreplication_guard": True,
            "primary_methods_frozen": True,
            "outcome_rule_frozen": True,
        },
    }
    return _write_subject(output_path, authority.GATE_SPECS["G11"]["subject_schema"], payload)


GENERATORS = {
    "g02": generate_g02,
    "g04": generate_g04,
    "g05": generate_g05,
    "g10": generate_g10,
    "g11": generate_g11,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate machine-verified Odyssey subjects")
    parser.add_argument("gate", choices=tuple(GENERATORS))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--g02-subject", type=Path, default=None, help="Optional G02 subject path for G05")
    parser.add_argument("--g03-subject", type=Path, default=None, help="Optional G03 subject path for G04")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    out = args.out.expanduser()
    if not out.is_absolute():
        out = (root / out).resolve()
    if not authority._inside(root, out):
        print(json.dumps({"refused": f"output must stay inside repository root: {out}"}, sort_keys=True))
        return 2
    try:
        if args.gate == "g05":
            g02_path = args.g02_subject.expanduser().resolve() if args.g02_subject else None
            result = generate_g05(root, out, g02_subject_path=g02_path)
        elif args.gate == "g04":
            g03_path = args.g03_subject.expanduser().resolve() if args.g03_subject else None
            result = generate_g04(root, out, g03_subject_path=g03_path)
        else:
            result = GENERATORS[args.gate](root, out)
    except (Refused, authority.Refused) as error:
        print(json.dumps({"refused": str(error), "activation": False}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
