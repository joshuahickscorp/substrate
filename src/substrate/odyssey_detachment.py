"""Non-activating integrity checks for the three Odyssey LaunchAgent jobs.

This module deliberately has no ``launchctl`` calls and never writes, copies,
or installs a plist in ``~/Library/LaunchAgents``.  Before authority exists it
may create one private *workspace-only* staging copy of the exact supervisor
plist, bound to the current frozen source map.  That file is not an installed
job, does not grant authority, and cannot be used as a receipt.  Once a real
authority and all three safe installed jobs exist, the only other optional
write is a private, write-once receipt.  Neither path can manufacture a
passing receipt before a real sealed authority exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import stat
import tempfile
from pathlib import Path
from typing import Any

from substrate import odyssey_authority, odyssey_transition

PROGRAM = "substrate-odyssey-7d-v1"
AUTHORITY_SCHEMA = "SUBSTRATE_ODYSSEY_7D_AUTHORITY/v1"
FROZEN_SCHEMA = "SUBSTRATE_ODYSSEY_FROZEN_BUILD/v1"
RECEIPT_SCHEMA = "SUBSTRATE_ODYSSEY_DETACHMENT_CONFIG_RECEIPT/v1"
STAGE_SCHEMA = "SUBSTRATE_ODYSSEY_SUPERVISOR_STAGE/v1"
HANDOFF_SCHEMA = "SUBSTRATE_ODYSSEY_DETACHMENT_HANDOFF/v1"
PLAN = Path("docs/plans/substrate/tangible_next_launch")
RUN_ROOT = Path("runs/substrate/odyssey7d/v1")
STAGING_ROOT = Path("runs/substrate/odyssey_transition/detachment-staging")
AUTHORITY_NAME = "ODYSSEY_7D.authority.json"
FROZEN_NAME = "ODYSSEY_FROZEN_BUILD.json"
RECEIPT_NAME = "DETACHMENT_CONFIG_RECEIPT.json"
MAX_DOCUMENT_BYTES = 256 * 1024

SUPERVISOR_LABEL = "org.substrate.odyssey7d.v1"
RUN_NOTIFIER_LABEL = "org.substrate.odyssey7d.telegram"
PREFLIGHT_NOTIFIER_LABEL = "org.substrate.odyssey-preflight.telegram"
CAFFEINATE_EXECUTABLE = "/usr/bin/caffeinate"
CAFFEINATE_FLAGS = ("-i", "-s")
POWER_ASSERTION_ENV = "SUBSTRATE_ODYSSEY_POWER_ASSERTION"
POWER_ASSERTION_VALUE = "caffeinate-current-user-v1"


class Refused(RuntimeError):
    """Raised when a non-activating detachment integrity check fails."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _contains_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        folded = value.casefold()
        return (
            not value.strip()
            or "replace" in folded
            or "placeholder" in folded
            or folded in {"todo", "tbd", "unknown", "pending"}
            or "${" in value
            or "$(" in value
        )
    if isinstance(value, dict):
        return any(_contains_placeholder(key) or _contains_placeholder(child) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_placeholder(child) for child in value)
    return False


def _root(root: Path) -> Path:
    return root.expanduser().resolve()


def authority_path(root: Path) -> Path:
    return _root(root) / PLAN / AUTHORITY_NAME


def frozen_path(root: Path) -> Path:
    return _root(root) / PLAN / FROZEN_NAME


def receipt_path(root: Path) -> Path:
    return _root(root) / RUN_ROOT / RECEIPT_NAME


def staging_root(root: Path) -> Path:
    """Return the workspace-only location for inert supervisor staging.

    It deliberately sits under the R2-to-Odyssey transition run directory,
    not under the current user's LaunchAgents directory and not inside the
    eventual Odyssey worker run root.  No launcher observes this path.
    """
    return _root(root) / STAGING_ROOT


def staged_supervisor_paths(root: Path, frozen_sha256: str) -> tuple[Path, Path]:
    """Return versioned plist and manifest paths for one frozen build."""
    if not _is_sha256(frozen_sha256):
        raise Refused("staged supervisor requires a valid frozen-build SHA-256")
    directory = staging_root(root)
    return (
        directory / f"{frozen_sha256}.supervisor.plist",
        directory / f"{frozen_sha256}.stage.json",
    )


def launch_agents_root(launch_agents_dir: Path | None = None) -> Path:
    """Return the only LaunchAgents directory accepted by this verifier."""
    raw = Path.home() / "Library/LaunchAgents" if launch_agents_dir is None else launch_agents_dir
    if raw.is_symlink():
        raise Refused("LaunchAgents directory must not be a symlink")
    return raw.expanduser().resolve()


def plist_paths(root: Path, *, launch_agents_dir: Path | None = None) -> dict[str, Path]:
    del root  # The installed paths intentionally stay outside the workspace.
    directory = launch_agents_root(launch_agents_dir)
    return {
        "supervisor": directory / f"{SUPERVISOR_LABEL}.plist",
        "run_notifier": directory / f"{RUN_NOTIFIER_LABEL}.plist",
        "preflight_notifier": directory / f"{PREFLIGHT_NOTIFIER_LABEL}.plist",
    }


def supervisor_program_arguments(root: Path, authority: str | Path) -> list[str]:
    """Return the exact user-session power-protected supervisor invocation.

    ``caffeinate`` is the outer process rather than a detached sidecar: its
    current-user ``-i -s`` assertion lasts exactly until the Python supervisor
    exits, including every child worker restart.  A LaunchAgent already runs
    as the logged-in user, so this introduces neither a privileged helper nor
    a stored credential.
    """
    workspace = _root(root)
    python = workspace / ".venv/bin/python"
    return [
        CAFFEINATE_EXECUTABLE,
        *CAFFEINATE_FLAGS,
        str(python),
        "-m",
        "substrate.odyssey7d",
        "supervise",
        "--root",
        str(workspace),
        "--authority",
        str(authority),
    ]


def power_resilience_contract() -> dict[str, Any]:
    """Describe the fail-closed current-user sleep-prevention contract.

    This is documentation carried by the staged template; the executable
    enforcement lives in the exact plist shape and the supervisor's inherited
    contract marker.  It is intentionally not a launchd key.
    """
    return {
        "mode": "current_user_caffeinate_child",
        "executable": CAFFEINATE_EXECUTABLE,
        "flags": list(CAFFEINATE_FLAGS),
        "scope": (
            "The LaunchAgent current user owns caffeinate and the supervisor; "
            "caffeinate execs the supervisor and its assertion ends when that supervisor exits."
        ),
        "runtime_marker": {"environment": POWER_ASSERTION_ENV, "value": POWER_ASSERTION_VALUE},
    }


def expected_supervisor_plist(root: Path) -> dict[str, Any]:
    """Return the one safe, non-restarting supervisor LaunchAgent shape."""
    workspace = _root(root)
    authority = authority_path(workspace)
    run_root = workspace / RUN_ROOT
    return {
        "Label": SUPERVISOR_LABEL,
        "ProgramArguments": supervisor_program_arguments(workspace, authority),
        "WorkingDirectory": str(workspace),
        "EnvironmentVariables": {
            "SUBSTRATE_ODYSSEY_SUPERVISOR": "launchd",
            POWER_ASSERTION_ENV: POWER_ASSERTION_VALUE,
        },
        "KeepAlive": False,
        "RunAtLoad": False,
        "ProcessType": "Adaptive",
        "ThrottleInterval": 60,
        "AbandonProcessGroup": False,
        "Umask": 0o077,
        "StandardOutPath": str(run_root / "launchd.stdout.log"),
        "StandardErrorPath": str(run_root / "launchd.stderr.log"),
    }


def expected_run_notifier_plist(root: Path) -> dict[str, Any]:
    """Return the exact live-status notifier shape; no credentials live here."""
    workspace = _root(root)
    python = workspace / ".venv/bin/python"
    notifier = workspace / "ops/tools/odyssey7d_telegram_notifier.py"
    logs = workspace / "runs/substrate/odyssey7d"
    return {
        "Label": RUN_NOTIFIER_LABEL,
        "ProgramArguments": [str(python), str(notifier), "tick", "--deliver"],
        "WorkingDirectory": str(workspace),
        "StartInterval": 120,
        "RunAtLoad": False,
        "ProcessType": "Background",
        "Umask": 0o077,
        "StandardOutPath": str(logs / "telegram.stdout.log"),
        "StandardErrorPath": str(logs / "telegram.stderr.log"),
    }


def expected_preflight_notifier_plist(root: Path) -> dict[str, Any]:
    """Return the exact preflight notifier shape; it has no secret environment."""
    workspace = _root(root)
    python = workspace / ".venv/bin/python"
    notifier = workspace / "ops/tools/odyssey7d_telegram_notifier.py"
    logs = workspace / "runs/substrate/odyssey_transition"
    return {
        "Label": PREFLIGHT_NOTIFIER_LABEL,
        "ProgramArguments": [str(python), str(notifier), "tick", "--phase", "preflight", "--deliver"],
        "WorkingDirectory": str(workspace),
        "StartInterval": 120,
        "RunAtLoad": True,
        "ProcessType": "Background",
        "ThrottleInterval": 60,
        "Umask": 0o077,
        "StandardOutPath": str(logs / "preflight-telegram.stdout.log"),
        "StandardErrorPath": str(logs / "preflight-telegram.stderr.log"),
    }


def expected_plists(root: Path) -> dict[str, dict[str, Any]]:
    """Expose all expected shapes under stable receipt keys."""
    return {
        "supervisor": expected_supervisor_plist(root),
        "run_notifier": expected_run_notifier_plist(root),
        "preflight_notifier": expected_preflight_notifier_plist(root),
    }


def _inside(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _private_staging_directory(root: Path) -> Path:
    """Create the one private workspace-only staging directory if necessary."""
    workspace = _root(root)
    # Refuse symlinked ancestors before ``mkdir`` so a hostile or accidental
    # link cannot redirect even the staging write outside the workspace.
    directory = workspace
    for component in STAGING_ROOT.parts:
        candidate = directory / component
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            try:
                candidate.mkdir(mode=0o700)
                metadata = candidate.lstat()
            except OSError as error:
                raise Refused("cannot create Odyssey supervisor staging directory") from error
        except OSError as error:
            raise Refused("Odyssey supervisor staging directory is unreadable") from error
        if candidate.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise Refused("Odyssey supervisor staging directory must be a real directory")
        directory = candidate
    try:
        metadata = directory.lstat()
    except OSError as error:
        raise Refused("Odyssey supervisor staging directory is unreadable") from error
    if directory.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise Refused("Odyssey supervisor staging directory must be a real directory")
    if metadata.st_uid != os.geteuid():
        raise Refused("Odyssey supervisor staging directory must be owned by the current user")
    if not _inside(workspace, directory.resolve()):
        raise Refused("Odyssey supervisor staging directory must stay inside the workspace")
    try:
        os.chmod(directory, 0o700)
    except OSError as error:
        raise Refused("cannot make Odyssey supervisor staging directory private") from error
    return directory


def _plist_bytes(value: dict[str, Any]) -> bytes:
    return plistlib.dumps(value, fmt=plistlib.FMT_XML, sort_keys=True)


def _publish_private_new(path: Path, payload: bytes, *, label: str) -> None:
    """Publish one private immutable staging member without following a link.

    A hard-link publication makes concurrent staging attempts converge only if
    their bytes are identical.  This is deliberately the same no-overwrite
    posture as the final detachment receipt, while remaining entirely inside
    the workspace.
    """
    if path.exists() or path.is_symlink():
        try:
            metadata = path.lstat()
        except OSError as error:
            raise Refused(f"{label} is unreadable") from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise Refused(f"{label} must be a current-user regular file with mode 0600")
        try:
            observed = path.read_bytes()
        except OSError as error:
            raise Refused(f"{label} is unreadable") from error
        if observed != payload:
            raise Refused(f"refusing to overwrite a drifted {label}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError:
            # A racing peer may have published the exact same stage.  Recurse
            # through the read-only branch so a different stage cannot win.
            _publish_private_new(path, payload, label=label)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _frozen_staging_context(root: Path) -> tuple[dict[str, Any], str, str, str]:
    """Return a current frozen map plus exact stage-relevant source hashes.

    This does not inspect or require an authority.  It lets a preparatory
    stage exist before human/custodian gates are complete, but fails closed if
    the freeze has drifted or omitted either source that controls its shape.
    """
    workspace = _root(root)
    try:
        frozen = odyssey_authority.validate_current_frozen_build(workspace)
    except odyssey_authority.Refused as error:
        raise Refused("current Odyssey frozen build failed staging validation") from error
    frozen_sha256 = _self_digest(frozen, label="current Odyssey frozen build")
    implementation = frozen.get("implementation_sha256")
    if not isinstance(implementation, dict):
        raise Refused("current Odyssey frozen build has no implementation map")
    detachment_source = odyssey_transition.canonical_source_digest(Path(__file__))
    supervisor_path = workspace / "src/substrate/odyssey7d.py"
    if not supervisor_path.is_file():
        raise Refused("Odyssey supervisor source is missing")
    supervisor_source = odyssey_transition.canonical_source_digest(supervisor_path)
    if (
        implementation.get("odyssey_detachment") != detachment_source
        or implementation.get("frontier_renderer") != supervisor_source
    ):
        raise Refused("current Odyssey frozen build does not bind the staged supervisor sources")
    return frozen, frozen_sha256, detachment_source, supervisor_source


def construct_staged_supervisor(root: Path) -> dict[str, Any]:
    """Construct an exact inert supervisor stage without creating any files.

    The object is intentionally neither an authority nor a launch receipt.  It
    merely specifies the private workspace artifact that may be staged before
    an authority exists.  Its expected plist is already non-restarting and
    points at the canonical future authority path rather than a placeholder.
    """
    workspace = _root(root)
    _frozen, frozen_sha256, detachment_source, supervisor_source = _frozen_staging_context(workspace)
    plist_path, manifest_path = staged_supervisor_paths(workspace, frozen_sha256)
    expected = expected_supervisor_plist(workspace)
    payload = _plist_bytes(expected)
    target = plist_paths(workspace)["supervisor"]
    if _inside(plist_paths(workspace)["supervisor"].parent, plist_path.resolve()):
        raise Refused("staged supervisor must not reside in LaunchAgents")
    document = {
        "schema": STAGE_SCHEMA,
        "program": PROGRAM,
        "status": "inert_staged",
        "activation": False,
        "external_activation": False,
        "frozen_build": {
            "path": str(frozen_path(workspace)),
            "sha256": frozen_sha256,
        },
        "implementation": {
            "odyssey_detachment_sha256": detachment_source,
            "odyssey7d_sha256": supervisor_source,
        },
        "staged_plist": {
            "path": str(plist_path),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "shape_sha256": _digest({"plist": expected}),
            "label": SUPERVISOR_LABEL,
        },
        "stage_manifest_path": str(manifest_path),
        "installation_target": {
            "path": str(target),
            "label": SUPERVISOR_LABEL,
        },
        "stage_contract": {
            "workspace_only": True,
            "requires_sealed_authority_before_install_or_activation": True,
            "command_never_calls_launchctl": True,
            "command_never_writes_launchagents": True,
            "command_never_writes_detachment_receipt": True,
            "run_at_load_is_false": expected.get("RunAtLoad") is False,
            "keep_alive_is_false": expected.get("KeepAlive") is False,
        },
    }
    document["sha256"] = _digest(document)
    return document


def stage_supervisor(root: Path) -> dict[str, Any]:
    """Write an immutable workspace-only supervisor stage, never an installed job."""
    workspace = _root(root)
    document = construct_staged_supervisor(workspace)
    directory = _private_staging_directory(workspace)
    plist_path = Path(document["staged_plist"]["path"])
    manifest_path = Path(document["stage_manifest_path"])
    if plist_path.parent != directory or manifest_path.parent != directory:
        raise Refused("staged supervisor paths must remain in the private staging directory")
    expected = expected_supervisor_plist(workspace)
    _publish_private_new(plist_path, _plist_bytes(expected), label="staged Odyssey supervisor plist")
    _publish_private_new(
        manifest_path,
        json.dumps(document, sort_keys=True, indent=2).encode("utf-8") + b"\n",
        label="staged Odyssey supervisor manifest",
    )
    return verify_staged_supervisor(workspace)


def verify_staged_supervisor(root: Path) -> dict[str, Any]:
    """Fail closed unless the private workspace stage equals the live freeze."""
    workspace = _root(root)
    expected = construct_staged_supervisor(workspace)
    plist_path = Path(expected["staged_plist"]["path"])
    manifest_path = Path(expected["stage_manifest_path"])
    directory = staging_root(workspace)
    try:
        directory_metadata = directory.lstat()
    except OSError as error:
        raise Refused("Odyssey supervisor staging directory is unreadable") from error
    if directory.is_symlink() or not stat.S_ISDIR(directory_metadata.st_mode) or directory_metadata.st_uid != os.geteuid():
        raise Refused("Odyssey supervisor staging directory is unsafe")
    if stat.S_IMODE(directory_metadata.st_mode) != 0o700:
        raise Refused("Odyssey supervisor staging directory must be mode 0700")
    if not _inside(workspace, directory.resolve()):
        raise Refused("Odyssey supervisor staging directory must stay inside the workspace")
    observed_plist = _read_regular_plist(plist_path, label="staged Odyssey supervisor plist")
    if observed_plist != expected_supervisor_plist(workspace):
        raise Refused("staged Odyssey supervisor plist does not exactly match the frozen safe shape")
    try:
        raw_plist = plist_path.read_bytes()
    except OSError as error:
        raise Refused("staged Odyssey supervisor plist is unreadable") from error
    if hashlib.sha256(raw_plist).hexdigest() != expected["staged_plist"]["file_sha256"]:
        raise Refused("staged Odyssey supervisor plist bytes do not match the frozen stage")
    observed = _read_regular_json(manifest_path, label="staged Odyssey supervisor manifest", private=True)
    _self_digest(observed, label="staged Odyssey supervisor manifest")
    if _contains_placeholder(observed):
        raise Refused("staged Odyssey supervisor manifest contains a placeholder")
    if observed != expected:
        raise Refused("staged Odyssey supervisor manifest does not match the current frozen stage")
    return observed


def prepare_handoff(root: Path) -> dict[str, Any]:
    """Audit the exact no-activation handoff after a real authority is sealed.

    This is intentionally not an installer.  It joins the independently
    verified workspace stage to the sealed authority, then names the remaining
    discrete operations.  The final operation is explicitly external because
    a read-only staging command must not bootstrap or kickstart a user job.
    """
    workspace = _root(root)
    stage = verify_staged_supervisor(workspace)
    authority_sha256, frozen_sha256, _detachment_source = _sealed_authority_context(workspace)
    if stage.get("frozen_build", {}).get("sha256") != frozen_sha256:
        raise Refused("staged supervisor does not bind the sealed authority frozen build")
    handoff = {
        "schema": HANDOFF_SCHEMA,
        "program": PROGRAM,
        "status": "sealed_handoff_ready_no_activation",
        "activation": False,
        "external_activation": False,
        "authority_path": str(authority_path(workspace)),
        "authority_sha256": authority_sha256,
        "frozen_build_sha256": frozen_sha256,
        "stage_manifest": {
            "path": stage["stage_manifest_path"],
            "sha256": stage["sha256"],
        },
        "supervisor": {
            "source_staged_plist": stage["staged_plist"]["path"],
            "installation_target": stage["installation_target"]["path"],
            "label": SUPERVISOR_LABEL,
        },
        "ordered_external_steps": [
            {
                "order": 1,
                "operation": "copy_exact_staged_supervisor_to_current_user_launchagents_with_mode_0600",
                "activation": False,
                "must_revalidate": ["sealed_authority", "workspace_stage"],
            },
            {
                "order": 2,
                "operation": "write_then_verify_detachment_configuration_receipt",
                "activation": False,
                "must_revalidate": ["three_exact_private_plists", "sealed_authority", "current_frozen_build"],
            },
            {
                "order": 3,
                "operation": "bootstrap_and_explicitly_kickstart_one_current_user_supervisor",
                "activation": True,
                "requires": ["verified_detachment_configuration_receipt", "sealed_authority", "all_fifteen_gates_pass"],
            },
        ],
        "forbidden_by_this_command": [
            "LaunchAgents_installation",
            "launchctl_bootstrap",
            "launchctl_kickstart",
            "supervisor_start",
            "worker_start",
            "authority_or_gate_creation",
        ],
    }
    handoff["sha256"] = _digest(handoff)
    return handoff


def _read_regular_json(path: Path, *, label: str, private: bool = False) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_DOCUMENT_BYTES:
            raise Refused(f"{label} is not a bounded regular file")
        if private and stat.S_IMODE(metadata.st_mode) != 0o600:
            raise Refused(f"{label} must be mode 0600")
        if private and metadata.st_uid != os.geteuid():
            raise Refused(f"{label} must be owned by the current user")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise Refused(f"{label} must be a JSON object")
    return value


def _read_regular_plist(path: Path, *, label: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_DOCUMENT_BYTES:
            raise Refused(f"{label} is not a bounded regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise Refused(f"{label} must be mode 0600")
        if metadata.st_uid != os.geteuid():
            raise Refused(f"{label} must be owned by the current user")
        with path.open("rb") as handle:
            value = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException, ValueError) as error:
        raise Refused(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise Refused(f"{label} must be a plist dictionary")
    return value


def _self_digest(value: dict[str, Any], *, label: str) -> str:
    claimed = value.get("sha256")
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    if not _is_sha256(claimed) or claimed != _digest(unsigned):
        raise Refused(f"{label} self-digest is invalid")
    return claimed


def _digest_map(value: Any, *, label: str) -> None:
    if not isinstance(value, dict) or not value or any(not isinstance(key, str) or not _is_sha256(child) for key, child in value.items()):
        raise Refused(f"{label} must be a non-empty SHA-256 map")


def _sealed_authority_context(root: Path) -> tuple[str, str, str]:
    """Validate the sealed authority/frozen binding without requiring plists.

    The stage-to-install handoff needs this check before a supervisor plist is
    copied into LaunchAgents.  Receipt construction still separately requires
    all three exact installed plist files.
    """
    workspace = _root(root)
    authority = _read_regular_json(authority_path(workspace), label="sealed Odyssey authority")
    authority_sha256 = _self_digest(authority, label="sealed Odyssey authority")
    if _contains_placeholder(authority):
        raise Refused("sealed Odyssey authority contains a placeholder")
    program = authority.get("program")
    seal = authority.get("seal")
    worker = authority.get("worker")
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != "sealed_admitted"
        or authority.get("activation") is not False
        or authority.get("external_activation") is not False
        or authority.get("launch_allowed") is not True
        or not isinstance(program, dict)
        or program.get("id") != PROGRAM
        or program.get("launch_allowed") is not True
        or program.get("activation") is not False
        or not isinstance(seal, dict)
        or seal.get("status") != "sealed"
        or not isinstance(worker, dict)
        or worker.get("run_root") != str(RUN_ROOT)
    ):
        raise Refused("authority is not the exact sealed Odyssey detachment authority")
    run_id = authority.get("run_id")
    if not isinstance(run_id, str) or _contains_placeholder(run_id):
        raise Refused("sealed Odyssey authority run id is invalid")
    frozen = _read_regular_json(frozen_path(workspace), label="current Odyssey frozen build")
    frozen_sha256 = _self_digest(frozen, label="current Odyssey frozen build")
    if _contains_placeholder(frozen):
        raise Refused("current Odyssey frozen build contains a placeholder")
    if (
        frozen.get("schema") != FROZEN_SCHEMA
        or frozen.get("activation") is not False
        or frozen.get("scientific_status") != "frozen_waiting_for_verified_r2"
    ):
        raise Refused("current Odyssey frozen build is not an inactive frozen build")
    _digest_map(frozen.get("input_sha256"), label="frozen input source map")
    _digest_map(frozen.get("implementation_sha256"), label="frozen implementation source map")
    if authority.get("frozen_build_sha256") != frozen_sha256 or seal.get("frozen_build_sha256") != frozen_sha256:
        raise Refused("sealed authority does not bind the current frozen build")
    if not _is_sha256(seal.get("protocol_digest")) or not _is_sha256(seal.get("authority_source_sha256")):
        raise Refused("sealed authority has invalid source/protocol bindings")
    source_sha256 = odyssey_transition.canonical_source_digest(Path(__file__))
    if frozen["implementation_sha256"].get("odyssey_detachment") != source_sha256:
        raise Refused("frozen build does not bind the detachment verifier source")
    try:
        validated_frozen = odyssey_authority.validate_current_frozen_build(workspace)
        authority_verification = odyssey_authority.verify(workspace, authority_path(workspace))
    except odyssey_authority.Refused as error:
        raise Refused("sealed authority or frozen build failed its authoritative verification") from error
    if validated_frozen.get("sha256") != frozen_sha256 or authority_verification.get("all_pass") is not True:
        raise Refused("sealed authority or frozen build failed its authoritative verification")
    return authority_sha256, frozen_sha256, source_sha256


def _sealed_context(root: Path) -> tuple[str, str, str]:
    """Compatibility wrapper for receipt construction's authority checks."""
    return _sealed_authority_context(root)


def _validate_installed_plists(root: Path, *, launch_agents_dir: Path | None) -> dict[str, dict[str, str]]:
    workspace = _root(root)
    expected = expected_plists(workspace)
    paths = plist_paths(workspace, launch_agents_dir=launch_agents_dir)
    records: dict[str, dict[str, str]] = {}
    for name in ("supervisor", "run_notifier", "preflight_notifier"):
        path = paths[name]
        observed = _read_regular_plist(path, label=f"{name} LaunchAgent plist")
        if _contains_placeholder(observed):
            raise Refused(f"{name} LaunchAgent plist contains a placeholder")
        if observed != expected[name]:
            raise Refused(f"{name} LaunchAgent plist does not exactly match its safe expected shape")
        records[name] = {
            "label": str(expected[name]["Label"]),
            "path": str(path),
            "plist_sha256": _digest({"plist": expected[name]}),
        }
    return records


def construct_receipt(root: Path, *, launch_agents_dir: Path | None = None) -> dict[str, Any]:
    """Return a self-digested receipt only after all safe config is present.

    The function is deliberately read-only.  In particular, a caller cannot
    use it to turn a template, absent plist, or unsealed authority into a pass.
    """
    workspace = _root(root)
    authority_sha256, frozen_sha256, source_sha256 = _sealed_context(workspace)
    directory = launch_agents_root(launch_agents_dir)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "program": PROGRAM,
        "status": "verified",
        "activation": False,
        "external_activation": False,
        "authority_path": str(authority_path(workspace)),
        "authority_sha256": authority_sha256,
        "frozen_build_path": str(frozen_path(workspace)),
        "frozen_build_sha256": frozen_sha256,
        "detachment_source_sha256": source_sha256,
        "launch_agents_root": str(directory),
        "plists": _validate_installed_plists(workspace, launch_agents_dir=directory),
    }
    receipt["sha256"] = _digest(receipt)
    return receipt


def write_receipt(root: Path, *, launch_agents_dir: Path | None = None) -> dict[str, Any]:
    """Persist one private receipt after the non-activating checks pass.

    This is intentionally a write-once handoff, not an installer: it neither
    creates a LaunchAgent nor invokes ``launchctl``.  A hard-link publish keeps
    an already-present receipt from being overwritten, including during a
    second concurrent invocation.
    """
    workspace = _root(root)
    receipt = construct_receipt(workspace, launch_agents_dir=launch_agents_dir)
    target = receipt_path(workspace)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise Refused("refusing to overwrite an existing Odyssey detachment config receipt")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(receipt, sort_keys=True, indent=2).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise Refused("refusing to overwrite an existing Odyssey detachment config receipt") from error
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt


def verify_receipt(root: Path, *, launch_agents_dir: Path | None = None, path: Path | None = None) -> dict[str, Any]:
    """Fail closed unless a saved receipt is identical to current safe config."""
    workspace = _root(root)
    expected_path = receipt_path(workspace)
    observed_path = expected_path if path is None else path.expanduser().resolve()
    if observed_path != expected_path:
        raise Refused("detachment receipt must use the canonical Odyssey run-root path")
    receipt = _read_regular_json(observed_path, label="Odyssey detachment config receipt", private=True)
    _self_digest(receipt, label="Odyssey detachment config receipt")
    if _contains_placeholder(receipt):
        raise Refused("Odyssey detachment config receipt contains a placeholder")
    expected = construct_receipt(workspace, launch_agents_dir=launch_agents_dir)
    if receipt != expected:
        raise Refused("Odyssey detachment config receipt does not match current authority, frozen build, and plists")
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Expose receipt construction without providing any process activation."""
    parser = argparse.ArgumentParser(description="Verify the non-activating Odyssey detachment configuration")
    parser.add_argument("command", choices=("construct", "write", "verify", "construct-stage", "stage", "verify-stage", "handoff"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--launch-agents-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "construct":
            result = construct_receipt(args.root, launch_agents_dir=args.launch_agents_dir)
        elif args.command == "write":
            result = write_receipt(args.root, launch_agents_dir=args.launch_agents_dir)
        elif args.command == "verify":
            result = verify_receipt(args.root, launch_agents_dir=args.launch_agents_dir)
        elif args.command == "construct-stage":
            result = construct_staged_supervisor(args.root)
        elif args.command == "stage":
            result = stage_supervisor(args.root)
        elif args.command == "verify-stage":
            result = verify_staged_supervisor(args.root)
        else:
            result = prepare_handoff(args.root)
    except Refused as error:
        print(json.dumps({"activation": False, "refused": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
