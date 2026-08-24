"""Bind a completed R2 continuity receipt to its exact clean source state.

The continuity verifier establishes that a 24-hour lane completed.  It cannot
by itself prove that the lane, current checkout, and clean-clone reproduction
were built from the same source.  This companion verifier supplies that
provenance gate without rewriting any historical R2 evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from substrate import sandbox_campaign

PROGRAM = "substrate-tangible-sandbox-r2"
EVIDENCE = Path("evidence/substrate/tangible_sandbox")


class Refused(RuntimeError):
    """The current checkout cannot be proven to match R2 continuity evidence."""


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


def _contains_true_activation(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            (key.casefold() in {"activation", "external_activation"} and child is not False)
            or _contains_true_activation(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_true_activation(child) for child in value)
    return False


def _read_json(path: Path, *, require_digest: bool = False) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Refused(f"{path} must contain a JSON object")
    if _contains_true_activation(value):
        raise Refused(f"{path} enables activation")
    claimed = value.get("sha256")
    if require_digest and not isinstance(claimed, str):
        raise Refused(f"{path} is missing a self-digest")
    if isinstance(claimed, str):
        unsigned = dict(value)
        unsigned.pop("sha256")
        if digest(unsigned) != claimed:
            raise Refused(f"{path} self-digest mismatch")
    return value


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _relative(root: Path, path: Path) -> str:
    if not _inside(root, path):
        raise Refused(f"path escapes repository root: {path}")
    return str(path.resolve().relative_to(root.resolve()))


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Refused(f"refusing to overwrite {path}")
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    if completed.returncode or not completed.stdout.strip():
        raise Refused(completed.stderr.strip() or "cannot resolve git HEAD")
    return completed.stdout.strip()


def _campaign_source_digest(root: Path) -> str:
    """Call the canonical R2 digest function against the caller's checkout."""
    original = sandbox_campaign.ROOT
    try:
        sandbox_campaign.ROOT = root
        return sandbox_campaign.source_digest()
    finally:
        sandbox_campaign.ROOT = original


def _sealed_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema": "SUBSTRATE_SANDBOX_R2_PROVENANCE_VERIFICATION/v1",
        "program": PROGRAM,
        "scientific_status": "pass",
        "independently_verified": True,
        "verification_scope": (
            "R2 source-provenance binding only; this receipt does not alter the "
            "historical terminal classification or the continuity result."
        ),
        **payload,
        "activation": False,
        "external_activation": False,
        "unqualified_nous": False,
    }
    body["sha256"] = digest(body)
    return body


def _sealed_ref(root: Path, path: Path, document: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": _relative(root, path),
        "file_sha256": file_digest(path),
        "sha256": document["sha256"],
        "schema": document.get("schema"),
        "source_digest": document.get("source_digest"),
    }


def verify(root: Path, output_path: Path) -> dict[str, Any]:
    """Write a receipt only when R2, clean clone, and checkout are identical."""
    result_path = root / EVIDENCE / "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT.json"
    clean_clone_path = root / EVIDENCE / "SUBSTRATE_SANDBOX_CLEAN_CLONE.json"
    result = _read_json(result_path, require_digest=True)
    clean_clone = _read_json(clean_clone_path, require_digest=True)
    if result.get("schema") != "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT" or result.get("program") != PROGRAM:
        raise Refused("unexpected R2 longitudinal-result schema or program")
    if result.get("scientific_status") != "complete":
        raise Refused("R2 longitudinal result is not complete")
    result_source = result.get("source_digest")
    if not isinstance(result_source, str) or len(result_source) != 64:
        raise Refused("R2 longitudinal result lacks a source digest")
    current_source = _campaign_source_digest(root)
    if current_source != result_source:
        raise Refused("current sandbox_campaign.source_digest does not match the longitudinal result")
    if clean_clone.get("schema") != "SUBSTRATE_SANDBOX_CLEAN_CLONE":
        raise Refused("unexpected R2 clean-clone schema")
    if clean_clone.get("scientific_status") != "pass" or clean_clone.get("all_pass") is not True:
        raise Refused("clean-clone receipt is not a passing receipt")
    checkout = clean_clone.get("checkout")
    if not isinstance(checkout, dict) or checkout.get("all_pass") is not True:
        raise Refused("clean-clone checkout is not passing")
    current_head = _git_head(root)
    if checkout.get("head") != current_head:
        raise Refused("clean-clone checkout.head does not match current git HEAD")
    if clean_clone.get("source_digest") != result_source:
        raise Refused("clean-clone source_digest does not match the longitudinal result")
    receipt = _sealed_receipt(
        {
            "longitudinal_result": _sealed_ref(root, result_path, result),
            "clean_clone": _sealed_ref(root, clean_clone_path, clean_clone),
            "git_head": current_head,
            "sandbox_campaign_source_digest": current_source,
            "checks": {
                "longitudinal_result_self_digested": True,
                "current_campaign_source_matches_result": True,
                "clean_clone_self_digested": True,
                "clean_clone_passes": True,
                "clean_clone_head_matches_current_head": True,
                "clean_clone_source_matches_result": True,
            },
            "verifier_source_sha256": file_digest(Path(__file__)),
        }
    )
    _write_json(output_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify R2 continuity provenance against a clean checkout")
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if not _inside(root, output):
        print(json.dumps({"refused": "output path must stay inside repository root", "activation": False}, sort_keys=True))
        return 2
    try:
        receipt = verify(root, output)
    except Refused as error:
        print(json.dumps({"refused": str(error), "activation": False}, sort_keys=True))
        return 2
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
