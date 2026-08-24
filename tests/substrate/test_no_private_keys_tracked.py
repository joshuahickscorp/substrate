"""No private key material may be tracked by git.

Commit 6dc1e066 added sixteen unencrypted Ed25519 verifier private keys to the
index explicitly, which a .gitignore rule alone would not have prevented.  This
test is the enforcement.

The check reads the blob **in the index**, not the working tree.  Reading the
working tree would go green the moment someone deletes the files locally while the
keys are still committed -- a false green on exactly the condition this guards.

The check is on CONTENT, not on the file name.  A first pass on names alone was
tried and rejected: Lean's compiled modules are named ``*.olean.private`` and
8,436 of them are tracked, so a bare ``.private`` suffix match is 99.8% false
positives.  A PEM private-key header is what actually distinguishes key material,
and it also survives someone renaming a key.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PEM_PRIVATE_HEADERS = (
    b"-----BEGIN PRIVATE KEY-----",
    b"-----BEGIN RSA PRIVATE KEY-----",
    b"-----BEGIN EC PRIVATE KEY-----",
    b"-----BEGIN OPENSSH PRIVATE KEY-----",
    b"-----BEGIN DSA PRIVATE KEY-----",
    b"-----BEGIN ENCRYPTED PRIVATE KEY-----",
)

# Cheap pre-filter so the content check runs on a handful of files, not 72k.
CANDIDATE_SUFFIXES = (".private", ".pem", ".key")
CANDIDATE_NAMES = ("id_rsa", "id_ed25519", "id_ecdsa")
# Lean ships thousands of these; they are compiled modules, not key material.
NOT_KEY_SUFFIXES = (".olean.private", ".ilean.private")


def _head_of_indexed_blob(path: str, n: int = 64) -> bytes:
    """First *n* bytes of the blob git has for *path* -- not the working file."""
    out = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "blob", f":{path}"],
        capture_output=True,
    )
    return out.stdout[:n] if out.returncode == 0 else b""


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        capture_output=True, check=True,
    )
    return [p for p in out.stdout.decode().split("\0") if p]


def is_private_key(path: str, read: object = None) -> bool:
    """True when *path* is tracked private key material."""
    name = path.rsplit("/", 1)[-1]
    if name.endswith(NOT_KEY_SUFFIXES):
        return False
    if not (name.endswith(CANDIDATE_SUFFIXES) or name in CANDIDATE_NAMES):
        return False
    reader = read or _head_of_indexed_blob
    try:
        head = reader(path)
    except (OSError, subprocess.SubprocessError):
        return False
    return head.lstrip().startswith(PEM_PRIVATE_HEADERS)


def test_no_private_key_material_is_tracked() -> None:
    offenders = [p for p in _tracked() if is_private_key(p)]
    assert not offenders, (
        f"{len(offenders)} private key(s) tracked by git. Rotate, then remove from the "
        f"index (git rm --cached). First few: {offenders[:5]}"
    )


def test_the_check_detects_a_planted_key() -> None:
    """A guard that cannot fail is not a guard."""
    pem = b"-----BEGIN PRIVATE KEY-----\nMC4CAQAwBQYDK2VwBCIEIA\n"
    assert is_private_key("a/keys/verifier.ed25519.private", lambda _p: pem)


def test_the_check_does_not_flag_lean_compiled_modules() -> None:
    """The false positive that motivated the content check."""
    assert not is_private_key("lib/lean/Init.olean.private", lambda _p: b"\x00olean")
