"""Clean clone: does a fresh checkout of this commit reproduce what the working tree claims.

The working tree is the least trustworthy place to verify anything. It has caches, untracked files, a
`runs` directory full of state, and whatever the last command left behind. A clean clone at the exact
commit has none of that, so it is the only honest answer to whether the evidence travels.

The check that matters most is the last one. Regenerating the sealed artifacts in the clone and comparing
their hashes against the committed ones proves the artifacts are a function of the tree rather than of the
machine that happened to produce them.

House style: no dashes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from mop.cognition import io

PY = sys.executable
CHECKS = ("exact_commit_checkout", "package_import", "declared_tests", "graph_valid",
          "artifacts_regenerate_identically", "independent_recomputation", "no_activation")


class Refused(RuntimeError):
    """A clean clone the check will not certify."""


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> tuple[int, str]:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def run(keep: bool = False) -> dict:
    commit = io.commit()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "tests"],
                           cwd=io.ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise Refused(f"the working tree is not committed, so a clone cannot verify it: {dirty[:200]}")

    tmp = Path(tempfile.mkdtemp(prefix="substrate-cleanclone-"))
    clone = tmp / "mop"
    results: dict[str, dict] = {}
    try:
        code, out = _run(["git", "clone", "--quiet", "--no-hardlinks", str(io.ROOT), str(clone)], tmp)
        results["clone"] = {"ok": code == 0, "detail": out[-300:]}
        code, out = _run(["git", "checkout", "--quiet", commit], clone)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True,
                              text=True).stdout.strip()
        results["exact_commit_checkout"] = {"ok": head == commit, "head": head, "expected": commit}

        env = {**__import__("os").environ, "PYTHONPATH": str(clone / "src"),
               "PYTHONDONTWRITEBYTECODE": "1"}
        code, out = _run([PY, "-c", "import mop.cognition.program, mop.cognition.runtime"], clone, env)
        results["package_import"] = {"ok": code == 0, "detail": out[-300:]}

        code, out = _run([PY, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
                          "tests/cognition"], clone, env)
        results["declared_tests"] = {"ok": code == 0, "detail": out.strip().splitlines()[-1:]}

        code, out = _run([PY, "-c",
                          "import json,sys;from mop.cognition import graph;"
                          "d=graph.declaration();print(json.dumps({'valid':d['valid'],"
                          "'blocked':d['externally_blocked']}))"], clone, env)
        results["graph_valid"] = {"ok": code == 0 and '"valid": true' in out.replace("'", '"').lower(),
                                  "detail": out.strip()[-200:]}

        # The load bearing check: are the artifacts a function of the tree or of the machine.
        #
        # Two fields must be excluded from the comparison or it measures the wrong thing. source_commit
        # is stamped at seal time, and a committed artifact was necessarily sealed at an earlier commit
        # than the one being cloned, so including it reports every artifact as drifted on every run.
        # sha256 is a digest over the rest and moves with it. What is compared is the content.
        volatile = {"source_commit", "sha256"}
        # Two artifacts report on the tree as it stands after the commit, so they cannot be reproduced
        # from the commit alone and their drift is bootstrap rather than machine dependence. The clean
        # clone receipt is produced by cloning the commit that would have to contain it, and the
        # structural audit enumerates the artifacts on disk, which in a clone excludes both of these.
        bootstrap = {"SUBSTRATE_CLEAN_CLONE.json", "SUBSTRATE_STRUCTURAL_AUDIT.json"}

        def content(path: Path) -> str:
            try:
                doc = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                return "unreadable"
            return io.sha_obj({k: v for k, v in doc.items() if k not in volatile})

        proof = clone / io.PROOF.relative_to(io.ROOT)
        before = {p.name: content(p) for p in sorted(proof.glob("SUBSTRATE_*.json"))}
        code, out = _run([PY, "-m", "mop.cognition.deliverables", "seal-modules"], clone, env)
        once = {p.name: content(p) for p in sorted(proof.glob("SUBSTRATE_*.json"))}
        code2, _ = _run([PY, "-m", "mop.cognition.deliverables", "seal-modules"], clone, env)
        twice = {p.name: content(p) for p in sorted(proof.glob("SUBSTRATE_*.json"))}

        drifted = sorted(k for k in before
                         if k in once and before[k] != once[k] and k not in bootstrap)
        nondeterministic = sorted(k for k in once if k in twice and once[k] != twice[k])
        results["artifacts_regenerate_identically"] = {
            "ok": code == 0 and code2 == 0 and not drifted and not nondeterministic,
            "drifted_from_committed": drifted,
            "nondeterministic_across_two_runs": nondeterministic,
            "compared": len(before), "excluded_fields": sorted(volatile),
            "bootstrap_artifacts_excluded": sorted(bootstrap),
            "why_bootstrap": ("both report on the tree as it stands after the commit, so neither can be "
                              "reproduced from the commit alone. Nondeterminism is still checked for "
                              "them, because that would be a real defect"),
            "detail": out[-300:] if drifted else "",
            "note": ("content is compared with the commit stamp and its digest excluded, because a "
                     "committed artifact was sealed at an earlier commit by construction. An artifact "
                     "whose content depends on the machine shows up as drifted; one that differs between "
                     "two runs in the same clone shows up as nondeterministic")}

        code, out = _run([PY, "-m", "mop.cognition.verify", "recompute"], clone, env)
        results["independent_recomputation"] = {"ok": code == 0 and "0 disagreements" in out,
                                                "detail": out.strip()[-200:]}

        code, out = _run([PY, "-c",
                          "import json;from mop.cognition import runtime;"
                          "print(json.dumps(runtime.declaration()['activation']))"], clone, env)
        results["no_activation"] = {"ok": code == 0 and out.strip() == "false", "detail": out.strip()}
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)

    checks = {k: bool(v.get("ok")) for k, v in results.items()}
    return {"schema": "substrate-clean-clone/v1", "commit": commit, "checks": checks,
            "detail": results, "failed": sorted(k for k, v in checks.items() if not v),
            "all_pass": all(checks.values()),
            "why_a_clone": ("the working tree has caches, untracked state and whatever the last command "
                            "left behind. A clone at the exact commit has none of it"),
            "activation": False}


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] not in ("run", "seal"):
        raise ValueError(argv)
    doc = run()
    path = io.seal("SUBSTRATE_CLEAN_CLONE.json", doc)
    print(json.dumps({"sealed": path.relative_to(io.ROOT).as_posix(),
                      "all_pass": doc["all_pass"], "failed": doc["failed"],
                      "checks": doc["checks"]}, indent=2))


if __name__ == "__main__":
    main()
