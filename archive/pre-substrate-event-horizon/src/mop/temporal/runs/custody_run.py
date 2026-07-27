"""Data custody authority, inventory, recovery record and mutation suite.

Seals the corpus inventory, proves the deletion guard refuses the cases that already went wrong, and records
the exact commands that rebuild anything missing.

House style: no dashes.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from mop.temporal import custody as C
from mop.temporal import io

D = C.CANONICAL_ROOT

CORPORA = [
    C.Corpus(
        logical_identity="uci_har",
        official_source="UCI Machine Learning Repository, Human Activity Recognition Using Smartphones",
        license="CC BY 4.0",
        citation="Anguita et al 2013, A Public Domain Dataset for Human Activity Recognition Using Smartphones",
        archive_url_authority="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        canonical_path=str(D / "har" / "UCI HAR Dataset"),
        redownload_command="curl -L -o /tmp/har.zip https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip && unzip -o /tmp/har.zip -d " + str(D / "har"),
        rebuild_command="not derived",
        retention_class="principal_active",
        kind="raw_data",
        experiments_using_it=["E1", "E2", "har_stream"],
        derived_caches=[],
    ),
    C.Corpus(
        logical_identity="speech_commands",
        official_source="Google Speech Commands v0.02",
        license="CC BY 4.0",
        citation="Warden 2018, Speech Commands: A Dataset for Limited Vocabulary Speech Recognition",
        archive_url_authority="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        canonical_path=str(D / "speech" / "speech_commands"),
        redownload_command="curl -L -o /tmp/sc.tar.gz http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz && mkdir -p " + str(D / "speech" / "speech_commands") + " && tar xzf /tmp/sc.tar.gz -C " + str(D / "speech" / "speech_commands"),
        rebuild_command="not derived",
        retention_class="principal_active",
        kind="raw_data",
        experiments_using_it=["E1", "E2", "speech_stream"],
        derived_caches=["speech_features_cache"],
    ),
    C.Corpus(
        logical_identity="speech_features_cache",
        official_source="derived from speech_commands by fastforge.data.load_speech",
        license="inherits speech_commands",
        citation="derived artifact",
        archive_url_authority="derived",
        canonical_path=str(D / "speech" / "speech_feats.npz"),
        redownload_command="rebuild from speech_commands",
        rebuild_command="PYTHONPATH=src python3.12 -c 'from fastforge import data; data.load_speech()'",
        retention_class="derived_rebuildable",
        kind="non_rebuildable_cache",
        experiments_using_it=["E1", "E2"],
        derived_caches=[],
    ),
    C.Corpus(
        logical_identity="pamap2",
        official_source="UCI Machine Learning Repository, PAMAP2 Physical Activity Monitoring",
        license="CC BY 4.0",
        citation="Reiss and Stricker 2012, Introducing a New Benchmarked Dataset for Activity Monitoring",
        archive_url_authority="https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
        canonical_path=str(D / "pamap2" / "PAMAP2_Dataset"),
        redownload_command="python3.12 -m mop.temporal.runs.custody_run recover pamap2",
        rebuild_command="not derived",
        retention_class="secondary_active",
        kind="raw_data",
        experiments_using_it=["third bed preflight", "E2 third bed"],
        derived_caches=[],
    ),
    C.Corpus(
        logical_identity="harth",
        official_source="UCI Machine Learning Repository, HARTH human activity recognition trondheim",
        license="CC BY 4.0",
        citation="Logacjov et al 2021, HARTH: A Human Activity Recognition Dataset for Machine Learning",
        archive_url_authority="https://archive.ics.uci.edu/static/public/779/harth.zip",
        canonical_path=str(D / "harth" / "harth"),
        redownload_command="python3.12 -m mop.temporal.runs.custody_run recover harth",
        rebuild_command="not derived",
        retention_class="secondary_active",
        kind="raw_data",
        experiments_using_it=["third bed preflight", "E2 third bed"],
        derived_caches=[],
    ),
    C.Corpus(
        logical_identity="starss23",
        official_source="STARSS23 sound event localization and detection",
        license="see corpus licence file",
        citation="Politis et al 2023, STARSS23",
        archive_url_authority="https://zenodo.org/records/7880637",
        canonical_path=str(D / "starss23"),
        redownload_command="manual download from the zenodo record",
        rebuild_command="not derived",
        retention_class="historical_reproducibility",
        kind="raw_data",
        experiments_using_it=["inherited STARSS23 beds, all terminal"],
        derived_caches=[],
    ),
]

RECOVERABLE = {
    "pamap2": {
        "url": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
        "dest": D / "pamap2",
        "archive": "/tmp/pamap2.zip",
        "inner_zip": True,
    },
    "harth": {
        "url": "https://archive.ics.uci.edu/static/public/779/harth.zip",
        "dest": D / "harth",
        "archive": "/tmp/harth.zip",
        "inner_zip": False,
    },
}


# ---------------------------------------------------------------- mutations


def mutations() -> dict:
    """Every case the guard must refuse, rebuilt on a throwaway tree."""
    res = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wt = tmp / "disposable_worktree"
        (wt / "runs" / "data").mkdir(parents=True)
        (wt / "runs" / "data" / "raw.bin").write_bytes(b"x" * 1024)

        raw = C.Corpus(
            logical_identity="corpus_inside_a_disposable_worktree",
            official_source="test", license="test", citation="test",
            archive_url_authority="", canonical_path=str(wt / "runs" / "data"),
            redownload_command="", rebuild_command="", retention_class="principal_active", kind="raw_data")
        g = C.guard(wt, [raw])
        res["dataset_inside_disposable_worktree"] = {
            "expected": "refused", "actual": "refused" if not g["allowed"] else "allowed",
            "reason": g["reason"], "pass": not g["allowed"]}

        outside = tmp / "canonical" / "har"
        outside.mkdir(parents=True)
        (outside / "raw.bin").write_bytes(b"y" * 512)
        ext = C.Corpus(
            logical_identity="corpus_outside_every_worktree",
            official_source="test", license="test", citation="test",
            archive_url_authority="https://example.invalid/a.zip", canonical_path=str(outside),
            redownload_command="curl", rebuild_command="", retention_class="principal_active",
            kind="raw_data")
        g2 = C.guard(wt, [ext])
        res["external_dataset_path"] = {
            "expected": "allowed", "actual": "allowed" if g2["allowed"] else "refused",
            "pass": g2["allowed"]}

        missing = C.Corpus(
            logical_identity="missing_archive", official_source="t", license="t", citation="t",
            archive_url_authority="https://example.invalid/a.zip",
            canonical_path=str(tmp / "does_not_exist"), redownload_command="curl",
            rebuild_command="", retention_class="secondary_active", kind="raw_data")
        v = C.verify_corpus(missing)
        res["missing_archive"] = {"expected": "absent", "actual": v["status"],
                                  "recovery_recorded": bool(v["recovery"]),
                                  "pass": v["status"] == "absent" and bool(v["recovery"])}

        partial = tmp / "partial"
        partial.mkdir()
        (partial / "a.txt").write_text("a")
        pc = C.Corpus(
            logical_identity="partial_extraction", official_source="t", license="t", citation="t",
            archive_url_authority="u", canonical_path=str(partial), redownload_command="curl",
            rebuild_command="", retention_class="secondary_active", kind="raw_data",
            extracted_hashes={"a.txt": io.sha_file(partial / "a.txt"), "b.txt": "deadbeef"})
        v = C.verify_corpus(pc)
        res["partial_extraction"] = {"expected": "damaged", "actual": v["status"],
                                     "pass": v["status"] == "damaged"}

        bad = tmp / "badhash"
        bad.mkdir()
        (bad / "a.txt").write_text("changed")
        bc = C.Corpus(
            logical_identity="hash_mismatch", official_source="t", license="t", citation="t",
            archive_url_authority="u", canonical_path=str(bad), redownload_command="curl",
            rebuild_command="", retention_class="secondary_active", kind="raw_data",
            extracted_hashes={"a.txt": "0" * 64})
        v = C.verify_corpus(bc)
        res["hash_mismatch"] = {"expected": "damaged", "actual": v["status"],
                                "pass": v["status"] == "damaged"}

        srcdir = tmp / "src_corpus"
        srcdir.mkdir()
        (srcdir / "raw.bin").write_bytes(b"z")
        cachedir = wt / "cache"
        cachedir.mkdir(parents=True)
        (cachedir / "c.npz").write_bytes(b"c")
        src = C.Corpus(
            logical_identity="live_source", official_source="t", license="t", citation="t",
            archive_url_authority="u", canonical_path=str(srcdir), redownload_command="curl",
            rebuild_command="", retention_class="principal_active", kind="raw_data",
            derived_caches=["stale_cache"])
        cache = C.Corpus(
            logical_identity="stale_cache", official_source="derived", license="t", citation="t",
            archive_url_authority="derived", canonical_path=str(cachedir),
            redownload_command="rebuild", rebuild_command="python -c rebuild",
            retention_class="derived_rebuildable", kind="non_rebuildable_cache",
            derived_caches=[])
        g3 = C.guard(wt, [src, cache])
        res["stale_cache_with_live_source"] = {
            "expected": "allowed", "actual": "allowed" if g3["allowed"] else "refused",
            "pass": g3["allowed"]}

        orphan = C.Corpus(
            logical_identity="orphan_cache", official_source="derived", license="t", citation="t",
            archive_url_authority="derived", canonical_path=str(cachedir),
            redownload_command="rebuild", rebuild_command="python -c rebuild",
            retention_class="derived_rebuildable", kind="non_rebuildable_cache")
        g4 = C.guard(wt, [orphan])
        res["missing_source_with_rebuildable_cache"] = {
            "expected": "refused", "actual": "refused" if not g4["allowed"] else "allowed",
            "note": "no surviving source means the cache is the only copy",
            "pass": not g4["allowed"]}

        pub = C.Corpus(
            logical_identity="publicly_recoverable_inside_worktree", official_source="t", license="t",
            citation="t", archive_url_authority="https://example.invalid/a.zip",
            canonical_path=str(wt / "runs" / "data"), redownload_command="curl -L ...",
            rebuild_command="", retention_class="publicly_recoverable_inactive", kind="raw_data")
        g5 = C.guard(wt, [pub])
        g6 = C.guard(wt, [pub], allow_publicly_recoverable=True)
        res["redownload_recovery_override"] = {
            "expected": "refused by default and allowed only with an explicit override",
            "default_allowed": g5["allowed"], "override_allowed": g6["allowed"],
            "pass": (not g5["allowed"]) and g6["allowed"]}

        ev = {"proof/substrate/x/only_here.json"}
        res["unindexed_evidence_inside_worktree"] = {
            "expected": "refused",
            "actual": "refused" if not C.guard(io.ROOT, [], ev)["allowed"] else "allowed",
            "pass": not C.guard(io.ROOT, [], ev)["allowed"]}

        removable = tmp / "removable"
        removable.mkdir()
        r = C.remove_worktree(removable, [], dry_run=False)
        res["worktree_deletion_when_nothing_unique_is_held"] = {
            "expected": "removed", "actual": "removed" if r["removed"] else "kept",
            "pass": r["removed"] and not removable.exists()}

    res["all_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    return res


# ---------------------------------------------------------------- recovery


def recover(name: str) -> dict:
    import subprocess
    import zipfile

    spec = RECOVERABLE[name]
    t0 = time.time()
    dest = Path(spec["dest"])
    dest.mkdir(parents=True, exist_ok=True)
    archive = Path(spec["archive"])
    r = subprocess.run(["curl", "-fsSL", "--max-time", "3600", "-o", str(archive), spec["url"]],
                       capture_output=True, text=True)
    if r.returncode != 0 or not archive.is_file():
        return {"corpus": name, "ok": False, "stage": "download", "stderr": r.stderr[-400:]}
    ah = io.sha_file(archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    if spec["inner_zip"]:
        for inner in sorted(dest.rglob("*.zip")):
            with zipfile.ZipFile(inner) as z:
                z.extractall(inner.parent)
    files = [p for p in dest.rglob("*") if p.is_file()]
    return {
        "corpus": name,
        "ok": True,
        "url": spec["url"],
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": ah,
        "extracted_files": len(files),
        "extracted_bytes": sum(p.stat().st_size for p in files),
        "canonical_path": str(dest),
        "wall_seconds": round(time.time() - t0, 1),
    }


def refresh_hashes(corpora: list) -> None:
    """Record a small stable sample of extracted hashes so damage is detectable without hashing gigabytes."""
    for c in corpora:
        p = Path(c.canonical_path)
        if not p.exists() or c.extracted_hashes:
            continue
        if p.is_file():
            c.extracted_hashes = {p.name: io.sha_file(p)}
            continue
        files = sorted(x for x in p.rglob("*") if x.is_file())
        pick = files[:3] + files[len(files) // 2 : len(files) // 2 + 2] + files[-2:]
        c.extracted_hashes = {str(f.relative_to(p)): io.sha_file(f) for f in pick[:7]}


def main(argv=None):
    import sys

    argv = argv or sys.argv[1:]
    if argv and argv[0] == "recover":
        r = recover(argv[1])
        io.run_json(f"recover_{argv[1]}.json", r, "custody")
        print(json.dumps(r)[:300], flush=True)
        print("RECOVER_DONE", flush=True)
        return

    t0 = time.time()
    refresh_hashes(CORPORA)
    inv = inventory_doc()
    io.seal("MOP_DATA_CUSTODY_INVENTORY.json", inv)
    mut = mutations()
    io.seal("MOP_DATA_CUSTODY_MUTATIONS.json", {"schema": "mop-data-custody-mutations/v1", **mut})
    verifications = {c.logical_identity: C.verify_corpus(c) for c in CORPORA}
    io.seal("MOP_DATA_CUSTODY_AUTHORITY.json", {
        "schema": "mop-data-custody-authority/v1",
        "canonical_root": str(D),
        "rule": ("no corpus, non rebuildable cache, split authority, principal checkpoint or unindexed "
                 "evidence may live only inside a disposable worktree. The guard refuses removal when it "
                 "does, and the refusal is the default"),
        "retention_classes": list(C.RETENTION_CLASSES),
        "unique_kinds": list(C.UNIQUE_KINDS),
        "n_corpora": len(CORPORA),
        "verifications": verifications,
        "intact": [k for k, v in verifications.items() if v["status"] == "intact"],
        "absent": [k for k, v in verifications.items() if v["status"] == "absent"],
        "damaged": [k for k, v in verifications.items() if v["status"] == "damaged"],
        "guard_mutations_all_pass": mut["all_pass"],
        "inventory_declared": inv["all_declared"],
        "wall_seconds": round(time.time() - t0, 1),
    })
    rows = "\n".join(
        f"| {c.logical_identity} | {c.retention_class} | {'present' if c.present() else 'ABSENT'} | "
        f"`{c.canonical_path}` |" for c in CORPORA)
    io.seal_md("MOP_DATA_CUSTODY_RECOVERY.md", f"""# Data custody and recovery

Canonical root: `{D}`. Nothing under a worktree.

| corpus | retention | state | canonical path |
|---|---|---|---|
{rows}

## What went wrong

Two worktrees pointed absolute data paths at each other. Removing one destroyed the only local copies of
PAMAP2 and HARTH. Nothing sealed depended on them, which is why it went unnoticed, and that is the reason the
guard exists rather than a reason it does not need to.

## Recovery

Every corpus records a re download command. For the two lost corpora:

```
python3.12 -m mop.temporal.runs.custody_run recover pamap2
python3.12 -m mop.temporal.runs.custody_run recover harth
```

## The guard

`mop.temporal.custody.guard` refuses removal of any directory that uniquely holds raw data, a non rebuildable
cache, a split authority, a principal checkpoint or unindexed evidence. A publicly recoverable corpus can be
released only through an explicit override, never by default. {len([k for k, v in mutations().items() if isinstance(v, dict)])} guard mutations are
sealed in `MOP_DATA_CUSTODY_MUTATIONS.json`.
""")
    print(f"custody: {len(CORPORA)} corpora, mutations {mut['all_pass']}, "
          f"absent {[k for k, v in verifications.items() if v['status'] == 'absent']}", flush=True)
    print("CUSTODY_DONE", flush=True)


def inventory_doc() -> dict:
    return C.inventory(CORPORA)


if __name__ == "__main__":
    main()
