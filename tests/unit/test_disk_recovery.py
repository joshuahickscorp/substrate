import subprocess

from mop.studio.disk_recovery import DiskRecoveryConfig, build_disk_recovery_plan, write_disk_recovery_plan


def _git_init(root):
    subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.DEVNULL)


def test_tool_cache_is_safe_dry_run_candidate(tmp_path):
    _git_init(tmp_path)
    (tmp_path / ".gitignore").write_text(".pytest_cache/\n")
    cache = tmp_path / ".pytest_cache"
    cache.mkdir()
    (cache / "node").write_bytes(b"x" * 12)
    (cache / "README.md").write_text("tool cache documentation, not a project receipt\n")

    report = build_disk_recovery_plan(
        DiskRecoveryConfig(repo_root=tmp_path, scan_paths=(".pytest_cache",), include_defaults=False)
    )

    candidate = report["candidates"][0]
    assert report["all_ok"] is True
    assert candidate["kind"] == "repo_tool_cache"
    assert candidate["safe_to_delete"] is True
    assert report["actions"][0]["status"] == "would_delete"
    assert cache.exists()


def test_unbundled_receipt_blocks_generated_run_deletion(tmp_path):
    _git_init(tmp_path)
    (tmp_path / ".gitignore").write_text("runs/\n")
    run = tmp_path / "runs" / "e1_baseline" / "001"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text('{"ok": true}\n')

    report = build_disk_recovery_plan(
        DiskRecoveryConfig(repo_root=tmp_path, scan_paths=(run,), include_defaults=False)
    )

    candidate = report["candidates"][0]
    assert candidate["kind"] == "repo_ignored_run"
    assert candidate["safe_to_delete"] is False
    assert "unbundled receipt-like" in " ".join(candidate["blockers"])
    assert report["actions"][0]["status"] == "blocked"


def test_tracked_file_protects_even_ignored_run(tmp_path):
    _git_init(tmp_path)
    (tmp_path / ".gitignore").write_text("runs/\n")
    run = tmp_path / "runs" / "e1_baseline" / "002"
    run.mkdir(parents=True)
    receipt = run / "metrics.json"
    receipt.write_text('{"ok": true}\n')
    subprocess.run(["git", "add", "-f", str(receipt.relative_to(tmp_path))], cwd=tmp_path, check=True)

    report = build_disk_recovery_plan(
        DiskRecoveryConfig(repo_root=tmp_path, scan_paths=(run,), include_defaults=False)
    )

    candidate = report["candidates"][0]
    assert candidate["git_tracked_count"] == 1
    assert candidate["safe_to_delete"] is False
    assert "git-tracked" in " ".join(candidate["blockers"])


def test_execute_requires_explicit_allow_rule(tmp_path):
    _git_init(tmp_path)
    (tmp_path / ".gitignore").write_text(".pytest_cache/\n")
    cache = tmp_path / ".pytest_cache"
    cache.mkdir()
    (cache / "node").write_bytes(b"x")

    report = build_disk_recovery_plan(
        DiskRecoveryConfig(
            repo_root=tmp_path,
            scan_paths=(".pytest_cache",),
            include_defaults=False,
            execute=True,
        )
    )

    assert report["all_ok"] is False
    assert "execute requires" in report["problems"][0]
    assert cache.exists()


def test_execute_deletes_only_allowed_safe_candidate(tmp_path):
    _git_init(tmp_path)
    (tmp_path / ".gitignore").write_text(".pytest_cache/\n")
    cache = tmp_path / ".pytest_cache"
    cache.mkdir()
    (cache / "node").write_bytes(b"x")

    report = build_disk_recovery_plan(
        DiskRecoveryConfig(
            repo_root=tmp_path,
            scan_paths=(".pytest_cache",),
            include_defaults=False,
            execute=True,
            allow_classes=("repo_tool_cache",),
        )
    )

    assert report["all_ok"] is True
    assert report["actions"][0]["status"] == "deleted"
    assert not cache.exists()


def test_write_disk_recovery_plan_round_trips(tmp_path):
    _git_init(tmp_path)
    report = build_disk_recovery_plan(DiskRecoveryConfig(repo_root=tmp_path, include_defaults=False))
    out = tmp_path / "plan.json"
    write_disk_recovery_plan(report, out)
    assert "mop-disk-recovery-plan/v1" in out.read_text()
