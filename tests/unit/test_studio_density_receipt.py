import json
import subprocess

import scripts.studio_density_receipt as density_cli

from mop.studio.density_receipt import (
    DensityReceiptConfig,
    build_density_receipt,
    write_density_receipt,
)


def _git_init(root):
    subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.DEVNULL)


def test_density_receipt_reports_workspace_artifact_mass_and_cleanup(tmp_path):
    _git_init(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n")
    (tmp_path / "runs" / "studio_wave0").mkdir(parents=True)
    (tmp_path / "runs" / "studio_wave0" / "receipt.json").write_text("{}\n")
    (tmp_path / "data" / "cache" / "x").mkdir(parents=True)
    (tmp_path / "data" / "cache" / "x" / "cache_manifest.json").write_text("{}\n")
    disk = tmp_path / "runs" / "studio_wave0" / "disk_recovery.json"
    disk.write_text(
        json.dumps(
            {
                "schema": "mop-disk-recovery-plan/v1",
                "all_ok": True,
                "dry_run": False,
                "execute_requested": True,
                "summary": {
                    "deleted_bytes": 12,
                    "would_delete_bytes": 0,
                    "safe_bytes": 12,
                    "blocked_bytes": 0,
                },
                "actions": [
                    {
                        "display_path": ".pytest_cache",
                        "kind": "repo_tool_cache",
                        "status": "deleted",
                        "size_bytes": 12,
                    }
                ],
            }
        )
    )
    subprocess.run(["git", "add", "src/app.py"], cwd=tmp_path, check=True)

    receipt = build_density_receipt(DensityReceiptConfig(repo_root=tmp_path, disk_recovery_path=disk))

    assert receipt["schema"] == "mop-studio-density-receipt/v1"
    assert receipt["all_ok"] is True
    assert receipt["workspace"]["total_files"] >= 3
    assert receipt["source_loc"]["total_lines"] == 1
    assert receipt["artifact_mass"]["runs"]["total_bytes"] > 0
    assert receipt["artifact_mass"]["data/cache"]["total_bytes"] > 0
    assert receipt["before_after"]["known_cleanup_delta_bytes"] == 12
    assert receipt["cleanup"]["largest_targets"][0]["display_path"] == ".pytest_cache"


def test_density_receipt_flags_unexpected_disk_recovery_schema(tmp_path):
    disk = tmp_path / "disk.json"
    disk.write_text(json.dumps({"schema": "wrong"}))

    receipt = build_density_receipt(DensityReceiptConfig(repo_root=tmp_path, disk_recovery_path=disk))

    assert receipt["all_ok"] is False
    assert "unexpected disk recovery schema" in receipt["problems"][0]


def test_density_receipt_cli_writes_receipt(tmp_path):
    out = tmp_path / "density.json"
    rc = density_cli.main(
        ["--repo-root", str(tmp_path), "--disk-recovery", str(tmp_path / "missing.json"), "--out", str(out)]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["schema"] == "mop-studio-density-receipt/v1"


def test_write_density_receipt_round_trips(tmp_path):
    receipt = build_density_receipt(
        DensityReceiptConfig(repo_root=tmp_path, disk_recovery_path=tmp_path / "missing.json")
    )
    out = tmp_path / "density.json"
    write_density_receipt(receipt, out)
    assert json.loads(out.read_text())["schema"] == "mop-studio-density-receipt/v1"
