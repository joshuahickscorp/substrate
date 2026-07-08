import json

from mop.studio.long_run import DaemonJob, load_plan, run_daemon, write_plan_template


def test_template_plan_round_trips(tmp_path):
    plan_path = tmp_path / "plan.json"
    write_plan_template(plan_path)
    jobs = load_plan(plan_path)
    assert [j.job_id for j in jobs] == [
        "transfer_check",
        "doctor",
        "profiles",
        "docs_gate",
        "acceptance",
        "dr1_smoke",
        "encode_microbench",
        "wave0_report",
    ]


def test_daemon_dry_run_writes_resumable_state(tmp_path):
    jobs = [DaemonJob("a", ("python", "-V")), DaemonJob("b", ("python", "-V"))]
    state = run_daemon(
        jobs,
        out_dir=tmp_path,
        profile_name="m3pro-local-max",
        execute=False,
        disk_probe=lambda: (True, 100.0),
    )
    assert state["summary"] == {"dry-run": 2}
    saved = json.loads((tmp_path / "daemon_state.json").read_text())
    assert saved["jobs"]["a"]["status"] == "dry-run"


def test_daemon_resume_skips_successful_jobs(tmp_path):
    calls = []

    def runner(job, out_dir):
        calls.append(job.job_id)
        return 0

    jobs = [DaemonJob("a", ("ok",)), DaemonJob("b", ("ok",))]
    first = run_daemon(
        jobs,
        out_dir=tmp_path,
        profile_name="m3pro-local-max",
        execute=True,
        runner=runner,
        disk_probe=lambda: (True, 100.0),
    )
    assert first["summary"] == {"success": 2}
    second = run_daemon(
        jobs,
        out_dir=tmp_path,
        profile_name="m3pro-local-max",
        execute=True,
        runner=runner,
        disk_probe=lambda: (True, 100.0),
    )
    assert second["summary"] == {"success": 2}
    assert calls == ["a", "b"]


def test_daemon_blocks_on_disk_floor_before_start(tmp_path):
    jobs = [DaemonJob("a", ("ok",))]
    state = run_daemon(
        jobs,
        out_dir=tmp_path,
        profile_name="studio-m1ultra",
        execute=True,
        runner=lambda _job, _out: 0,
        disk_probe=lambda: (False, 42.0),
    )
    assert state["summary"] == {"blocked": 1}
    assert "free disk" in state["jobs"]["a"]["reason"]


def test_daemon_stops_after_failure(tmp_path):
    def runner(job, out_dir):
        return 7 if job.job_id == "a" else 0

    jobs = [DaemonJob("a", ("bad",)), DaemonJob("b", ("ok",))]
    state = run_daemon(
        jobs,
        out_dir=tmp_path,
        profile_name="m3pro-local-max",
        execute=True,
        runner=runner,
        disk_probe=lambda: (True, 100.0),
    )
    assert state["summary"] == {"failed": 1}
    assert "b" not in state["jobs"]
