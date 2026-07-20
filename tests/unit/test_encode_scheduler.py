from mop.studio.encode_scheduler import EncodeBenchmark, benchmark_from_autoselect, plan_encode

ENC = {"name": "vjepa2_vitl_fpc64_256", "embed_dim": 1024}


def test_scheduler_picks_parallel_cpu_when_mps_slower():
    plan = plan_encode(
        profile_name="studio-m1ultra",
        benchmark=EncodeBenchmark(cpu_s_per_clip=16.0, mps_s_per_clip=3.0),
        encoder_config=ENC,
        requested_clips=1000,
        free_gb=7000.0,
    )
    assert plan["ok_to_launch"]
    assert plan["winner"]["device"] == "cpu"
    assert plan["winner"]["workers"] == 16
    assert plan["winner"]["wall_s_per_clip"] == 1.0
    assert plan["checkpoint"]["every_clips"] == 1800


def test_scheduler_picks_mps_when_it_beats_parallel_cpu():
    plan = plan_encode(
        profile_name="studio-m1ultra",
        benchmark=EncodeBenchmark(cpu_s_per_clip=16.0, mps_s_per_clip=0.5),
        encoder_config=ENC,
        requested_clips=1000,
        free_gb=7000.0,
    )
    assert plan["ok_to_launch"]
    assert plan["winner"]["device"] == "mps"
    assert plan["winner"]["wall_s_per_clip"] == 0.5


def test_scheduler_blocks_when_dense_cache_would_breach_floor():
    plan = plan_encode(
        profile_name="studio-m1ultra",
        benchmark=EncodeBenchmark(cpu_s_per_clip=16.0, mps_s_per_clip=None),
        encoder_config=ENC,
        requested_clips=100_000,
        dense=True,
        free_gb=500.0,
    )
    assert not plan["ok_to_launch"]
    assert any("projected after" in r for r in plan["blocked_reasons"])


def test_scheduler_clamps_clip_cap_as_warning_not_hard_block():
    plan = plan_encode(
        profile_name="m3pro-local-max",
        benchmark=EncodeBenchmark(cpu_s_per_clip=10.0),
        encoder_config=ENC,
        requested_clips=999,
        free_gb=100.0,
    )
    assert plan["effective_clips"] == 128
    assert plan["ok_to_launch"]
    cap_gate = next(g for g in plan["gates"] if g["name"] == "clip_cap")
    assert cap_gate["ok"] is False and cap_gate["warning_only"] is True


def test_autoselect_benchmark_ignores_failed_mps_string():
    bench = benchmark_from_autoselect({"cpu_s_per_clip": 12.0, "mps": "failed:RuntimeError", "n_clips": 3})
    assert bench.cpu_s_per_clip == 12.0
    assert bench.mps_s_per_clip is None
    assert bench.n_clips == 3


def test_scheduler_carries_memory_envelope_from_autoselect():
    envelope = {"schema": "mop-memory-envelope/v1", "n_samples": 2}
    plan = plan_encode(
        profile_name="studio-m1ultra",
        benchmark={"cpu_s_per_clip": 12.0, "mps": None, "n_clips": 2, "memory_envelope": envelope},
        encoder_config=ENC,
        requested_clips=100,
        free_gb=7000.0,
    )
    assert plan["memory_envelope"] == envelope


def test_scheduler_blocks_when_autoselect_has_no_speed():
    plan = plan_encode(
        profile_name="studio-m1ultra",
        benchmark={"winner": "blocked", "cpu_s_per_clip": None, "mps": "not-tested", "n_clips": 0},
        encoder_config=ENC,
        requested_clips=100,
        free_gb=7000.0,
    )
    assert plan["ok_to_launch"] is False
    assert any("no usable" in r for r in plan["blocked_reasons"])
