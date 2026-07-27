from mop.temporal import factorial as Fx
from mop.temporal.runs import e2


def test_only_appended_convergence_identities_are_parallel_backfilled(monkeypatch, tmp_path):
    configs = [dict(Fx.REFERENCE) for _ in range(e2.LEGACY_CONVERGENCE_CONFIG_COUNT + 2)]
    calls = []
    monkeypatch.setattr(e2, "CONVERGE_CONFIGS", configs)
    monkeypatch.setattr(e2.io, "RUNS", tmp_path)
    monkeypatch.setattr(e2, "_parallel_backfill",
                        lambda bed, command, indices: calls.append((bed, command, list(indices))))
    e2._backfill_appended_convergence("har_stream")
    expected = [e2.LEGACY_CONVERGENCE_CONFIG_COUNT, e2.LEGACY_CONVERGENCE_CONFIG_COUNT + 1]
    assert calls == [("har_stream", "converge_shard", expected),
                     ("har_stream", "extend_converge_shard", expected)]


def test_predecessor_identity_range_never_triggers_backfill(monkeypatch, tmp_path):
    monkeypatch.setattr(e2, "CONVERGE_CONFIGS",
                        [dict(Fx.REFERENCE) for _ in range(e2.LEGACY_CONVERGENCE_CONFIG_COUNT)])
    monkeypatch.setattr(e2.io, "RUNS", tmp_path)
    monkeypatch.setattr(e2, "_parallel_backfill",
                        lambda *_: (_ for _ in ()).throw(AssertionError("unexpected backfill")))
    e2._backfill_appended_convergence("speech_stream")
