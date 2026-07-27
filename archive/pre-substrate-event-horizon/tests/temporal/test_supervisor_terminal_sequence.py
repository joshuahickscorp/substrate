from mop.temporal.runs import supervisor


VERIFY = "MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json"
CORE = "MOP_OWNED_TEMPORAL_CORE_V1.json"


def _artifacts(monkeypatch, docs):
    monkeypatch.setattr(supervisor.io, "exists", lambda name: name in docs)
    monkeypatch.setattr(supervisor.io, "load", lambda name: docs[name])


def test_core_selection_is_independently_verified_before_successor_gating(monkeypatch):
    docs = {VERIFY: {"all_pass": True}, CORE: {"selected": True}}
    calls = []
    _artifacts(monkeypatch, docs)
    monkeypatch.setattr(supervisor, "run_sync", lambda mod, args=(): calls.append(mod) or True)

    assert supervisor.run_verified_successor_gates()
    assert calls == ["mop.temporal.runs.coresel", "mop.temporal.runs.verify",
                     "mop.temporal.runs.successors", "mop.temporal.runs.verify"]


def test_failed_post_core_verification_withdraws_selection_and_reverifies(monkeypatch, tmp_path):
    docs = {VERIFY: {"all_pass": True}, CORE: {"selected": False}}
    calls, verifies = [], 0
    _artifacts(monkeypatch, docs)
    monkeypatch.setattr(supervisor.io, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor.io, "PROOF", tmp_path / "proof")
    checkpoint_root = supervisor.io.PROOF / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "owned_temporal_core_v1_har_stream.pt").write_bytes(b"packaged")

    def run(mod, args=()):
        nonlocal verifies
        calls.append(mod)
        if mod.endswith("coresel"):
            docs[CORE]["selected"] = docs[VERIFY]["all_pass"]
        elif mod.endswith("verify"):
            verifies += 1
            docs[VERIFY]["all_pass"] = verifies > 1
        return True

    monkeypatch.setattr(supervisor, "run_sync", run)
    assert supervisor.run_verified_successor_gates()
    assert calls == ["mop.temporal.runs.coresel", "mop.temporal.runs.verify",
                     "mop.temporal.runs.coresel", "mop.temporal.runs.verify",
                     "mop.temporal.runs.successors", "mop.temporal.runs.verify"]
    assert docs[CORE]["selected"] is False
    assert not list(checkpoint_root.glob("owned_temporal_core_v1_*.pt"))
    quarantined = list((supervisor.io.PROOF / "checkpoint_quarantine").rglob("*.pt"))
    assert len(quarantined) == 1 and quarantined[0].read_bytes() == b"packaged"


def test_negative_verified_core_decision_seals_closed_successor_gates(monkeypatch):
    docs = {
        VERIFY: {"all_pass": True, "role_b": {"checks": {"a": True}},
                 "role_c": {"n_checks": 1}},
        CORE: {"selected": False, "selection": {"reason": "load bearing evidence is red"}},
    }
    calls = []
    _artifacts(monkeypatch, docs)

    def run(mod, args=()):
        calls.append(mod)
        if mod.endswith("verify"):
            docs[VERIFY]["all_pass"] = False
        elif mod.endswith("coresel"):
            docs[CORE]["selected"] = docs[VERIFY]["all_pass"]
        return True

    monkeypatch.setattr(supervisor, "run_sync", run)
    assert supervisor.run_verified_successor_gates()
    assert calls == ["mop.temporal.runs.coresel", "mop.temporal.runs.verify",
                     "mop.temporal.runs.coresel", "mop.temporal.runs.verify",
                     "mop.temporal.runs.successors", "mop.temporal.runs.verify"]


def test_successor_gating_stops_when_negative_core_decision_is_incomplete(monkeypatch):
    docs = {VERIFY: {"all_pass": False}, CORE: {"selected": False}}
    calls = []
    _artifacts(monkeypatch, docs)
    monkeypatch.setattr(supervisor, "run_sync", lambda mod, args=(): calls.append(mod) or True)
    assert not supervisor.run_verified_successor_gates()
    assert calls == ["mop.temporal.runs.coresel", "mop.temporal.runs.verify",
                     "mop.temporal.runs.coresel", "mop.temporal.runs.verify"]


def test_successor_result_refresh_verifies_the_resealed_queue():
    assert supervisor.SUCCESSOR_REFRESH[-2:] == (
        "mop.temporal.runs.successors",
        "mop.temporal.runs.verify",
    )
