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
                     "mop.temporal.runs.successors"]


def test_failed_post_core_verification_withdraws_selection_and_reverifies(monkeypatch):
    docs = {VERIFY: {"all_pass": True}, CORE: {"selected": False}}
    calls, verifies = [], 0
    _artifacts(monkeypatch, docs)

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
                     "mop.temporal.runs.successors"]
    assert docs[CORE]["selected"] is False


def test_successors_stay_closed_when_withdrawal_verification_is_not_green(monkeypatch):
    docs = {VERIFY: {"all_pass": True}, CORE: {"selected": False}}
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
    assert not supervisor.run_verified_successor_gates()
    assert calls == ["mop.temporal.runs.coresel", "mop.temporal.runs.verify",
                     "mop.temporal.runs.coresel", "mop.temporal.runs.verify"]
    assert "mop.temporal.runs.successors" not in calls
