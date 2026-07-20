import torch
from omegaconf import OmegaConf

from mop.shell import (
    EWC,
    SI,
    Consolidation,
    Ensemble,
    GaussianHead,
    Neuromodulation,
    PlasticityController,
    Predictor,
    ReplayBuffer,
    build_modulation,
)
from mop.shell.heads import ClassHead


def test_predictor_shapes_and_action_conditioning():
    p = Predictor(dim=16, hidden=32, depth=2)
    assert p(torch.randn(8, 16)).shape == (8, 16)
    ac = Predictor(dim=16, hidden=32, depth=1, action_dim=4)
    assert ac(torch.randn(8, 16), torch.randn(8, 4)).shape == (8, 16)


def test_gaussian_head_nll_and_calibration_split():
    h = GaussianHead(dim=16, out=16)
    x, t = torch.randn(8, 16), torch.randn(8, 16)
    loss = h.nll(x, t)
    assert loss.ndim == 0 and torch.isfinite(loss)
    mean, var = h.predict(x)
    assert mean.shape == (8, 16) and (var > 0).all()


def test_ensemble_disagreement_zero_when_identical():
    head = ClassHead(8, 3, depth=0)
    ens = Ensemble(lambda: head, size=3)  # same module -> identical outputs
    _, dis = ens.mean_and_disagreement(torch.randn(5, 8))
    assert torch.allclose(dis, torch.zeros(5), atol=1e-6)


def test_ensemble_disagreement_positive_when_varied():
    ens = Ensemble(lambda: ClassHead(8, 3, depth=0), size=4)
    _, dis = ens.mean_and_disagreement(torch.randn(5, 8))
    assert (dis > 0).all()


def _buf(**kw):
    d = dict(capacity=64, dim=8, prioritized=True, eviction="reservoir", seed=0)
    d.update(kw)
    return ReplayBuffer(**d)


def test_buffer_add_and_capacity():
    b = _buf(capacity=10)
    for _ in range(50):
        b.add(torch.randn(1, 8), torch.zeros(1, dtype=torch.long))
    assert len(b) == 10  # never exceeds capacity


def test_buffer_prioritized_sampling_prefers_high_priority():
    b = _buf(capacity=100, eviction="fifo")
    x = torch.randn(100, 8)
    y = torch.zeros(100, dtype=torch.long)
    prio = torch.ones(100)
    prio[0] = 1000.0  # one very high-priority item
    b.add(x, y, priority=prio)
    counts = 0
    for _ in range(50):
        s = b.sample(8)
        counts += int((s["idx"] == 0).sum())
    assert counts > 5  # high-priority item shows up far more than 1/100 chance


def test_buffer_is_weights_in_unit_range():
    b = _buf(capacity=50, eviction="fifo")
    b.add(torch.randn(50, 8), torch.zeros(50, dtype=torch.long), priority=torch.rand(50) + 0.1)
    w = b.sample(16)["is_weight"]
    assert w.max() <= 1.0 + 1e-5 and w.min() > 0


def test_reservoir_is_uniform_over_stream():
    b = _buf(capacity=100, eviction="reservoir", seed=0)
    n = 3000
    for i in range(n):
        b.add(torch.randn(1, 8), torch.tensor([i]))  # label = stream index
    assert b.seen == n
    retained = b.y[: b.size]
    from_first_fifth = int((retained < n // 5).sum())
    assert from_first_fifth >= 6, f"recency-biased: only {from_first_fifth} of 100 from first 20%"


def test_buffer_retrieval_finds_nearest(tmp_path):
    b = _buf(capacity=50, eviction="fifo", index="brute")
    x = torch.eye(8).repeat(6, 1)[:50]
    b.add(x, torch.arange(50) % 8)
    q = x[3:4]
    r = b.retrieve(q, k=1)
    assert torch.allclose(b.x[r["idx"][0, 0]], x[3], atol=1e-5)


def test_ewc_penalty_hand_case():
    lin = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        lin.weight.fill_(0.0)
    ewc = EWC(lam=2.0)
    ewc.fisher = {"weight": torch.tensor([[3.0]])}
    ewc.star = {"weight": torch.tensor([[0.0]])}
    with torch.no_grad():
        lin.weight.fill_(4.0)  # moved by 4 from star
    assert torch.isclose(ewc.penalty(lin), torch.tensor(96.0))


def test_si_penalty_hand_case():
    lin = torch.nn.Linear(1, 1, bias=False)
    si = SI(c=0.5)
    si.omega = {"weight": torch.tensor([[2.0]])}
    si.star = {"weight": torch.tensor([[1.0]])}
    with torch.no_grad():
        lin.weight.fill_(3.0)  # (3-1)^2 = 4
    assert torch.isclose(si.penalty(lin), torch.tensor(4.0))


def test_si_path_integral_uses_per_step_deltas():
    lin = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        lin.weight.fill_(0.0)
    si = SI()
    si.begin_task(lin)
    si.before_step(lin)
    lin.weight.grad = torch.tensor([[1.0]])
    with torch.no_grad():
        lin.weight.fill_(2.0)
    si.after_step(lin)
    si.before_step(lin)
    lin.weight.grad = torch.tensor([[1.0]])
    with torch.no_grad():
        lin.weight.fill_(3.0)
    si.after_step(lin)
    assert torch.isclose(si._w["weight"], torch.tensor([[-3.0]]))


def test_ewc_fisher_estimation_runs():
    model = torch.nn.Linear(4, 2)
    cfg = OmegaConf.create(
        {"method": "ewc", "ewc_lambda": 1.0, "si_c": 0.1, "si_xi": 1e-3, "fisher_samples": 3}
    )
    con = Consolidation(cfg)
    batches = [(torch.randn(5, 4), torch.randint(0, 2, (5,))) for _ in range(3)]

    def loss_fn(b):
        return torch.nn.functional.cross_entropy(model(b[0]), b[1])

    con.consolidate(model, batches, loss_fn)
    assert con.penalty(model) >= 0  # zero at star, grows as weights move


def _plast(schedule):
    cfg = OmegaConf.create({"schedule": schedule, "lr": 1e-3, "rigidity": 1.0, "pnn_fraction": 0.3})
    return PlasticityController(cfg)


def test_soft_schedule_decays_to_positive_floor():
    p = _plast("soft")
    assert p.lr_scale(0.0) > 0.9
    assert p.lr_scale(1.0) >= p.floor and p.lr_scale(1.0) < 0.3
    assert p.lr_scale(0.0) > p.lr_scale(1.0)  # monotone-ish decay


def test_hard_schedule_step_drop():
    p = _plast("hard")
    assert p.lr_scale(0.1) == 1.0
    assert p.lr_scale(0.9) == p.floor


def test_triggered_reopening_raises_scale():
    p = _plast("hard")
    closed = p.lr_scale(0.9, signal=0.0)
    reopened = p.lr_scale(0.9, signal=3.0)
    assert reopened > closed


def test_pnn_rigidity_grows_with_stability():
    p = _plast("soft")
    model = torch.nn.Linear(4, 4)
    p.init_pnn(model)
    for _ in range(5):
        p.update_rigidity(model)  # no movement -> stability ~1 -> rigidity grows
    rig = next(iter(p._rigidity.values()))
    assert rig.mean() > 0.1


def test_pnn_freeze_zeros_masked_grads():
    p = _plast("soft")
    model = torch.nn.Linear(4, 4)
    p.init_pnn(model)
    out = model(torch.randn(3, 4)).sum()
    out.backward()
    p.apply_pnn_freeze(model)
    mask = p._pnn_mask["weight"]
    assert (model.weight.grad[mask] == 0).all()


def _neuro():
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "surprise_gain": 1.0,
            "novelty_gain": 1.0,
            "uncertainty_gain": 1.0,
            "gate_floor": 0.1,
            "gate_ceil": 3.0,
        }
    )
    return Neuromodulation(cfg)


def test_neuromod_surprise_is_prediction_error():
    nm = _neuro()
    s = nm.signals(torch.zeros(4, 8), torch.ones(4, 8))
    assert abs(s["surprise"] - 1.0) < 1e-5


def test_neuromod_gate_bounded_and_monotone():
    nm = _neuro()
    for v in [0.5, 0.6, 0.4, 0.55]:
        nm.gate("surprise", v)
    lo = nm.gate("surprise", 0.1)
    hi = nm.gate("surprise", 5.0)
    assert nm.floor <= lo <= nm.ceil and nm.floor <= hi <= nm.ceil
    assert hi > lo


def test_neuromod_disabled_returns_unity():
    cfg = OmegaConf.create(
        {
            "enabled": False,
            "surprise_gain": 1.0,
            "novelty_gain": 1.0,
            "uncertainty_gain": 1.0,
            "gate_floor": 0.1,
            "gate_ceil": 3.0,
        }
    )
    assert Neuromodulation(cfg).gate("surprise", 99.0) == 1.0


def test_build_modulation_toggles():
    cfg = OmegaConf.create(
        {"context_gating": True, "working_memory": True, "chunking": True, "n_contexts": 4, "wm_slots": 3}
    )
    mods = build_modulation(cfg, dim=8)
    assert set(mods) == {"context_gating", "working_memory", "chunking"}
    h = torch.randn(5, 8)
    gated = mods["context_gating"](h, torch.randint(0, 4, (5,)))
    assert gated.shape == (5, 8)
    read, mem = mods["working_memory"](h)
    assert read.shape == (5, 8) and mem.shape == (5, 3, 8)


def test_chunking_detects_boundaries():
    from mop.shell import Chunking

    seq = torch.zeros(1, 5, 4)
    seq[0, 3] = 10.0  # a jump at t=3
    b = Chunking(threshold=1.0).boundaries(seq)
    assert b.shape == (1, 4) and b[0, 2] == 1.0
