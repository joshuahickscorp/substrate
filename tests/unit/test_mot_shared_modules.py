"""WP-02 shared-module tests: tiny synthetic tensors only, no network, no weights, no encoder loads
(the RandomInitViT build path is asserted deferred, not exercised). Covers adapter (pixel control +
registry), alignment, cross_substrate, riskcov, continual_metrics, workspace, capmatch, plus the
pass-2 existing-file extensions (kWTA/MoE heads, attention/kNN FLOP counters, package re-exports)."""

import math

import pytest
import torch
from torch import nn

from mop.diagnostics import continual_metrics as cm
from mop.diagnostics import riskcov as rc
from mop.diagnostics.alignment import alignment_suite, alignment_table, permutation_pvalue
from mop.diagnostics.compute import attention_flops, knn_flops, linear_flops, param_count
from mop.diagnostics.cross_substrate import cross_substrate_agreement
from mop.shell.capmatch import fixed_total_params_sweep, matched_capacity, width_for_param_count
from mop.shell.heads import KWTAHead, MoEHead, moe_expert_hidden_for_dense, routing_entropy
from mop.shell.workspace import WorkspaceShell
from mop.substrate.adapter import RandomInitViTAdapter, RandomPixelAdapter, SubstrateRegistry

# ---------------------------------------------------------------- adapter


def test_random_pixel_adapter_shape_and_determinism():
    a = RandomPixelAdapter(embed_dim=16, ds=4, tsub=2, seed=0)
    clips = torch.rand(3, 4, 3, 8, 8)
    z1, z2 = a.extract(clips), a.extract(clips)
    assert z1.shape == (3, 16)
    assert torch.allclose(z1, z2)
    assert a.meta.pretrained is False and a.meta.input_resolution == 4


def test_random_pixel_adapter_norm_convention():
    a = RandomPixelAdapter(embed_dim=16, ds=4, tsub=2, seed=1)
    z = a.extract(torch.rand(2, 4, 3, 8, 8))
    assert torch.allclose(z.norm(dim=-1), torch.full((2,), math.sqrt(16.0)), atol=1e-3)


def test_registry_rejects_duplicate_tags():
    reg = SubstrateRegistry()
    reg.register(RandomPixelAdapter(embed_dim=8, ds=4, tsub=2))
    with pytest.raises(ValueError):
        reg.register(RandomPixelAdapter(embed_dim=8, ds=4, tsub=2))


def test_random_init_vit_build_is_deferred_until_first_extract():
    """Constructing the adapter must never load a model (live-encode constraint); the from_config
    build happens lazily on first extract and is cached. Exercised via a stubbed _build so the test
    stays weight-free."""

    class Cfg:
        name, embed_dim, hf_id = "toy", 8, "toy/never-fetched"

    a = RandomInitViTAdapter(Cfg(), input_resolution=32, frames=4)
    assert a._encoder is None  # nothing built at construction
    assert a.tag == "toy_randinit" and a.meta.pretrained is False

    builds = []

    class StubEnc:
        def encode(self, clips):
            builds.append(1)
            return torch.zeros(clips.shape[0], 8)

    a._build = lambda: StubEnc()  # type: ignore[method-assign]
    out = a.extract(torch.rand(2, 4, 3, 8, 8))
    assert out.shape == (2, 8)
    a.extract(torch.rand(1, 4, 3, 8, 8))
    assert isinstance(a._encoder, StubEnc)  # built once, then cached


# ---------------------------------------------------------------- alignment


def test_permutation_pvalue_add_one():
    assert permutation_pvalue(1.0, [0.1, 0.2, 0.3]) == 1 / 4
    assert permutation_pvalue(0.0, [0.1, 0.2, 0.3]) == 1.0


def test_alignment_suite_self_alignment():
    x = torch.randn(24, 6, generator=torch.Generator().manual_seed(0))
    rep = alignment_suite(x, x, n_permutations=20)
    assert rep["linear_cka"] > 0.99 and rep["p_value"] < 0.1


def test_alignment_suite_mapping_mode_builds_pair_matrices():
    g = torch.Generator().manual_seed(0)
    base = torch.randn(24, 6, generator=g)
    reps = {
        "vision_vjepa2": base,
        "caption_text": base[:, :3] + 0.01 * torch.randn(24, 3, generator=g),
        "randinit_control": torch.randn(24, 6, generator=g),
    }
    table = alignment_suite(reps, n_permutations=20, seed=1, k=3)
    assert table["schema"] == "mop-alignment-suite/v1"
    assert table["tags"] == ["caption_text", "randinit_control", "vision_vjepa2"]
    assert table["matrices"]["linear_cka"]["vision_vjepa2"]["caption_text"] > 0.5
    assert "caption_text__vision_vjepa2" in table["pairs"]
    assert table["warnings"] == []


def test_alignment_table_refuses_mismatched_referent_counts():
    with pytest.raises(ValueError, match="share N referents"):
        alignment_table({"a": torch.randn(4, 2), "b": torch.randn(5, 2)})


def test_alignment_table_warns_without_random_encoder_control():
    table = alignment_table(
        {"a": torch.randn(6, 2), "random_projection": torch.randn(6, 2)}, n_permutations=3
    )
    assert any("random projection" in w for w in table["warnings"])
    assert any("no random-encoder" in w for w in table["warnings"])


# ---------------------------------------------------------------- cross_substrate


def test_cross_substrate_agreement_keys_and_symmetry():
    g = torch.Generator().manual_seed(0)
    y = torch.arange(30) % 2
    base = torch.randn(30, 8, generator=g) + y[:, None].float() * 3
    lat = {"a": base, "b": base + 0.01 * torch.randn(30, 8, generator=g)}
    rep = cross_substrate_agreement(lat, y, probe_epochs=30)
    assert rep["tags"] == ["a", "b"]
    assert rep["cka"]["a"]["b"] == rep["cka"]["b"]["a"]
    assert rep["probe_acc"]["a"] > rep["shuffled_null"]["a"]


# ---------------------------------------------------------------- riskcov


def test_auroc_perfect_and_chance():
    assert rc.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert rc.auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5


def test_ece_equal_mass_zero_when_calibrated():
    assert rc.ece_equal_mass([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0], bins=2) == 0.0


def test_risk_coverage_monotone_setup():
    out = rc.risk_coverage([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert out["risk"][0] == 0.0 and out["risk_at_full"] == 0.5
    assert out["aurc"] < 0.5


def test_pareto_area_dominated_point_ignored():
    a = rc.pareto_area([(1.0, 0.5), (2.0, 0.4), (3.0, 0.9)])
    b = rc.pareto_area([(1.0, 0.5), (3.0, 0.9)])
    assert a == b


def test_seed_ci_and_sign_flip():
    ci = rc.seed_ci([0.1, 0.2, 0.3])
    assert ci["n"] == 3 and abs(ci["mean"] - 0.2) < 1e-9 and not ci["unstable"]
    flip = rc.sign_flip_report([0.1, -0.05, 0.2])
    assert flip["any_flip"] and flip["consistent_sign"] == 0
    assert rc.sign_flip_report([0.1, 0.2])["consistent_sign"] == 1


# ---------------------------------------------------------------- continual_metrics


def test_backward_transfer_matches_two_task_delta():
    m = [[0.9, 0.1], [0.7, 0.8]]
    assert abs(cm.backward_transfer(m) - (0.7 - 0.9)) < 1e-9


def test_forward_transfer():
    m = [[0.9, 0.6], [0.7, 0.8]]
    assert abs(cm.forward_transfer(m, [0.5, 0.5]) - 0.1) < 1e-9


def test_forgetting_area_zero_when_monotone():
    assert cm.forgetting_area([0.1, 0.5, 0.9]) == 0.0
    assert cm.forgetting_area([0.9, 0.4, 0.9]) > 0.0


def test_adaptation_speed():
    out = cm.adaptation_speed([0.0, 0.5, 0.95, 1.0], target_frac=0.9)
    assert out["reached"] and out["steps"] == 2


def test_lr_integral_accumulator_partitions_and_match():
    a, b = cm.LRIntegralAccumulator(), cm.LRIntegralAccumulator()
    a.add(1e-3, steps=10, partition="reducible")
    a.add(1e-3, steps=5, partition="irreducible")
    b.add(1e-3, steps=15)
    assert abs(a.total() - 0.015) < 1e-12 and a.total("reducible") == 0.01
    assert a.matched(b)


# ---------------------------------------------------------------- workspace


def test_workspace_shell_minimal_forward():
    shell = WorkspaceShell(8, head=nn.Linear(8, 3))
    out = shell(torch.randn(4, 8))
    assert out["latent"].shape == (4, 8) and out["head"].shape == (4, 3)
    assert "prediction" not in out


# ---------------------------------------------------------------- heads (WP-02 pass 2, e7 family)


def test_kwta_head_param_matches_dense_and_sparsifies():
    dense = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 3))
    kwta = KWTAHead(8, hidden=16, n_classes=3, k=2)
    assert param_count(kwta) == param_count(dense)  # activation sparsity, not bought capacity
    out = kwta(torch.randn(5, 8))
    assert out.shape == (5, 3)
    h = torch.nn.functional.gelu(kwta.fc1(torch.randn(5, 8)))
    thresh = h.topk(kwta.k, dim=-1).values[..., -1:]
    assert int((h * (h >= thresh) != 0).sum(-1).max()) <= kwta.k


def test_moe_head_gates_and_routing_entropy():
    head = MoEHead(8, n_classes=3, n_experts=4, expert_hidden=5)
    out = head(torch.randn(6, 8))
    assert out.shape == (6, 3)
    assert head.last_gates is not None and head.last_gates.shape == (6, 4)
    ent = routing_entropy(head.last_gates)
    assert 0.0 <= ent <= math.log(4) + 1e-6
    uniform = torch.full((6, 4), 0.25)
    assert abs(routing_entropy(uniform) - math.log(4)) < 1e-5


def test_moe_expert_hidden_matches_dense_budget():
    dim, hidden, n_classes, n_experts = 32, 64, 6, 4
    w = moe_expert_hidden_for_dense(dim, hidden, n_classes, n_experts)
    dense = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, n_classes))
    moe = MoEHead(dim, n_classes, n_experts, w)
    ratio = param_count(moe) / param_count(dense)
    assert 0.95 <= ratio <= 1.05  # e7 matching rule, within capmatch-style tolerance


# ---------------------------------------------------------------- compute (WP-02 pass 2 counters)


def test_attention_flops_projection_plus_scores():
    n, d = 4, 8
    expected = 4 * linear_flops(d, d, batch=n) + 2 * (2 * n * n * d)
    assert attention_flops(n, d) == expected


def test_knn_flops_is_query_key_matmul():
    assert knn_flops(n_queries=3, n_keys=10, dim=8) == 2 * 3 * 10 * 8
    assert knn_flops(3, 10, 8, batch=2) == 2 * knn_flops(3, 10, 8)


def test_package_reexports_land():
    from mop.diagnostics import attention_flops as af
    from mop.diagnostics import auroc, backward_transfer, cross_substrate_agreement  # noqa: F401
    from mop.shell import KWTAHead as kh
    from mop.shell import MoEHead, WorkspaceShell, matched_capacity  # noqa: F401

    assert af is attention_flops and kh is KWTAHead


# ---------------------------------------------------------------- capmatch


def test_width_for_param_count_exact():
    w, p = width_for_param_count(lambda h: nn.Linear(10, h), target_params=110, tol=0.02)
    assert w == 10 and p == 110


def test_matched_capacity_within_tol():
    ref = nn.Linear(10, 10)

    def make(h):
        return nn.Sequential(nn.Linear(4, h), nn.Linear(h, 4))

    from mop.diagnostics.compute import param_count

    m = matched_capacity(ref, make, tol=0.05)
    assert abs(param_count(m) - param_count(ref)) <= 0.05 * param_count(ref)


def test_fixed_total_params_sweep_holds_total():
    def make(slots, width):
        return nn.ModuleList([nn.Linear(6, width) for _ in range(slots)])

    out = fixed_total_params_sweep(make, total_params=280, slots=[1, 2], tol=0.05)
    assert set(out) == {1, 2}
    for row in out.values():
        assert abs(row["params"] - 280) <= 0.05 * 280


def test_capmatch_raises_on_unreachable_target():
    with pytest.raises(ValueError):
        width_for_param_count(lambda h: nn.Linear(1000, h), target_params=10, tol=0.02)


# ---------------------------------------------------------------- aggregate report (WP-02 pass 2)


def test_aggregate_report_collects_verdicts_and_missing(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "mop_aggregate_report",
        Path(__file__).resolve().parents[2] / "scripts" / "mop_aggregate_report.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    win_block = {
        "per_seed": [0.1, 0.2],
        "ci": {"n": 2, "mean": 0.15, "sd": 0.05, "unstable": True},
        "sign_flips": {"any_flip": False},
        "win": True,
    }
    (tmp_path / "mt5_adaptive_halting.json").write_text(
        json.dumps({"experiment": "mop_mt5", "null_supported": False, "delta": win_block, "seconds": 1})
    )
    (tmp_path / "dr9_verify_revise.json").write_text(
        json.dumps({"experiment": "mop_dr9", "null_supported": True, "seconds": 2})
    )
    (tmp_path / "broken.json").write_text("{not json")
    # a not-evaluable row (missing input): even an old-style null_supported=False must land in the
    # not_evaluable bucket via the verdict prefix, never among the rejected nulls
    (tmp_path / "at4_programmatic_ceiling.json").write_text(
        json.dumps(
            {
                "experiment": "mop_at4",
                "verdict": "NO CEILING: the programmatic_reference store is not on disk",
                "null_supported": False,
                "seconds": 1,
            }
        )
    )

    rep = mod.aggregate(tmp_path)
    assert rep["n_jsons"] == 4
    assert rep["null_rejected_ids"] == ["mop_mt5"]
    assert rep["not_evaluable_ids"] == ["mop_at4"] and rep["n_not_evaluable"] == 1
    assert rep["unstable_ci_ids"] == ["mop_mt5"]
    by_id = {r["experiment"]: r for r in rep["rows"]}
    assert by_id["mop_mt5"]["n_wins"] == 1 and not by_id["mop_mt5"]["any_sign_flip"]
    assert "error" in by_id["broken"]
    assert "dr10_retrieve_reason" in rep["missing_expected"]
    assert "mt5_adaptive_halting" not in rep["missing_expected"]
    assert "mop_mt5" in mod.render_table(rep)
