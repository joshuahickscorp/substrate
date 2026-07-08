import torch

from mop.process_c import (
    DenseTokenMeanBaseline,
    ProcessCDenseTokenClassifier,
    binding_specificity,
    dense_hidden_for_target_params,
    param_count,
    process_c_budget_report,
)


def test_process_c_classifier_shapes_and_attention_normalization():
    model = ProcessCDenseTokenClassifier(8, 3, n_slots=4, slot_dim=12, iterations=2)
    out = model(torch.randn(5, 7, 8))
    assert out["slots"].shape == (5, 4, 12)
    assert out["pooled"].shape == (5, 12)
    assert out["logits"].shape == (5, 3)
    assert torch.allclose(out["attention"].sum(dim=-1), torch.ones(5, 4), atol=1e-5)
    assert 0.0 <= float(out["assignment_entropy"]) <= 1.0


def test_dense_token_mean_baseline_masks_padding():
    baseline = DenseTokenMeanBaseline(6, 2, hidden=5)
    tokens = torch.randn(3, 4, 6)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
    logits = baseline(tokens, mask=mask)
    assert logits.shape == (3, 2)


def test_dense_hidden_for_target_params_matches_slot_budget():
    slot = ProcessCDenseTokenClassifier(10, 4, n_slots=3, slot_dim=16)
    hidden = dense_hidden_for_target_params(10, 4, param_count(slot))
    baseline = DenseTokenMeanBaseline(10, 4, hidden)
    ratio = param_count(baseline) / param_count(slot)
    assert 0.8 <= ratio <= 1.2


def test_binding_specificity_reports_target_slot_motion():
    before = torch.zeros(2, 3, 4)
    after = before.clone()
    after[:, 1] = 2.0
    after[:, 0] = 0.5
    report = binding_specificity(before, after, target_slot=1)
    assert report["target_is_largest"] is True
    assert report["specificity_ratio"] > 1.0


def test_process_c_budget_report_blocks_unlicensed_and_bounds():
    model = ProcessCDenseTokenClassifier(8, 3, n_slots=2, slot_dim=8)
    blocked = process_c_budget_report(model, licensed=False, min_params=1, max_params=10_000)
    assert blocked["within_budget"] is False
    assert "not licensed" in blocked["problems"][0]

    ok = process_c_budget_report(model, licensed=True, min_params=1, max_params=10_000)
    assert ok["within_budget"] is True
