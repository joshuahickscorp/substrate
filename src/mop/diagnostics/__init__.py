"""Diagnostics that gate the experiments: linear-probe distinctiveness, noisy-TV
epistemic/aleatoric guard, calibration, Fisher trace (critical-period signature), and the
determinism sanity loop."""

from __future__ import annotations

from .alignment import alignment_null, alignment_suite, cross_seed_alignment, permutation_pvalue
from .bottleneck import capability_per_bit, quantization_robustness
from .buffer_compression import retention_per_byte
from .calibration import calibration_plot, reliability
from .compute import (
    accounting,
    attention_flops,
    knn_flops,
    matched_within,
    mlp_flops,
    param_count,
    refiner_flops,
)
from .continual_metrics import (
    LRIntegralAccumulator,
    adaptation_speed,
    backward_transfer,
    forgetting_area,
    forward_transfer,
)
from .convergence import basin_stability, convergence_report
from .cross_substrate import cross_substrate_agreement, random_map_floor, shuffled_label_null
from .determinism import assert_reproducible, determinism_loop
from .difficulty_calibration import calibrated_tie, reference_separation
from .fisher_trace import critical_period_signature, fisher_trace, fisher_trace_over_training
from .geometry import (
    anisotropy,
    effective_rank,
    geometry_report,
    kernel_cka,
    linear_cka,
    neighborhood_overlap,
    rsa,
)
from .held_out_combo import compositionality_report, factorized_latents, held_out_combination
from .latent_robustness import degradation_curve
from .linear_probe import linear_probe
from .noisy_tv import noisy_tv_diagnostic
from .nonlinear_probe import nonlinear_probe, readout_contribution
from .riskcov import (
    auroc,
    ece_equal_mass,
    pareto_area,
    pareto_frontier,
    risk_coverage,
    seed_ci,
    sign_flip_report,
)
from .seed_consistency import code_stability, cross_seed_cka, hungarian_code_agreement
from .substrate_ablation import substrate_ablation
from .sysid import controllability_gramian_rank, sysid_report
from .transfer_matrix import transfer_matrix

__all__ = [
    "linear_probe",
    "noisy_tv_diagnostic",
    "reliability",
    "calibration_plot",
    "fisher_trace",
    "fisher_trace_over_training",
    "critical_period_signature",
    "determinism_loop",
    "assert_reproducible",
    # geometry battery (D1 / EX12)
    "linear_cka",
    "kernel_cka",
    "rsa",
    "effective_rank",
    "anisotropy",
    "neighborhood_overlap",
    "geometry_report",
    # compute accounting (D5)
    "param_count",
    "mlp_flops",
    "refiner_flops",
    "matched_within",
    "accounting",
    # substrate-ablation control (D2)
    "substrate_ablation",
    # convergence / attractor (Y1/Y2/N9)
    "convergence_report",
    "basin_stability",
    # nonlinear probe + readout-contribution (P10/P1)
    "nonlinear_probe",
    "readout_contribution",
    # held-out-combination compositionality (C1/C9/S6)
    "factorized_latents",
    "held_out_combination",
    "compositionality_report",
    # seed-consistency (Y3/P5/S5)
    "cross_seed_cka",
    "hungarian_code_agreement",
    "code_stability",
    # capability-per-bit (I1/I8)
    "capability_per_bit",
    "quantization_robustness",
    # system identification / controllability (Y7 / D6 rollout gate)
    "sysid_report",
    "controllability_gramian_rank",
    # difficulty calibration (D3)
    "reference_separation",
    "calibrated_tie",
    # transfer matrix (D4)
    "transfer_matrix",
    # replay-buffer compression (A3)
    "retention_per_byte",
    # latent robustness (A4)
    "degradation_curve",
    # alignment aggregator (WP-02, AL rows)
    "permutation_pvalue",
    "alignment_null",
    "alignment_suite",
    "cross_seed_alignment",
    # cross-substrate agreement (WP-02, AT1/AL2/WS1)
    "cross_substrate_agreement",
    "random_map_floor",
    "shuffled_label_null",
    # risk-coverage / selective prediction (WP-02, H-RISKCOV)
    "auroc",
    "ece_equal_mass",
    "risk_coverage",
    "pareto_frontier",
    "pareto_area",
    "seed_ci",
    "sign_flip_report",
    # continual-learning metrics (WP-02, PR rows)
    "backward_transfer",
    "forward_transfer",
    "forgetting_area",
    "adaptation_speed",
    "LRIntegralAccumulator",
    # attention / retrieval FLOP counters (WP-02)
    "attention_flops",
    "knn_flops",
]
