
from __future__ import annotations

from .alignment import (
    alignment_null,
    alignment_suite,
    alignment_table,
    cross_seed_alignment,
    pair_alignment,
    permutation_pvalue,
)
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
from .operational_awareness import (
    compute_value,
    confidence_calibration,
    crisis_detection,
    memory_availability,
    missing_form_detection,
    mode_selection,
    oa_suite,
    render_oa_md,
    report_grounding,
    rewrite_caution,
)
from .performance_density import density_block, timed
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
    "linear_cka",
    "kernel_cka",
    "rsa",
    "effective_rank",
    "anisotropy",
    "neighborhood_overlap",
    "geometry_report",
    "param_count",
    "mlp_flops",
    "refiner_flops",
    "matched_within",
    "accounting",
    "substrate_ablation",
    "convergence_report",
    "basin_stability",
    "nonlinear_probe",
    "readout_contribution",
    "factorized_latents",
    "held_out_combination",
    "compositionality_report",
    "cross_seed_cka",
    "hungarian_code_agreement",
    "code_stability",
    "capability_per_bit",
    "quantization_robustness",
    "sysid_report",
    "controllability_gramian_rank",
    "reference_separation",
    "calibrated_tie",
    "transfer_matrix",
    "retention_per_byte",
    "degradation_curve",
    "permutation_pvalue",
    "alignment_null",
    "alignment_suite",
    "alignment_table",
    "pair_alignment",
    "cross_seed_alignment",
    "cross_substrate_agreement",
    "random_map_floor",
    "shuffled_label_null",
    "auroc",
    "ece_equal_mass",
    "risk_coverage",
    "pareto_frontier",
    "pareto_area",
    "seed_ci",
    "sign_flip_report",
    "backward_transfer",
    "forward_transfer",
    "forgetting_area",
    "adaptation_speed",
    "LRIntegralAccumulator",
    "attention_flops",
    "knn_flops",
    "density_block",
    "timed",
    "missing_form_detection",
    "confidence_calibration",
    "memory_availability",
    "mode_selection",
    "compute_value",
    "crisis_detection",
    "rewrite_caution",
    "report_grounding",
    "oa_suite",
    "render_oa_md",
]
