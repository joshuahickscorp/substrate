#!/bin/bash
# MoT maximization queue, Stages 1-3 (EXECUTION_MANIFEST.md section 4).
# Strictly sequential: exactly one heavy OR one encode job at any moment.
# Resumable: a row is skipped when its output JSON already exists.
# Failure policy: log and continue (a failed row must not kill the queue).
cd /Users/scammermike/Downloads/brain
export PYTHONPATH=/Users/scammermike/Downloads/brain
PY=.venv/bin/python
LOGDIR=runs/mot/logs
PROG=runs/mot/queue_progress.log
mkdir -p "$LOGDIR" runs/mot

row() { # row <id> <out_json> <cmd...>
  local id="$1"; local out="$2"; shift 2
  if [ -f "$out" ]; then echo "$(date +%m-%d\ %H:%M) $id SKIP (exists)" >> "$PROG"; return 0; fi
  echo "$(date +%m-%d\ %H:%M) $id START" >> "$PROG"
  if "$@" > "$LOGDIR/$id.log" 2>&1; then
    echo "$(date +%m-%d\ %H:%M) $id DONE" >> "$PROG"
  else
    echo "$(date +%m-%d\ %H:%M) $id FAIL (see $LOGDIR/$id.log)" >> "$PROG"
  fi
}

echo "$(date +%m-%d\ %H:%M) ==== STAGE 1 ====" >> "$PROG"
row Q1.0a runs/mot/at4_programmatic_features.json $PY scripts/featurize_programmatic.py
row Q1.0b runs/mot/at4_handcrafted_features.json  $PY scripts/featurize_handcrafted.py
row Q1.1  runs/mot/at5_probe_class_sweep.json     $PY scripts/mot_at5_probe_class_sweep.py --seeds 0-4
row Q1.2  runs/mot/at4_programmatic_ceiling.json  $PY scripts/mot_at4_programmatic_ceiling.py --seeds 0-4
row Q1.3  runs/mot/mt5_adaptive_halting.json      $PY scripts/mot_mt5_adaptive_halting.py --seeds 0-4
row Q1.4  runs/mot/mt6_confidence_stop.json       $PY scripts/mot_mt6_confidence_stop.py --seeds 0-4
row Q1.5  runs/mot/dr9_verify_revise.json         $PY scripts/mot_dr9_verify_revise.py --seeds 0-4
row Q1.6  runs/mot/dr8_fixed_point_vjepa.json     $PY scripts/mot_dr8_fixed_point.py --cache vjepa --seeds 0-4
row Q1.7  runs/mot/mt7_beam_search.json           $PY scripts/mot_mt7_beam_search.py --seeds 0-4
row Q1.8  runs/mot/dr6_rollout_planning.json      $PY scripts/mot_dr6_rollout_planning.py --seeds 0-4
row Q1.9  runs/mot/dr13_horizon_limit.json        $PY scripts/mot_dr13_horizon_limit.py --seeds 0-4
row Q1.10 runs/mot/dr11_mc_rollouts.json          $PY scripts/mot_dr11_mc_rollouts.py --seeds 0-4
row Q1.11 runs/mot/dr10_retrieve_reason.json      $PY scripts/mot_dr10_retrieve_reason.py --seeds 0-4
row Q1.12 runs/mot/pr8_retrieval_head.json        $PY scripts/mot_pr8_retrieval_head.py --seeds 0-4
row Q1.13 runs/mot/pr7_fast_slow.json             $PY scripts/mot_pr7_fast_slow.py --seeds 0-4
row Q1.14 runs/mot/pr4_epistemic_gate.json        $PY scripts/mot_pr4_epistemic_gate.py --seeds 0-4
row Q1.15 runs/mot/pr5_content_gated_cp.json      $PY scripts/mot_pr5_content_gated_cp.py --seeds 0-4
row Q1.16 runs/mot/pr6_sleep_consolidation.json   $PY scripts/mot_pr6_sleep_consolidation.py --seeds 0-4
row Q1.17 runs/mot/al1_uncertainty_router.json    $PY scripts/mot_al1_uncertainty_router.py --seeds 0-4
row Q1.18 runs/mot/dr12_disagreement.json         $PY scripts/mot_dr12_disagreement.py --seeds 0-4
row Q1.19 runs/mot/mt8_latent_debate.json         $PY scripts/mot_mt8_latent_debate.py --seeds 0-4
row Q1.20 runs/mot/mt123_router_pilots.json       $PY scripts/mot_mt123_router_pilots.py --seeds 0-4
row Q1.21 runs/mot/dr14_corruption.json           $PY scripts/mot_dr14_corruption.py --seeds 0-2
row Q1.22 runs/mot/dr2_sparse_real_pilot.json     $PY scripts/mot_dr2_sparse_real_pilot.py --seeds 0-4
row Q1.23 runs/mot/ws5_slot_ablation_pilot.json   $PY scripts/mot_ws5_slot_ablation_pilot.py --seeds 0-4
row Q1.24 runs/mot/cm4_workspace_pilot.json       $PY scripts/mot_cm4_workspace_pilot.py --seeds 0-2

echo "$(date +%m-%d\ %H:%M) ==== STAGE 2 (encoder lane, serial) ====" >> "$PROG"
row Q2.1 runs/mot/cache_randominit_vitl.json $PY scripts/cache_randominit_vitl_features.py
row Q2.2 runs/mot/cache_singleframe.json     $PY scripts/cache_vjepa_single_frame.py
row Q2.3 runs/mot/cache_dinov2s.json         $PY scripts/cache_dinov2s_nuisance.py
row Q2.4 runs/mot/cache_qwen.json            $PY scripts/cache_qwen_textified.py
row Q2.5 runs/mot/cache_wav2vec2.json        $PY scripts/cache_wav2vec2_sonified.py

echo "$(date +%m-%d\ %H:%M) ==== STAGE 3 ====" >> "$PROG"
row Q3.1 runs/mot/pr2_plasticity_substrates.json    $PY scripts/mot_pr2_plasticity_substrates.py --seeds 0-4
row Q3.2 runs/mot/at3_time_axis.json                $PY scripts/mot_at3_time_axis.py --seeds 0-4
row Q3.3 runs/mot/dr8_fixed_point_randominit.json   $PY scripts/mot_dr8_fixed_point.py --cache randominit_vitl --seeds 0-4
row Q3.4 runs/mot/ws1_agreement_vs_confidence.json  $PY scripts/mot_ws1_agreement_vs_confidence.py --seeds 0-4
row Q3.5 runs/mot/ws2_fusion_tournament.json        $PY scripts/mot_ws2_fusion_tournament.py --seeds 0-4
row Q3.6 runs/mot/ws4_bandwidth_sweep.json          $PY scripts/mot_ws4_bandwidth_sweep.py --seeds 0-4
row Q3.7 runs/mot/ws3_arbitration.json              $PY scripts/mot_ws3_arbitration.py --seeds 0-4
row Q3.8 runs/mot/at1_grid_pilot.json               $PY scripts/mot_at1_grid_pilot.py --seeds 0-4
row Q3.9 runs/mot/al2_alignment_pilot.json          $PY scripts/mot_al2_alignment_pilot.py --seeds 0-4

echo "$(date +%m-%d\ %H:%M) ==== STAGES 1-3 COMPLETE ====" >> "$PROG"
