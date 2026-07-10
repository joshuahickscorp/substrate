# Local ceiling residual audit, 2026-07-10

## Decision

No current receipt proves a ViT-H boundary or any experiment-specific M3 Pro hardware
boundary. The active registry contains zero non-F rows with `studio-scale`, `moonshot`, or
`gpu-later`. The current local profile is 180 minutes, one heavy model at a time, with a 40 GB
free-disk floor.

ViT-H must remain in exact model names, configuration, and provenance. What is retired is the
claim that its name implies unavailable or remote-only work.

This audit is read-only with respect to canonical result generation. It does not reinterpret a
programmatic pilot as natural-video evidence and does not edit the custom-substrate requirements
ledger while CM7 is active.

## Exact evidence boundary

| Evidence | Current fact | SHA-256 |
|---|---|---|
| `proof/ENCODER_SCALE_VITH_LOAD.json` | strict offline load passed | `5f210271b20333242d1951a5151b286ed7d7f7c2e89c3a45bc02ccbefe4fb1a4` |
| `proof/ENCODER_SCALE_VITH_CPU_FORWARD.json` | supervised local CPU forward passed in 42.164 s outer wall, finite output, 3,775,889,408 peak process-tree RSS bytes, no hardware limit | `ee7df520b99ca1344d2281c59665ab4815ef4bf3c6ee9d51e42a73a73f795721` |
| `proof/REAL_ENCODER_VITH_LOCAL8.json` | eight shared referents cached at 34.728 s/clip | `7eed24391552256c13a59bb969d31b40744e56c9dbb542de0256950378bdf2f5` |
| `proof/ENCODER_SCALE_VITG_CPU_FORWARD.json` | largest published scale also passed locally in 199.193 s outer wall, 5,528,895,488 peak RSS bytes | `6e6a531370b997c600f20634e6d432abbaab703825d174200e2a6591d3e7935d` |
| `proof/REAL_ENCODER_VITG_LOCAL8.json` | eight shared referents cached serially | `872cc520f3fbdc5fc8d07b778581706a3fa2e3da3ba817cb1c0844c2796eb6a4` |
| `proof/STUDIO_READINESS_CURRENT_HOST.json` | 15/15 checks pass; M3 Pro, 19.3 GB unified memory, 42.135 GB free; 180-minute profile; no measured hardware limit | `a3b0708d2fa73af47a5e6a7879f4e3ae48e35c197ab10bbcb7a74c3c2183783f` |
| `proof/PROJECT_EXPERIMENT_EXHAUSTION.json` | 175 non-F rows accounted exactly once; measured-hardware-blocked count 0 | `73f43905a9efadd21eec207af038dffdc23d9b5aab0c067429d6386922ac502e` |
| `proof/FRONTIER_LOCALIZATION.json` | all 24 historical hardware-flavored non-F tags reclassified; measured-hardware-blocked count 0 | `95fa2d254fcf64908c2dd4bf7abd3516adad39509cc46c1eb1665f596a7dfc93` |
| `proof/VJEPA_SCALE_ATLAS_LOCAL.json` | local serial execution and model availability pass; scientific promotion false at n=8 | `21cc5a60983b731dd683a65e76a19f1ddfc556a9ec568208bca2eec55125014e` |

The scale atlas is mechanics and availability evidence only. Its own promotion gate fails because
the inputs are programmatic, n=8, matched random-architecture caches are incomplete, and
byte-identical learned/control stimuli are not proven.

## Registry and label audit

- Current non-F registry rows with `resource_tier: studio-scale`: 0.
- Current non-F registry rows with `resource_tier: moonshot`: 0.
- Current non-F registry rows with `exp_tier: gpu-later`: 0.
- `gpu-later`, `studio-scale`, and `moonshot` remain in schema enums, historical discussion, and
  negative boundary validators. Those references do not schedule work remotely.
- The campaign's ViT-g resource row was residual stale metadata. It is now `cpu-now`, tier C,
  explicitly serial, because load, forward, and cache receipts already support that statement.
- Full-preset 32 GB guards remain in the legacy PR9, atlas, and DR1 scripts. They are conservative
  policy guards, not measurements. Their bounded/smoke paths are local; any future larger rung needs
  a measured projection before it is called a hardware boundary.

## Every remaining scientific input blocker

The canonical non-F ledger currently groups 12 rows as rights/data and 10 as upstream-model. The
second bucket is too broad. The causal audit below separates upstream integration from locally
generatable caches, controls, and missing implementations. None is hardware-blocked. A same-day
primary-source recheck after this audit confirmed that the V-JEPA 2.1 dense checkpoints were
officially released on 2026-03-16, retiring the one apparent upstream-availability blocker
([official repository](https://github.com/facebookresearch/vjepa2),
[primary paper](https://arxiv.org/abs/2603.14482)).

| ID | Honest first blocker | Category | Local use or demotion |
|---|---|---|---|
| `e5_curiosity` | none for the registered fixed-pool result or bounded trajectory mechanics | local | persistent learnable/noisy rollouts now execute with exact replay; external ecologies are validation only |
| `e10_openended` | population search, environment generation, sustained horizon, transfer | implementation plus validation | persistent action mechanics exist; build the bounded ecology locally before any scaling premise |
| `mop_dr1_video_cache` | rights-clean natural clips with genuinely bound attributes and citable annotations | data/rights | bounded serial encoding is local once inputs pass |
| `mop_dr3_latent_scratchpad` | DR1 dense tokens and a working-memory-heavy task | data/task | shell mechanics can remain local |
| `mop_dr4_causal_intervention` | true factor-annotated counterfactual clip pairs | data/rights | local analysis after intake |
| `mop_dr7_latent_cot` | multi-step relational task with supervised intermediate states | data/task | local shell execution after intake |
| `mop_al3_audio_video_alignment` | rights-clean temporally aligned audio-video and citable caches | data/rights | small local audio/video encoders are admissible |
| `mop_cm1_compositional_gate` | rights-clean bound-attribute natural video | data/rights | local serial cache and probe |
| `mop_cm3_dense_vs_pooled` | dense natural-video tokens on the held-out-combination task | data/cache | bounded custom dense tokens can inform mechanics, not replace natural evidence |
| `mop_cm8_custom_jepa_pilot` | rights-cleared natural trajectories plus upstream gate receipts | data/rights | preflight and CM7 training are already local |
| `mop_cm9_slot_jepa_binding` | multi-object dense-token video and binding annotations | data/rights | local bounded slot pilot after intake |
| `mop_cm10_action_forward_model` | rendered action observations, citable substrate cache, exact V-JEPA 2-AC and depth controls | data/model controls | adapter and vector-mechanics pilot now run locally; the pilot currently loses to reactive |
| `e6_relational` | integrate and locally verify an official V-JEPA 2.1 dense checkpoint | upstream integration, not availability | official ViT-B/L/g/G checkpoints exist; probe the 80M ViT-B first and keep custom dense tokens separate |
| `mop_dr15_modality_general` | citable video, language/relational, and audio families on compatible tasks | mixed upstream model and data | stage small local families, then run serially |
| `mop_dr2_sparse_real` | full citable real-latent stream | data/cache, not upstream model | demote from model bucket when the ledger is refreshed |
| `mop_dr5_cross_substrate_consistency` | verified reasoning task, expanded shared rows, matched random-architecture controls | data/control | three learned scales already run locally; controls are a local queue item |
| `mop_dr14_corruption` | citable dense-token cache | data/cache | custom dense cache can supply a non-promotable pilot locally |
| `mop_at1_nuisance_grid` | matched random-init columns and complete citable shared content | local control plus data | immediate local demotion: run random controls serially after CM7 |
| `mop_al2_shared_latent_alignment` | meaningful shared content and preregistered equal-rank/matched-random controls | data/control | scale availability is retired; control work is local |
| `mop_cm2_atlas_gate` | CM1 result plus complete controlled atlas | dependency/data, not upstream model | remains local once CM1 and controls exist |
| `mop_cm4_workspace_shell` | full citable real-cache grid | data/cache, not upstream model | shell pilot already local |
| `mop_cm6_distilled_density` | executable distillation runner, citable teacher cache, matched student checkpoints | implementation plus data | custom workbench/checkpoints can inform a local runner |
| `f8_plastic_substrate_rewrite` | trusted rights-clean natural source, provenance authority, and real-weight source receipt | data/rights/provenance | current source and registry correctly say `env-later`, not hardware |
| `f16_perfect_slate_null` | same trusted natural source and provenance chain, plus its preregistered real-weight comparison | data/rights/provenance | current source and registry correctly say `env-later`, not hardware |

`mop_cm7_min_objective_probe` is not a nonlocal blocker. The current ledger records it as the one
`runnable-not-yet-run` row; it is an active local training job and must be refreshed only after its
own durable receipt completes.

## Generated-proof drift to refresh after CM7

Do not hand-edit these generated artifacts:

1. `proof/FORM_SUBSTRATE/SCORECARD.json` still stores `gpu-later` for F8 and F16 even though their
   configs, class metadata, registry rows, and campaign entries now say `env-later` and
   `environment-needed`.
2. `proof/FORM_SUBSTRATE/PRE_STUDIO_BOUNDARY.json` still classifies F8 and F16 as
   `studio-scale-claim-unproved`. Their first blocker is provenance/data, so the campaign
   preflight, collect, verifier, gate, scorecard, boundary, and bundle chain must be regenerated.
3. `proof/FRONTIER_LOCALIZATION.json` embeds the older scale-atlas SHA-256
   `5e8fb0fd4c6faa9911f6187633706e9a800544c441c4f83c29e1cb3f8c37151e`; the current atlas hash is
   `21cc5a60983b731dd683a65e76a19f1ddfc556a9ec568208bca2eec55125014e` after control hardening.
4. `proof/PROJECT_EXPERIMENT_EXHAUSTION.json` must be refreshed after the actual CM7 harness run.
   On refresh, narrow its broad upstream-model bucket using the causal classifications above.

## Local ceiling order after the active training lane clears

1. Validate and register the completed CM7 harness receipt, then refresh project exhaustion.
2. Run the ViT-L matched random-init cache first, verify its exact state and input hashes, then run
   ViT-H and ViT-g serially only while the 40 GB free-disk floor remains green.
3. Refresh the scale atlas and frontier localization with those controls. Keep the result
   non-promotable until natural shared content and byte-identical learned/control inputs exist.
4. Complete the rights-clean natural-video intake and map its attributes to DR1/CM1 requirements.
5. Regenerate the F campaign chain and all dependent boundary/bundle proofs.

That sequence consumes every currently local action before asking for a larger machine. The
remaining external dependencies are rights/task data, interactive environments, and compatible
audio/language inputs. E6 is now a local checkpoint-integration task, not an unpublished-model wait.
