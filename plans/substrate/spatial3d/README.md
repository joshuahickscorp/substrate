# Frontier G — native spatial3d backend

Primary 3D backend for Odyssey Frontier G.  Pure Python + stdlib z-buffer
renderer.  **Blender is optional comparison only and is never a launch blocker.**

## Identity

| Field | Value |
| --- | --- |
| Renderer | `substrate_spatial3d` |
| Version | `1.0.0` |
| Module | `src/substrate/spatial3d.py` |
| Registry ops | `three_d.build_scene`, `three_d.render`, `three_d.depth`, `three_d.move_object`, `three_d.set_camera`, `three_d.inspect_mesh` |

## Public seeds

Committed under `plans/substrate/spatial3d/seeds/`:

1. `canary_relative_position_v1`
2. `canary_support_containment_v1`
3. `canary_occlusion_v1`
4. `canary_viewpoint_change_v1`
5. `canary_object_permanence_v1`
6. `canary_depth_ordering_v1`
7. `canary_active_camera_v1`

Evaluator-only answer state lives under `seeds/evaluator/<seed_id>/` (path
token `evaluator` is refused in arm tool parameters).

## Candidate vs evaluator

| Candidate sensors | Evaluator-only |
| --- | --- |
| RGB PNG | Full scene graph |
| Depth u16 PNG (mm) | Answer state / ground truth |
| Depth vis PNG | Float32 work copy under sandbox `work/` |
| Local mesh stats (no world poses) | Support/containment relations |
| Object / camera ids | Object world transforms |

## Determinism

Same `seed_id` → byte-identical RGB and depth digests.  Timing is recorded on
tool receipts only, never inside admitted artifact payloads.

## Optional Blender

`three_d.render` with `backend=blender` runs Cycles CPU when Blender is on the
inventory.  Failure does not fall through to claim Blender as primary; the
default and honest primary path is always `substrate_spatial3d`.
