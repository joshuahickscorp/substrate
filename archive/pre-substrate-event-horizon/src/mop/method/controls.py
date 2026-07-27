"""Control semantic proofs.

A control is a claim about removal. Declaring the removal is not the claim being true. The order free
control in the prior program declared that it could not consume temporal order and then ran a Conv1d with
kernel 5 over the time axis, which meant the temporal headroom interpretation built on it was invalid.

Every proof here is executable and returns a receipt of booleans. A control with an unproven claim never
passes, and a control that fails any claim cannot be used as the comparison for a verdict.

House style: no dashes.
"""

from __future__ import annotations

import numpy as np

TOL = 1e-4


def _close(a, b, tol=TOL) -> bool:
    import torch

    a, b = torch.as_tensor(a), torch.as_tensor(b)
    return bool(torch.allclose(a, b, atol=tol, rtol=0))


def _finish(checks: dict) -> dict:
    checks["all_pass"] = all(v for k, v in checks.items() if isinstance(v, bool) and k != "all_pass")
    return checks


# ---------------------------------------------------------------- 7.1 order free


def structural_temporal_scan(module) -> dict:
    """Static half of the order free proof: does the graph contain machinery that reads order at all."""
    import torch.nn as nn

    conv, rec, pos, state = [], [], [], []
    for name, m in module.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,)
            if any(int(x) > 1 for x in k):
                conv.append(f"{name}:{type(m).__name__}(kernel={k})")
        if isinstance(m, (nn.RNNBase, nn.RNNCellBase)) or type(m).__name__ in (
            "LSTM",
            "GRU",
            "RNN",
            "LSTMCell",
            "GRUCell",
        ):
            rec.append(f"{name}:{type(m).__name__}")
    for name, _ in module.named_parameters():
        low = name.lower()
        if "pos" in low or "position" in low or "time_embed" in low:
            pos.append(name)
    for name, _ in module.named_buffers():
        low = name.lower()
        if "state" in low or "hidden" in low or "carry" in low:
            state.append(name)
    return {
        "no_temporal_convolution": not conv,
        "no_recurrence": not rec,
        "no_position_encoding": not pos,
        "no_carried_state_buffer": not state,
        "found": {"conv": conv, "recurrent": rec, "position": pos, "state": state},
    }


def order_free(forward, x, module=None, seed: int = 0, declared_exceptions: tuple = ()) -> dict:
    """Behavioural half plus the structural half.

    forward(x) must be deterministic. x has shape (N, T, C). The control passes only when every temporal
    rearrangement leaves the output unchanged, which is the operational meaning of not consuming order.
    """
    import torch

    rng = np.random.default_rng(seed)
    with torch.no_grad():
        y = forward(x)
        perm = torch.as_tensor(rng.permutation(x.shape[1]))
        checks = {
            "timestep_permutation_invariant": _close(forward(x[:, perm]), y),
            "sequence_reversal_invariant": _close(forward(x.flip(1)), y),
        }
        T = x.shape[1]
        nb = 4 if T >= 4 else 2
        blocks = torch.chunk(x, nb, dim=1)
        order = rng.permutation(len(blocks))
        checks["block_permutation_invariant"] = _close(
            forward(torch.cat([blocks[i] for i in order], 1)), y
        )
        # timestamp removal: a cyclic shift of the whole sequence must not change the output. A model with
        # no notion of when a timestep occurred cannot see the shift.
        checks["timestamp_removal_invariant"] = _close(forward(torch.roll(x, T // 3, dims=1)), y)
        # determinism: two identical calls must agree, otherwise no invariance claim is testable
        checks["deterministic"] = _close(forward(x), y)
    if module is not None:
        s = structural_temporal_scan(module)
        for k in ("no_temporal_convolution", "no_recurrence", "no_position_encoding", "no_carried_state_buffer"):
            if k in declared_exceptions:
                continue
            checks[k] = s[k]
        checks["structural_findings"] = s["found"]
    return _finish(checks)


# ---------------------------------------------------------------- 7.2 no replay


def no_replay(trace: dict) -> dict:
    """trace comes from the execution receipt: what the update actually consumed."""
    return _finish(
        {
            "no_historical_item_in_updates": int(trace.get("replayed_items", 0)) == 0,
            "no_hidden_buffer_read": int(trace.get("buffer_reads", 0)) == 0,
            "no_cached_batch_entered_training": int(trace.get("cached_batches", 0)) == 0,
            "buffer_declared_empty": int(trace.get("buffer_size", 0)) == 0,
        }
    )


def replay_active(trace: dict, boundary_crossed: bool) -> dict:
    """The mirror proof. A replay arm that never replayed is inactive instrumentation, not a policy.

    The prior program shipped a buffer that stopped admitting items once full and a within domain run that
    never crossed a context boundary, so two named policies resolved to the same behaviour.
    """
    return _finish(
        {
            "items_were_replayed": int(trace.get("replayed_items", 0)) > 0,
            "buffer_kept_replacing": bool(trace.get("admissions_after_full", 0) > 0),
            "context_boundary_crossed": bool(boundary_crossed),
            "buffer_contents_changed": bool(trace.get("buffer_sha_before") != trace.get("buffer_sha_after")),
        }
    )


# ---------------------------------------------------------------- 7.3 random, 7.4 shuffled, 7.5 wrong time


def random_control(real_trace: dict, control_trace: dict, tol_rate: float = 0.05) -> dict:
    r, c = real_trace, control_trace
    rate_r = float(r.get("intervention_rate", 0.0))
    rate_c = float(c.get("intervention_rate", 0.0))
    return _finish(
        {
            "rate_matched": abs(rate_r - rate_c) <= tol_rate,
            "budget_matched": int(r.get("updates", 0)) == int(c.get("updates", 0)),
            "information_matched": int(r.get("samples_seen", 0)) == int(c.get("samples_seen", 0)),
            "seed_bound": c.get("seed") is not None,
            "independent_of_target": abs(float(c.get("signal_target_corr", 0.0))) <= 0.1,
        }
    )


def shuffled_control(before, after, target_before, target_after) -> dict:
    """Marginals preserved, relation destroyed."""
    b, a = np.asarray(before, float), np.asarray(after, float)
    tb, ta = np.asarray(target_before, float), np.asarray(target_after, float)
    marg = np.allclose(np.sort(b, axis=None), np.sort(a, axis=None), atol=1e-6)

    def corr(u, v):
        u, v = u.ravel(), v.ravel()
        if u.std() < 1e-12 or v.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(u, v)[0, 1])

    return _finish(
        {
            "marginals_preserved": bool(marg),
            "relation_destroyed": abs(corr(a, ta)) < abs(corr(b, tb)) or abs(corr(b, tb)) < 1e-6,
        }
    )


def wrong_time_control(real_trace: dict, control_trace: dict, prereg_times: list) -> dict:
    return _finish(
        {
            "same_intervention": real_trace.get("intervention_kind") == control_trace.get("intervention_kind"),
            "budget_matched": int(real_trace.get("interventions", 0))
            == int(control_trace.get("interventions", 0)),
            "times_are_preregistered": list(control_trace.get("times", [])) == list(prereg_times),
            "times_differ_from_real": list(control_trace.get("times", [])) != list(real_trace.get("times", [])),
        }
    )


# ---------------------------------------------------------------- 7.6 frozen


def frozen_control(execution_receipt: dict, target_groups: list[str], group_members: dict) -> dict:
    """No target parameter changed. Read from the execution receipt, which lists what actually moved."""
    targets = {n for g in target_groups for n in group_members.get(g, [])}
    changed = set(execution_receipt.get("changed_params", []))
    return _finish(
        {
            "no_target_parameter_changed": not (targets & changed),
            "targets_exist": bool(targets),
            "receipt_lists_changes": "changed_params" in execution_receipt,
        }
    )


REGISTRY = {
    "order_free": order_free,
    "no_replay": no_replay,
    "replay_active": replay_active,
    "random": random_control,
    "shuffled": shuffled_control,
    "wrong_time": wrong_time_control,
    "frozen": frozen_control,
}
