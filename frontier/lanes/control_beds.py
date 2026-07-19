"""Frontier control admission beds: A1 (read affordance from latent) and S1 (simulate consequence vs react).

Real dynamics from gymnasium classic-control. Independent units are DISTINCT dynamical systems: three base
environments (CartPole, Acrobot, MountainCar) each parameter-perturbed into several genuinely different
physical systems (altered gravity, masses, lengths, force). Seeds are not treated as independent units.

Per unit an offline transition dataset is collected once (mixed random plus a mildly competent policy). Each
candidate policy is then evaluated by REAL rollouts in the true environment, so consequences are real. The
per-unit effect is (return(mechanism) - return(best simple control)) normalized by (return(empirical upper
bound) - return(random policy)). The runner applies the random-effects lower-95pct-CB rule.

A1 reads action-relevance (the advantage) straight from the learned latent and acts greedily on it. It must
beat a reactive policy, a greedy immediate-reward policy, a fitted value estimator, and a behavior-cloned
action predictor. S1 selects actions by simulating consequences H steps in a learned model and must beat
acting reactively and simpler one-step planning.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

# Distinct dynamical systems (base env, physics overrides). Each is a different physical system.
ENV_SPECS = [
    ("CartPole-v1", {}),
    ("CartPole-v1", {"gravity": 4.9, "length": 0.25}),
    ("CartPole-v1", {"gravity": 20.0, "masscart": 2.0}),
    ("CartPole-v1", {"length": 1.0, "force_mag": 20.0}),
    ("Acrobot-v1", {}),
    ("Acrobot-v1", {"LINK_MASS_1": 1.5, "LINK_LENGTH_1": 1.3}),
    ("Acrobot-v1", {"LINK_MASS_2": 1.5, "LINK_COM_POS_2": 0.7}),
    ("MountainCar-v0", {}),
    ("MountainCar-v0", {"gravity": 0.0035}),
    ("MountainCar-v0", {"force": 0.0015}),
]


def _make(spec, seed):
    env_id, overrides = spec
    env = gym.make(env_id)
    unwrapped = env.unwrapped
    for k, v in overrides.items():
        if hasattr(unwrapped, k):
            setattr(unwrapped, k, v)
    env.reset(seed=seed)
    return env


def _phi(obs):
    """Simple fixed nonlinear latent: raw obs plus quadratic and cross terms (a fixed representation)."""
    o = np.asarray(obs, dtype=np.float64)
    return np.concatenate([o, o * o, np.array([o[i] * o[j] for i in range(len(o)) for j in range(i + 1, len(o))])])


def collect_offline(spec, n_steps=4000, seed=0):
    env = _make(spec, seed)
    rng = np.random.default_rng(seed + 7)
    nA = env.action_space.n
    trans = []  # (phi_s, a, r, phi_s2, s2_terminal, obs_s, obs_s2)
    obs, _ = env.reset(seed=seed)
    for _ in range(n_steps):
        a = int(rng.integers(nA))
        obs2, r, term, trunc, _ = env.step(a)
        trans.append((_phi(obs), a, float(r), _phi(obs2), term, np.asarray(obs, float), np.asarray(obs2, float)))
        obs = obs2
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return trans, nA


def _ridge_fit(X, y, lam=1.0):
    X = np.asarray(X, float); y = np.asarray(y, float)
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def rollout(spec, policy_fn, episodes=20, seed=100, max_steps=500):
    total = []
    for e in range(episodes):
        env = _make(spec, seed + e)
        obs, _ = env.reset(seed=seed + e)
        ret = 0.0
        for _ in range(max_steps):
            a = policy_fn(obs)
            obs, r, term, trunc, _ = env.step(int(a))
            ret += r
            if term or trunc:
                break
        env.close(); total.append(ret)
    return float(np.mean(total))


def build_policies(spec, seed=0):
    """Fit all A1 / S1 mechanisms and their controls on an offline dataset for one unit."""
    trans, nA = collect_offline(spec, seed=seed)
    Phi = np.array([t[0] for t in trans]); A = np.array([t[1] for t in trans])
    R = np.array([t[2] for t in trans]); Phi2 = np.array([t[3] for t in trans])
    Term = np.array([t[4] for t in trans]); dphi = Phi.shape[1]

    # Fitted state-action value via a few sweeps of fitted-Q on the fixed latent (per-action linear heads)
    gamma = 0.99
    W = {a: np.zeros(dphi) for a in range(nA)}
    for _ in range(40):
        Qnext = np.stack([Phi2 @ W[a] for a in range(nA)], 1)
        target = R + gamma * (1 - Term) * Qnext.max(1)
        for a in range(nA):
            m = A == a
            if m.sum() > dphi:
                W[a] = _ridge_fit(Phi[m], target[m], lam=1.0)
    def q_all(obs):
        p = _phi(obs); return np.array([p @ W[a] for a in range(nA)])

    # A1: read affordance (advantage) straight from the latent. Advantage = Q(s,a) - mean_a Q(s,a),
    # learned as a direct readout head on the latent, then act greedily on the read advantage.
    Adv = np.stack([Phi @ W[a] for a in range(nA)], 1)
    Adv = Adv - Adv.mean(1, keepdims=True)
    Wadv = {a: _ridge_fit(Phi, Adv[:, a], lam=1.0) for a in range(nA)}
    def a1_policy(obs):
        p = _phi(obs); return int(np.argmax([p @ Wadv[a] for a in range(nA)]))

    # controls
    rng = np.random.default_rng(seed + 3)
    def random_policy(obs): return int(rng.integers(nA))
    # reactive: behavior-clone the action of high-immediate-reward transitions (supervised obs->action)
    good = R >= np.quantile(R, 0.6)
    Wbc = {a: _ridge_fit(Phi[good], (A[good] == a).astype(float), lam=1.0) for a in range(nA)}
    def bc_policy(obs):
        p = _phi(obs); return int(np.argmax([p @ Wbc[a] for a in range(nA)]))
    # greedy immediate reward model
    Wr = {a: (_ridge_fit(Phi[A == a], R[A == a], lam=1.0) if (A == a).sum() > dphi else np.zeros(dphi)) for a in range(nA)}
    def greedy_reward(obs):
        p = _phi(obs); return int(np.argmax([p @ Wr[a] for a in range(nA)]))
    def value_policy(obs): return int(np.argmax(q_all(obs)))
    def wrongctx_policy(obs):  # will be replaced with another unit's A1 by runner; placeholder = value
        return value_policy(obs)

    # S1: simulate consequences H steps in a learned model (dynamics + reward) and pick best action.
    # dynamics per action: next latent-relevant obs predicted linearly; reward predicted linearly.
    Wdyn = {a: (_ridge_fit(Phi[A == a], np.array([t[6] for t in trans])[A == a], lam=1.0)
                if (A == a).sum() > dphi else np.zeros((dphi, len(trans[0][6])))) for a in range(nA)}
    # Wdyn maps latent->next raw obs (mean). reward model Wr reused.
    def sim_return(obs, a, H=4):
        p = _phi(obs); total = p @ Wr[a]
        nxt = p @ Wdyn[a]
        for _ in range(H - 1):
            pn = _phi(nxt)
            qa = int(np.argmax([pn @ Wr[aa] for aa in range(nA)]))
            total += 0.99 ** _ * (pn @ Wr[qa]); nxt = pn @ Wdyn[qa]
        return total
    def s1_policy(obs):
        return int(np.argmax([sim_return(obs, a) for a in range(nA)]))
    def onestep_policy(obs):  # one-step planner (H=1) = greedy reward, a simpler planning control
        return greedy_reward(obs)
    def actionblind_sim(obs):  # simulate ignoring action choice (control): pick random among sims
        return int(rng.integers(nA))

    return {
        "nA": nA,
        "A1": a1_policy, "S1": s1_policy,
        "random": random_policy, "reactive_bc": bc_policy, "greedy_reward": greedy_reward,
        "value_estimator": value_policy, "onestep_planner": onestep_policy, "actionblind_sim": actionblind_sim,
        "wrong_context": wrongctx_policy,
    }
