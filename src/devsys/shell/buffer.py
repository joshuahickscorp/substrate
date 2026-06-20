"""Latent hippocampus: an episodic replay buffer over frozen-encoder latents. Because the
encoder is frozen, stored latents never go stale (the one place the frozen constraint
actively helps). Carries the three knobs the corpus separates:
  - prioritization (recency/surprise/reward/learning-progress -> a scalar priority)
  - a key-value index for content retrieval (faiss, with an exact brute-force fallback)
  - eviction (reservoir / fifo / priority)

Prioritized sampling is standard PER: P(i) ~ p_i^alpha, importance weight w_i = (N P(i))^-beta.
"""

from __future__ import annotations

import torch


class KVIndex:
    """Nearest-neighbor retrieval over keys. faiss if present and asked, else exact torch
    cdist (correct at toy scale; the API matches so the Studio can flip to faiss)."""

    def __init__(self, dim: int, kind: str = "faiss"):
        self.dim = dim
        self.kind = kind
        self._keys: torch.Tensor | None = None
        self._faiss = None
        if kind == "faiss":
            try:
                import faiss

                self._faiss = faiss.IndexFlatL2(dim)
            except Exception:
                self.kind = "brute"

    def rebuild(self, keys: torch.Tensor) -> None:
        self._keys = keys.detach().float().cpu()
        if self._faiss is not None:
            self._faiss.reset()
            if len(self._keys):
                self._faiss.add(self._keys.numpy())

    def search(self, query: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._keys is None or len(self._keys) == 0:
            raise RuntimeError("KVIndex empty; rebuild first")
        q = query.detach().float().cpu()
        k = min(k, len(self._keys))
        if self._faiss is not None:
            fd, fi = self._faiss.search(q.numpy(), k)
            return torch.from_numpy(fd), torch.from_numpy(fi)
        dist = torch.cdist(q, self._keys)  # [Q, N]
        td, ti = dist.topk(k, largest=False)
        return td, ti


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        dim: int,
        key_dim: int | None = None,
        prioritized: bool = True,
        alpha: float = 0.6,
        beta: float = 0.4,
        index: str = "faiss",
        eviction: str = "reservoir",
        seed: int = 0,
    ):
        self.capacity = capacity
        self.dim = dim
        self.key_dim = key_dim or dim
        self.prioritized = prioritized
        self.alpha, self.beta = alpha, beta
        self.eviction = eviction
        self.g = torch.Generator().manual_seed(seed)
        self.x = torch.zeros(capacity, dim)
        self.y = torch.zeros(capacity, dtype=torch.long)
        self.keys = torch.zeros(capacity, self.key_dim)
        self.prio = torch.zeros(capacity)
        self.size = 0
        self.pos = 0
        self.seen = 0
        self._index = KVIndex(self.key_dim, index)
        self._dirty = True

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        key: torch.Tensor | None = None,
        priority: torch.Tensor | float | None = None,
    ) -> None:
        x = x.detach().float()
        key = x if key is None else key.detach().float()
        n = x.shape[0]
        pmax = float(self.prio[: self.size].max()) if self.size else 1.0
        prio = self._as_prio(priority, n, pmax)
        for j in range(n):
            self.seen += 1  # count EVERY item seen (Algorithm-R t)
            slot = self._slot(float(prio[j]))
            if slot is None:
                continue
            self.x[slot], self.y[slot], self.keys[slot], self.prio[slot] = x[j], y[j], key[j], prio[j]
            self.size = max(self.size, slot + 1)
        self._dirty = True

    def _as_prio(self, priority, n, pmax) -> torch.Tensor:
        if priority is None:
            return torch.full((n,), pmax)
        if isinstance(priority, int | float):
            return torch.full((n,), float(priority))
        return priority.detach().float().reshape(-1)

    def _slot(self, prio: float) -> int | None:
        """Pick a write slot per eviction policy. Returns None to reject the item."""
        if self.size < self.capacity:
            s = self.pos
            self.pos = (self.pos + 1) % self.capacity
            return s
        if self.eviction == "fifo":
            s = self.pos
            self.pos = (self.pos + 1) % self.capacity
            return s
        if self.eviction == "priority":
            lo = int(self.prio.argmin())
            return lo if prio >= float(self.prio[lo]) else None
        # reservoir: replace a uniform slot with prob capacity/seen
        if torch.rand(1, generator=self.g).item() < self.capacity / max(1, self.seen):
            return int(torch.randint(0, self.capacity, (1,), generator=self.g))
        return None

    def sample(self, batch: int) -> dict:
        assert self.size > 0, "empty buffer"
        b = min(batch, self.size)
        if self.prioritized:
            p = self.prio[: self.size].clamp_min(1e-8) ** self.alpha
            probs = p / p.sum()
            idx = torch.multinomial(probs, b, replacement=b > self.size, generator=self.g)
            w = (self.size * probs[idx]).pow(-self.beta)
            w = w / w.max()
        else:
            idx = torch.randint(0, self.size, (b,), generator=self.g)
            w = torch.ones(b)
        return {"x": self.x[idx], "y": self.y[idx], "idx": idx, "is_weight": w}

    def update_priorities(self, idx: torch.Tensor, priorities: torch.Tensor) -> None:
        self.prio[idx] = priorities.detach().float().abs() + 1e-8

    def retrieve(self, query: torch.Tensor, k: int = 5) -> dict:
        if self._dirty:
            self._index.rebuild(self.keys[: self.size])
            self._dirty = False
        d, i = self._index.search(query, k)
        return {"dist": d, "idx": i, "x": self.x[i], "y": self.y[i]}
