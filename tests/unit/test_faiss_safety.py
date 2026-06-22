"""faiss + torch segfault guard (Apple Silicon). A buffer asked for the faiss index must never
crash: KVIndex probes faiss.search safety in a subprocess and falls back to exact brute force.
Correctness (nearest neighbor) holds either way."""

import torch

from devsys.shell.buffer import KVIndex, ReplayBuffer, faiss_search_safe


def test_faiss_probe_returns_bool_and_caches():
    a = faiss_search_safe()
    assert isinstance(a, bool)
    assert faiss_search_safe() is a  # cached


def test_faiss_index_never_crashes_falls_back_if_unsafe():
    idx = KVIndex(8, "faiss")  # may downgrade to brute if faiss.search is unsafe here
    assert idx.kind in ("faiss", "brute")
    idx.rebuild(torch.eye(8))
    d, i = idx.search(torch.eye(8)[3:4], k=1)
    assert int(i[0, 0]) == 3  # exact nearest neighbor regardless of backend


def test_buffer_retrieve_with_faiss_request_is_safe():
    b = ReplayBuffer(capacity=20, dim=8, index="faiss", eviction="fifo")
    b.add(torch.eye(8).repeat(3, 1)[:20], torch.arange(20) % 8)
    r = b.retrieve(torch.eye(8)[2:3], k=1)
    assert torch.allclose(b.x[r["idx"][0, 0]], torch.eye(8)[2], atol=1e-5)


def test_default_index_is_brute():
    b = ReplayBuffer(capacity=4, dim=4)
    assert b._index.kind == "brute"  # safe default, no probe needed
