from __future__ import annotations

from dataclasses import dataclass

import torch

from .latent_store import LatentStore


@dataclass
class Task:
    name: str
    x: torch.Tensor  # [N, D] latents
    y: torch.Tensor  # [N] integer labels
    xnext: torch.Tensor | None = None  # [N, D] next-latent target (forward dynamics)
    n_classes: int = 0  # global class count for head sizing
    task_id: int = 0


def _clusters(
    g: torch.Generator, n: int, dim: int, centers: torch.Tensor, sep: float
) -> tuple[torch.Tensor, torch.Tensor]:
    k = centers.shape[0]
    y = torch.randint(0, k, (n,), generator=g)
    x = centers[y] * sep + torch.randn(n, dim, generator=g) * 0.5
    return x, y


def make_task_stream(
    n_tasks: int = 5,
    dim: int = 1024,
    classes_per_task: int = 2,
    samples_per_task: int = 256,
    separation: float = 3.0,
    incremental: str = "class",  # class | task | domain
    forward_dynamics: bool = False,
    seed: int = 0,
) -> list[Task]:
    g = torch.Generator().manual_seed(seed)
    tasks: list[Task] = []
    total_classes = classes_per_task * (n_tasks if incremental == "class" else 1)
    for t in range(n_tasks):
        if incremental == "domain" or incremental == "task":
            centers = torch.randn(classes_per_task, dim, generator=g)
            label_base = 0
        else:  # class
            centers = torch.randn(classes_per_task, dim, generator=g)
            label_base = t * classes_per_task
        x, y_local = _clusters(g, samples_per_task, dim, centers, separation)
        y = y_local + label_base
        xnext = None
        if forward_dynamics:
            rot = torch.linalg.qr(torch.randn(dim, dim, generator=g))[0]
            xnext = x @ rot + 0.1 * separation
        tasks.append(Task(f"task{t}", x, y, xnext, n_classes=total_classes, task_id=t))
    return tasks


def noisy_tv_dataset(
    dim: int = 1024,
    n: int = 512,
    separation: float = 3.0,
    noise_scale: float = 5.0,
    seed: int = 0,
) -> dict[str, Task]:
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(2, dim, generator=g)
    xl, yl = _clusters(g, n, dim, centers, separation)
    rot = torch.linalg.qr(torch.randn(dim, dim, generator=g))[0]
    learnable = Task("learnable", xl, yl, xnext=xl @ rot, n_classes=2)
    xn = torch.randn(n, dim, generator=g) * separation
    noise = Task(
        "noise",
        xn,
        torch.zeros(n, dtype=torch.long),
        xnext=torch.randn(n, dim, generator=g) * noise_scale,
        n_classes=2,
    )
    return {"learnable": learnable, "noise": noise}


def stream_from_store(store: LatentStore, n_tasks: int, forward_dynamics: bool = False) -> list[Task]:
    x = store.latents().flatten(1)
    y = store.labels()
    if y is None:
        y = torch.zeros(len(store), dtype=torch.long)
    classes = sorted(set(y.tolist()))
    per = max(1, len(classes) // n_tasks)
    tasks = []
    for t in range(n_tasks):
        cls = set(classes[t * per : (t + 1) * per]) or {classes[-1]}
        mask = torch.tensor([c in cls for c in y.tolist()])
        xt = x[mask]
        tasks.append(
            Task(f"task{t}", xt, y[mask], xt if forward_dynamics else None, n_classes=len(classes), task_id=t)
        )
    return tasks
