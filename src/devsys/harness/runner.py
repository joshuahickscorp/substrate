"""Run one experiment from a composed config: resolve device, make a run dir, snapshot the
config, run, write the manifest with metrics. The single entry every script and the queue
go through."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from ..config import snapshot
from ..devices import resolve
from ..experiments import get_experiment
from ..logging_utils import RunManifest, get_logger, new_run_dir
from ..seeding import seed_everything

log = get_logger("runner")


def run_experiment(cfg: DictConfig, run_dir: Path | None = None) -> dict:
    eid = str(cfg.experiment.id)
    dev = resolve(str(cfg.device.kind))
    seed_everything(int(cfg.seed), bool(cfg.get("deterministic", True)))
    run_dir = run_dir or new_run_dir(eid)
    snapshot(cfg, run_dir / "config.yaml")
    log.info("run %s on %s (seed=%d) -> %s", eid, dev.note, int(cfg.seed), run_dir)

    exp = get_experiment(eid)
    man = RunManifest(name=eid, seed=int(cfg.seed), device=dev.kind, extra={"contract": exp.contract()})
    try:
        metrics = exp.run(cfg, dev, run_dir)
        man.metrics = metrics
        man.status = "ok"
    except Exception as ex:  # surface, never swallow
        man.status = f"error: {ex}"
        man.write(run_dir)
        raise
    import time

    man.finished = time.time()
    man.write(run_dir)
    return metrics
